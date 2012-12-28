/*
vim:ts=4:sw=4:
 * Mini-Toolbox
 *
 * David Harrison
 * University of California, Berkeley
 * 1988, 1989
 *
 * This file contains routines which implement simple display widgets
 * which can be used to construct simple dialog boxes.
 * A mini-toolbox has been written here (overkill but I didn't
 * want to use any of the standards yet -- they are too unstable).
 */

#ifdef XGRAPH
#include "../config.h"
#define _MAIN_C
#endif

#include <stdio.h>
#include <math.h>
#include <ctype.h>

#ifndef XGRAPH
#	define NO_ENUM_BOOLEAN
#	include "local/mxt.h"
#	include "local/Macros.h"
#endif

IDENTIFY("XGraph Mini Toolkit for X11");

#include "xgALLOCA.h"

#include "xgerrno.h"

static int xtb_errno= 0;
static int xtb_dispatch_modal= 0;

#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/keysym.h>
#include <X11/cursorfont.h>
#include <X11/Xatom.h>
#include "xtb.h"

/* filepointer defined in xgraph.c: */
extern FILE *StdErr;

extern char *index(), *d2str( double, char*, char*);

#define MAXKEYS   	10

#ifndef MAX
#define MAX(a,b)				(((a)>(b))?(a):(b))
#endif
#ifndef MIN
#define MIN(a,b)				(((a)<(b))?(a):(b))
#endif

#define SWAP(a,b,type)	{type c= (a); (a)= (b); (b)= c;}

#define CLIP(var,low,high)	if((var)<(low)){\
		(var)=(low);\
	}else if((var)>(high)){\
		(var)=(high);}

#ifndef CLIP_EXPR_CAST
/* A safe casting/clipping macro.	*/
#	define CLIP_EXPR_CAST(ttype,var,stype,expr,low,high)	{stype val=(expr); if(val<(low)){\
		(var)=(ttype)(low);\
	}else if(val>(high)){\
		(var)=(ttype)(high);\
	}else{\
		(var)=(ttype)val;}}
#endif

#undef abort

#ifdef XGRAPH
#	include "../ux11/ux11.h"
#	include "../copyright.h"
#	include "../NaN.h"
#else
#	include "local/ux11.h"
#endif

#ifndef XGRAPH
void _xfree(void *x, char *file, int lineno )
{
	if( x ){
#if DEBUG == 2
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
#endif
		free(x);
// 		x=NULL;
	}
}
#	define xfree(x) {_xfree((void*)(x),__FILE__,__LINE__);(x)=NULL;}
#else
#include "xfree.h"
#endif


typedef struct h_info {
	Window window;
	FNPTR(func, xtb_hret, (XEvent *evt, xtb_data info, Window parent)); /* Function to call */
	xtb_frame *frame;	/* frame this one belongs to    */
	union{
		xtb_registry_info *basic;
		xtb_data info;		/* Additional info  */
	} data;
} h_info;

typedef struct to_info {
	xtb_frame *frame;
	xtb_data val;
	FNPTR( func, xtb_hret, (Window win, xtb_data val) );
					/* Function to call   */
	FrameType type;
	char *text;   				/* Text to display */
	int pos, len;
	XFontStruct **ft, **ft2;		/* Font(s) to use     */
	unsigned long norm_pix, light_pix, back_pix;
	int entered, id;
	char **text_return;			/* to return *text to a parent widget with a number of text widgets	*/
	int *text_id_return;
} to_info;

typedef struct ti_info {
	xtb_frame *frame;
	xtb_data val;		/* User info          */
	FNPTR( func, xtb_hret, (Window win, int ch, char *textcopy, xtb_data val) );
					/* Function to call   */
	FrameType type;
	int maxlen, maxwidth;			/* Maximum characters */
	int curidx;   		/* Current index pos  */
	int curxval;		/* Current draw loc   */
	int curlen;
	int startidx, startxval, ti_crsp;
	int button_pressed, button_handled, button_xpos;
	char *text; /* Current text array */
	XFontStruct *font, *alt_font;
	int max_ascent,
		line_y, line_w, cursor_height;   	/* Entry/Exit line    */
	int enter_flag, _entered,
		clear_flag, focus_flag, rich_text;
	unsigned long norm_pix, light_pix, middle_pix, back_pix;
	unsigned long _norm_pix, _back_pix;
} ti_info;

typedef struct ti2_info {
	xtb_frame *frame;
	xtb_data val;		/* User data          */
	  /* stub: */
	FNPTR( func, xtb_hret, (Window win, int ch, char *textcopy, xtb_data val) );
	FrameType type;
	Window main_win;		/* Main button row    */
	unsigned long norm_pix, light_pix, middle_pix, back_pix;
	Window subwin[2];		/* sub windows     */
	ti_info *ti_info;
} ti2_info;

#define TI_HPAD 2
#define TI_VPAD 2
#define TI_LPAD 3
#define TI_BRDR 2
#define TI_CRSP 1
static int text_width( XFontStruct *font, char *str, int len, ti_info *info);

#define TIDRAW_DECL	ti_draw
static void TIDRAW_DECL( Window win, struct ti_info *ri, int c_flag);

typedef struct b_info {
	xtb_frame *frame;
	xtb_data val;		/* User defined info */
	FNPTR( func, xtb_hret, (Window win, int state, xtb_data val) );
				/* Function to call */
	FrameType type;
	char *text;   		/* Text of button   */
	XFontStruct *font, *alt_font;
	int x_text_offset, y_text_offset;
	int flag, flag2;			/* State of button  */
	unsigned long norm_pix, lighter_pix, light_pix, middle_pix, back_pix;
	int line_y, line_w;   	/* Entry/Exit line  */
	int pos, enter_flag, focus_flag;
} b_info;

typedef struct sr_info {
	xtb_frame *frame;
	xtb_data val;		/* User info          */
	FNPTR( func, xtb_hret, (Window win, int pos, double value, xtb_data val) );
					/* Function to call   */
	FrameType type;
	int minposition, maxposition;
	int prev_position, position;
	double minvalue, maxvalue, delta;
	double _minvalue, _maxvalue;
	double prev_value, value;
	int redraw, button_pressed, button_handled;
	char text[64];
	XFontStruct *font, *alt_font;
	int line_x, line_y, line_w, slide_x, slide_y, slide_w, slide_h,
		lr_w, lr_h;
	int clear_flag, int_flag, vert_flag;
	Window slide_, slide, less, more;
	unsigned long norm_pix, light_pix, middle_pix, back_pix;
} sr_info;

typedef struct err_info {
	xtb_frame *frame;
	xtb_data val;
	FNPTR( func, xtb_hret, (Window win, xtb_data val) );
					/* Function to call   */
	FrameType type;
	Window title;
	Window contbtn, cancbtn, ti_win, hlp_win, hlp_win2, hlp_win3;
	xtb_frame *contfr, *cancfr, *ti_fr, *hlp_fr, *hlp_fr2, *hlp_fr3;
	int cont_ypos, canc_ypos;
	int num_lines;
	int alloc_lines;
	xtb_frame *subframe;
	int subframes;
	int *width;
	int *ypos, *xcenter;
	char *selected_text;
	int selected_id;
	int ti_maxlen;
	FNPTR( hlp_fnc, xtb_hret, (Window win, int st, xtb_data val) );
	FNPTR( hlp_fnc2, xtb_hret, (Window win, int st, xtb_data val) );
	FNPTR( hlp_fnc3, xtb_hret, (Window win, int st, xtb_data val) );
} err_info;

#define E_LINES 9
#define E_VPAD	3
#define E_HPAD	3
#define E_INTER 1

#define SRDRAW_DECL	sr_draw
static void SRDRAW_DECL( Window win, sr_info *ri, int c_flag);

Display *t_disp= NULL;		/* Display          */
static int t_scrn;		/* Screen           */
#ifdef XGRAPH
extern Visual *vis;
extern Colormap cmap;
int t_screen;
extern int screen;
extern int depth;
#else
Visual *xtb_vis;
#	define vis xtb_vis
Colormap xtb_cmap;
#	define cmap	xtb_cmap
int xtb_vismap_type;
int t_screen;
int xtb_depth;
#	define depth	xtb_depth
#endif
XColor fg_color, bg_color;

static XEvent evt;

Pixel xtb_norm_pix; /* Foreground color */
Pixel xtb_back_pix; /* Background color */
Pixel xtb_middle_pix, xtb_light_pix, xtb_lighter_pix;
Pixel xtb_white_pix, xtb_black_pix, xtb_LLgray_pix, xtb_Lgray_pix, xtb_Mgray_pix, xtb_red_pix;
XColor RGBmiddle, RGBlight, RGBlighter;
Boolean RGBmiddle_set= False, RGBlight_set= False, RGBlighter_set= False;

static XFontStruct *norm_font, *greek_font; /* Normal font      */
static int max_ascent, max_descent;

#ifndef __STDC__
	extern char *malloc(), *calloc();
#endif
#define malloc(s)	calloc(1,(s))

/* #define STRDUP(str) (strcpy(malloc((unsigned) (strlen(str)+1)), (str)))	*/
#define STRDUP(str)	strdup(str)
#ifndef strcpy
	extern char *strcpy();
#endif
#ifndef strdup
	extern char *strdup();
#endif
extern void free();

extern int debugFlag, debugLevel;

static GC t_gc = (GC) 0;

static Cursor br_cursor= 0, bt_cursor= 0, ti_cursor= 0, to_cursor= 0, bk_cursor= 0, err_cursor= 0;
static Cursor sr_hcurs= 0, sr_hcursl= 0, sr_hcursr= 0;
static Cursor sr_vcurs= 0, sr_vcursu= 0, sr_vcursd= 0;

static Boolean SetParentBackground= False;

static xtb_hret ti_h(), bt_h(), sr_h(), to_h();
static void bt_draw(), to_draw();

extern int RemoteConnection;

  /* xtb_XSync() does the same as XSync() (3X11). The difference is that
   \ it sends an EnterNotify event to the window containing the pointer when
   \ discard==True. This allows us to discard all events in the eventinputqueue,
   \ but still "activate" (not operate!) the button/inputfield the pointer is
   \ over.
   */
int xtb_XSync( Display *disp, Bool discard )
{ int r=0;
#ifndef DONT_ALLOW_ENTER_RESEND
	if( discard ){
	  XEvent evt;
	  Window root_win, win_win, twin= None, pwin= 0, focus;
	  int n= 0, root_x, root_y, win_x, win_y, mask;
      int revert;
#ifdef XGRAPH
	  extern int Exitting;

	  if( !Exitting ){
#endif
		XGetInputFocus( disp, &focus, &revert);
		 /* Getting the window actually containing the pointer proves to be rather
		  \ tedious. We start out by asking the pointer's place on the specified
		  \ screen's rootwindow, and get the childwindow containing the pointer in
		  \ win_win. Maybe None, if the pointer is in the specified window (= rootwindow).
		  */
		XQueryPointer( disp, RootWindow(disp, t_screen), &root_win, &win_win,
			&root_x, &root_y, &win_x, &win_y, &mask
		);
		  /* As long as the returned child containing the window (win_win) is not None,
		   \ and as long as the last returned child is not equal to the new child (probably
		   \ never happens, as apparently None is returned in that case), we set the
		   \ eventual targetwindow twin to win_win, and query the pointer's position relative
		   \ to that window. Thus, we'll eventually get to the low(li)est child containing
		   \ the pointer. This one (twin) will receive the EnterNotify event.
		   */
		while( win_win!= None && win_win!= twin ){
			pwin= twin;
			twin= win_win;
			XQueryPointer( disp, twin, &root_win, &win_win,
				&root_x, &root_y, &win_x, &win_y, &mask
			);
			n+= 1;
		}
		mask= xtb_Mod2toMod1(mask);
		r= XSync( disp, discard );
		evt.type= EnterNotify;
		evt.xcrossing.display= disp;
		evt.xcrossing.root= root_win;
		evt.xcrossing.x= win_x;
		evt.xcrossing.y= win_y;
		evt.xcrossing.x_root= root_x;
		evt.xcrossing.y_root= root_y;
		evt.xcrossing.mode= NotifyNormal;
		evt.xcrossing.focus= (focus== twin);
		evt.xcrossing.state= mask;
		if( twin!= None ){
			if( debugFlag ){
				fprintf( StdErr,
					"xtb::xtb_XSync(%s): Sending EnterNotify to window %d (p=%d,c=%d,#=%d) @(%d,%d), focus=%d\n",
					(discard)? "True" : "False",
					twin, pwin, win_win, n, win_x, win_y, focus
				);
			}
			evt.xcrossing.window= twin;
			XSendEvent( disp, PointerWindow, 0, EnterWindowMask|LeaveWindowMask, &evt);
		}
		else{
			if( debugFlag ){
				fprintf( StdErr, "xtb::xtb_XSync(%s): Can't find window that contains pointer, sending to PointerWindow\n",
					(discard)? "True" : "False"
				);
			}
			evt.xcrossing.window= focus;
			XSendEvent( disp, PointerWindow, 0, EnterWindowMask|LeaveWindowMask, &evt);
		}
#ifdef XGRAPH
	  }
	  else{
		  r= XSync( disp, discard );
	  }
#endif
	}
	else if( !RemoteConnection ){
		r= XSync( disp, discard );
	}
#else
	if( discard || !RemoteConnection ){
		r= XSync( disp, discard );
	}
#endif
	return(r);
}

double xtb_PsychoMetric_Gray( XColor *rgb )
{
/* 	return( 0.3 * rgb->red+ 0.59 * rgb->green+ 0.11 * rgb->blue );	*/
	  /* According to the CIE-XYZ 1931 specification, from http://www1.tip.nl/~t876506/ColorDesign.html#gry : 	*/
	return( 0.298954 * rgb->red+ 0.586434 * rgb->green+ 0.114612 * rgb->blue );
}

int xtb_ti_serial= 0,
	xtb_ti2_serial= 0,
	xtb_to_serial= 0,
	xtb_br_serial= 0,
	xtb_sr_serial= 0,
	xtb_bk_serial= 0;

char *xtb_textfilter_buf= NULL;
int xtb_textfilter_buflen;
static int textalloc= False;

extern int ButtonContrast;

void xtb_InterpolateCRange( Pixel foreground, Pixel background, Pixel *middle, Pixel *light, Pixel *lighter )
{ unsigned short middlegray, lightgray, lightergray;
  char *colspec;
	if( depth>= 2 ){
	   XColor RGBnorm, RGBback;
	   double middle_scale= (1.0-ButtonContrast/65535.0);
	   double light_scale= (1.0- 0.5*ButtonContrast/65535.0);
	   double lighter_scale= (1.0- 0.38*ButtonContrast/65535.0);
		if( !rd_flag( "xtbGray") ){
		  /* RGB 192,208,152 == rgbi:0.777768/0.825391/0.619043	*/
		  /* RGB 190,210,155 == rgb:be/d2/9b (rgbi:0.746029/0.825391/0.619043)	*/
		  double gred= 190.0/255.0*65535, ggreen= 210.0/255.0*65535, gblue= 155.0/255.0*65535;
			if( rd_str( "xtbBaseColour") ){
			  extern char *def_str;
			  XColor spec;
				colspec= def_str;
				if( XParseColor( t_disp, cmap, colspec, &spec ) ){
					if( debugFlag && debugLevel ){
						fprintf( StdErr, "xtb_init(): xtbBaseColour='%s': RGB=(%u,%u,%u)=%lu\n", colspec,
							spec.red, spec.green, spec.blue, spec.pixel
						);
					}
					gred= spec.red, ggreen= spec.green, gblue= spec.blue;
				}
				else{
					fprintf( StdErr, "xtb_init(): can't parse xtbBaseColour='%s'\n", colspec );
				}
			}
			CLIP_EXPR_CAST( unsigned short, RGBmiddle.red, double, gred* middle_scale/ light_scale, 0, 65535);
			CLIP_EXPR_CAST( unsigned short, RGBmiddle.green, double, ggreen* middle_scale/ light_scale, 0, 65535);
			CLIP_EXPR_CAST( unsigned short, RGBmiddle.blue, double, gblue* middle_scale/ light_scale, 0, 65535);
/* 			RGBmiddle.red= (int) (0.24* 65535);	*/
/* 			RGBmiddle.green= (int) (0.29* 65535);	*/
/* 			RGBmiddle.blue= (int) (0.13* 65535);	*/
			CLIP_EXPR_CAST( unsigned short, RGBlight.red, double, gred, 0, 65535);
			CLIP_EXPR_CAST( unsigned short, RGBlight.green, double, ggreen, 0, 65535);
			CLIP_EXPR_CAST( unsigned short, RGBlight.blue, double, gblue, 0, 65535);
			CLIP_EXPR_CAST( unsigned short, RGBlighter.red, double, gred* lighter_scale/ light_scale, 0, 65535);
			CLIP_EXPR_CAST( unsigned short, RGBlighter.green, double, ggreen* lighter_scale/ light_scale, 0, 65535);
			CLIP_EXPR_CAST( unsigned short, RGBlighter.blue, double, gblue* lighter_scale/ light_scale, 0, 65535);
			middlegray= (unsigned short) xtb_PsychoMetric_Gray( &RGBmiddle );
			lightgray= (unsigned short) xtb_PsychoMetric_Gray( &RGBlight );
			lightergray= (unsigned short) xtb_PsychoMetric_Gray( &RGBlighter );
		}
		else{
			RGBnorm.pixel= foreground;
			RGBback.pixel= background;
			XQueryColor( t_disp, cmap, &RGBnorm );
			XQueryColor( t_disp, cmap, &RGBback );
			middlegray= (unsigned short) (( xtb_PsychoMetric_Gray( &RGBnorm ) + xtb_PsychoMetric_Gray( &RGBback ))* middle_scale);
			lightgray= (unsigned short) (( xtb_PsychoMetric_Gray( &RGBnorm ) + xtb_PsychoMetric_Gray( &RGBback ))* light_scale );
			lightergray= (unsigned short) (( xtb_PsychoMetric_Gray( &RGBnorm ) + xtb_PsychoMetric_Gray( &RGBback ))* lighter_scale );
			RGBmiddle.red= RGBmiddle.green= RGBmiddle.blue= middlegray;
			RGBlight.red= RGBlight.green= RGBlight.blue= lightgray;
			RGBlighter.red= RGBlighter.green= RGBlighter.blue= lightergray;
		}
		if( debugFlag ){
			fprintf( StdErr, "xtb_InterpolateCRange(): foreground is [#%lu,rgbi:%g/%g/%g]==%g\n",
				RGBnorm.pixel, RGBnorm.red/65535.0, RGBnorm.green/65535.0, RGBnorm.blue/65535.0,
				xtb_PsychoMetric_Gray( &RGBnorm )/65535.0
			);
			fprintf( StdErr, "xtb_InterpolateCRange(): background is [#%lu,rgbi:%g/%g/%g]==%g\n",
				RGBback.pixel, RGBback.red/65535.0, RGBback.green/65535.0, RGBback.blue/65535.0,
				xtb_PsychoMetric_Gray( &RGBback )/65535.0
			);
			fprintf( StdErr, "xtb_InterpolateCRange(): looking for middle gray %u, lightgray %u, lightergray %u\n",
				middlegray, lightgray, lightergray
			);
		}
		RGBmiddle.flags= DoRed|DoGreen|DoBlue;
		RGBlight.flags= DoRed|DoGreen|DoBlue;
		RGBlighter.flags= DoRed|DoGreen|DoBlue;
		if( !XAllocColor( t_disp, cmap, &RGBmiddle ) ){
			fprintf( StdErr, "xtb_InterpolateCRange(): can't allocate middle gray [rgbi:%g/%g/%g]\n",
				RGBmiddle.red/65535.0, RGBmiddle.green/65535.0, RGBmiddle.blue/65535.0
			);
			*middle= ((foreground + background)* middle_scale);
		}
		else{
			*middle= RGBmiddle.pixel;
			RGBmiddle_set= True;
			if( debugFlag ){
				fprintf( StdErr, "xtb_InterpolateCRange(): middle gray is [#%lu,rgbi:%g/%g/%g]==%g\n",
					RGBmiddle.pixel, RGBmiddle.red/65535.0, RGBmiddle.green/65535.0, RGBmiddle.blue/65535.0,
					xtb_PsychoMetric_Gray( &RGBmiddle )/65535.0
				);
			}
		}
		if( !XAllocColor( t_disp, cmap, &RGBlight ) ){
			fprintf( StdErr, "xtb_InterpolateCRange(): can't allocate light gray [rgbi:%g/%g/%g]\n",
				RGBlight.red/65535.0, RGBlight.green/65535.0, RGBlight.blue/65535.0
			);
			*light= ((foreground + background)* light_scale );
		}
		else{
			*light= RGBlight.pixel;
			RGBlight_set= True;
			if( debugFlag ){
				fprintf( StdErr, "xtb_InterpolateCRange(): light gray is [#%lu,rgbi:%g/%g/%g]==%g\n",
					RGBlight.pixel, RGBlight.red/65535.0, RGBlight.green/65535.0, RGBlight.blue/65535.0,
					xtb_PsychoMetric_Gray( &RGBlight )
				);
			}
		}
		if( !XAllocColor( t_disp, cmap, &RGBlighter ) ){
			fprintf( StdErr, "xtb_InterpolateCRange(): can't allocate lighter gray [rgbi:%g/%g/%g]\n",
				RGBlighter.red/65535.0, RGBlighter.green/65535.0, RGBlighter.blue/65535.0
			);
			*lighter= ((foreground + background)* lighter_scale );
		}
		else{
			*lighter= RGBlighter.pixel;
			RGBlighter_set= True;
			if( debugFlag ){
				fprintf( StdErr, "xtb_InterpolateCRange(): lighter gray is [#%lu,rgbi:%g/%g/%g]==%g\n",
					RGBlighter.pixel, RGBlighter.red/65535.0, RGBlighter.green/65535.0, RGBlighter.blue/65535.0,
					xtb_PsychoMetric_Gray( &RGBlighter )
				);
			}
		}
	}
	else{
		*middle= foreground;
		*light= background;
	}
}

Atom xtb_wm_delete_window;

#include "xtb_disabled_bg"
static Pixmap disabled_bg_light= 0;
static Pixmap disabled_bg_middle= 0;
int CygwinXorXmingServer = FALSE;

void xtb_init( Display *disp, int scrn, unsigned long foreground, unsigned long background, XFontStruct *font, XFontStruct *alt_font, int parent_background)
/*
 * Sets default parameters used by the mini-toolbox.
 */
{ static char called= 0;
/* #ifdef XGRAPH	*/
   extern int reverseFlag, reversePixel();
/* #endif	*/

	if( t_gc != (GC) 0 && t_disp ){
		XFreeGC( t_disp, t_gc );
	}
	t_disp = disp;
	t_scrn = scrn;
#ifndef XGRAPH
/* 	xtb_vismap_type= ux11_std_vismap(t_disp, &vis, &cmap, &t_screen, &depth, 0 );	*/
#else
	t_screen= screen;
#endif

/* #ifdef XGRAPH	*/

	if( reverseFlag ){
		ReversePixel( &foreground );
		ReversePixel( &background );
		GetColor("white", &xtb_black_pix );
		GetColor("black", &xtb_white_pix );
	}
	else{
		GetColor("black", &xtb_black_pix );
		GetColor("white", &xtb_white_pix );
	}
/* 	GetColor( "darkred", &xtb_red_pix );	*/
	GetColor( "#800000", &xtb_red_pix );
/* #endif	*/

	if( !called || disp!= t_disp ){
		xtb_wm_delete_window= XInternAtom( disp, "WM_DELETE_WINDOW", False );
	}

	 /* Some static variables must be (re)initialised to zero.
       \ This handles reopening a Display connection.
       */
	t_gc= (GC) 0;
	br_cursor= 0, bt_cursor= 0, ti_cursor= 0;
	sr_hcurs= 0, sr_hcursl= 0, sr_hcursr= 0;
	sr_vcurs= 0, sr_vcursu= 0, sr_vcursd= 0;

	xtb_InterpolateCRange( xtb_black_pix, xtb_white_pix, &xtb_Mgray_pix, &xtb_Lgray_pix, &xtb_LLgray_pix );
	xtb_norm_pix = foreground;
	xtb_back_pix = background;

	SetParentBackground= parent_background;

	xtb_InterpolateCRange( foreground, background, &xtb_middle_pix, &xtb_light_pix, &xtb_lighter_pix );

	norm_font = font;
	greek_font= alt_font;
	if( alt_font ){
		max_ascent= MAX( norm_font->ascent, greek_font->ascent );
		max_descent= MAX( norm_font->descent, greek_font->descent );
	}
	else{
		max_ascent= norm_font->ascent;
		max_descent= norm_font->descent;
	}

	xtb_ti_serial= 0;
	xtb_ti2_serial= 0;
	xtb_to_serial= 0;
	xtb_br_serial= 0;
	xtb_sr_serial= 0;
	xtb_bk_serial= 0;

	  /* 20030625 */
	if( !xtb_textfilter_buf ){
		if( !(xtb_textfilter_buf= calloc( (xtb_textfilter_buflen=1025), sizeof(char))) ){
			xtb_textfilter_buflen= 0;
			textalloc= False;
		}
		else{
			textalloc= 2;
		}
	}

	if( called ){
		if( disabled_bg_light!= None ){
			XFreePixmap( t_disp, disabled_bg_light );
		}
		if( disabled_bg_middle!= None ){
			XFreePixmap( t_disp, disabled_bg_middle );
		}
	}
	disabled_bg_light= XCreatePixmapFromBitmapData( t_disp, RootWindow(t_disp, t_screen),
		xtb_disabled_bg_bits, xtb_disabled_bg_width, xtb_disabled_bg_height,
		xtb_black_pix, xtb_light_pix, depth
	);
	disabled_bg_middle= XCreatePixmapFromBitmapData( t_disp, RootWindow(t_disp, t_screen),
		xtb_disabled_bg_bits, xtb_disabled_bg_width, xtb_disabled_bg_height,
		xtb_white_pix, xtb_middle_pix, depth
	);

	if( strcmp( ServerVendor(disp), "The Cygwin/X Project" ) == 0
	   || strcmp( ServerVendor(disp), "Colin Harrison" ) == 0
	){
		CygwinXorXmingServer = TRUE;
	}
	else{
		CygwinXorXmingServer = FALSE;
	}

	called= 1;
}

void xtb_close()
{
	if( RGBmiddle_set ){
		XFreeColors( t_disp, cmap, &xtb_middle_pix, 1, 0);
	}
	if( RGBlight_set ){
		XFreeColors( t_disp, cmap, &xtb_light_pix, 1, 0);
	}
}

int Boing(n)
int n;
{
	if( t_disp ){
		XBell( t_disp, n);
	}
	return( n);
}

int xtb_ProcessStyle(char *style, char *buf, int maxbuf)
/* Textual line style spec */ /* Returned buf            */ /* Maximum size of buffer  */
/*
 * Translates a textual specification for a line style into
 * an appropriate dash list for X11.  Returns the length.
 */
{
	int len, i;

//20101018: strncpy() shouldn't be called with overlapping buffers?!
// 	strncpy(buf, style, maxbuf-1);
	if( buf && style && maxbuf ){
	  int n = strlen(style);
		memmove(buf, style, sizeof(char) * MIN( (maxbuf-1), n ) );
		buf[maxbuf-1] = '\0';
	}
	else{
		buf[0] = '\0';
	}
/* 	len = strlen(buf);	*/
	len= MIN( strlen(buf), maxbuf );
	if( debugFlag && debugLevel ){
		fprintf( StdErr, "xtb_ProcessStyle(\"%s\",%d)= ",
			buf, maxbuf
		);
	}
	for (i = 0;   i < len;   i++) {
		if( buf[i]< '0' ){
			buf[i]= 0;
		}
		else if ((buf[i] >= '0') && (buf[i] <= '9')) {
			buf[i] = buf[i] - '0';
		}
		else if ((buf[i] >= 'a') && (buf[i] <= 'f')) {
			buf[i] = buf[i] - 'a' + 10;
		}
		else if ((buf[i] >= 'A') && (buf[i] <= 'F')) {
			buf[i] = buf[i] - 'A' + 10;
		}
		else{
			buf[i]= 15;
		}
		if( debugFlag && debugLevel ){
			fprintf( StdErr, "%d ", buf[i]);
		}
	}
	if( buf[0]== 0 && len== 1 ){
		if( debugFlag ){
			if( !debugLevel ){
				fprintf( StdErr, "xtb_ProcessStyle(\"%s\",%d) - illegal (empty) style: assuming solid line request\n",
					buf, maxbuf
				);
			}
			else{
				fprintf( StdErr, "- illegal (empty) style: assuming solid line request -" );
			}
		}
		len= 0;
	}
	if( debugFlag && debugLevel ){
		fprintf( StdErr, "l=%d\n", len );
	}
	return len;
}

GC xtb_set_gc( Window win, unsigned long fg, unsigned long bg, Font font )
/*
 * Sets and returns the fields listed above in a global graphics context.
 * If graphics context does not exist,  it is created.
 */
{
	XGCValues gcvals;
	unsigned long gcmask;

	gcvals.foreground = fg;
	gcvals.background = bg;
	gcvals.font = font;
	gcvals.line_style= LineSolid;
	gcvals.line_width= 0;
	gcmask = GCForeground | GCBackground | GCFont | GCLineStyle | GCLineWidth;
	if (t_gc == (GC) 0) {
		t_gc = XCreateGC(t_disp, win, gcmask, &gcvals);
	} else {
		XChangeGC(t_disp, t_gc, gcmask, &gcvals);
	}
	return t_gc;
}

static XContext h_context = (XContext) 0;

int xtb_unregister(win, info)
Window win;
xtb_registry_info **info;
/*
 * Removes `win' from the dialog association table.  `info' is
 * returned to allow the user to delete it (if desired).  Returns
 * a non-zero status if the association was found and properly deleted.
 */
{ struct h_info *hi;
  int error;

	if( !(error= XFindContext(t_disp, win, h_context, (caddr_t *) &hi))) {
		XDeleteContext(t_disp, win, h_context);
		if( info ){
			*info = hi->data.info;
		}
		xfree( hi);
		return 1;
	} else return 0;
}

void xtb_register_( xtb_frame *frame, Window win, xtb_hret (*func)(XEvent *, xtb_data, Window), xtb_registry_info *info)
/*
 * Associates the event handling function `func' with the window
 * `win'.  Additional information `info' will be passed to `func'.
 * The routine should return one of the return codes given above.
 */
{
	struct h_info *new_info;

	if (h_context == (XContext) 0) {
		h_context = XUniqueContext();
	}
	new_info = (struct h_info *) malloc(sizeof(struct h_info));
	new_info->window= win;
	new_info->func = func;
	new_info->data.info = info;
	new_info->frame = frame;
	XSaveContext(t_disp, win, h_context, (caddr_t) new_info);
}

void xtb_register(frame, win, func, info)
xtb_frame *frame;
Window win;
FNPTR( func, xtb_hret, (XEvent *evt, xtb_data info, Window parent) );
xtb_registry_info *info;
/*
 * Associates the event handling function `func' with the window
 * `win'.  Additional information `info' will be passed to `func'.
 * The routine should return one of the return codes given above.
 * frame->win is set to win.
 */
{
	frame->win= win;
	frame->info= info;
	xtb_register_( frame, win, func, info);
}

int xtb_update_registry( Window win, xtb_frame *frame, xtb_hret (*func)(XEvent *, xtb_data, Window), xtb_registry_info *info, int mask)
/*
 * Changes or updates the registry for <win>, that must already be registered.
 */
{ xtb_data data;

	if( !XFindContext(t_disp, win, h_context, (caddr_t*) &data) ) {
	  struct h_info *info= (struct h_info*) data;
		if( CheckMask(mask, XTB_UPDATE_REG_FRAME) ){
			info->frame= frame;
		}
		if( CheckMask(mask, XTB_UPDATE_REG_FUNC) ){
			info->func= func;
		}
		if( CheckMask(mask, XTB_UPDATE_REG_INFO) ){
			info->data.info= info;
		}
		  /* I don't think it is actually necessary to save the new info again.
		   \ Yet, it *is* not impossible that there are servers on which it
		   \ may be necessary... (the manpages don't say whether the stored info
		   \ is just the pointer we passed, or a local or remote or whatever copy).
		   */
		XSaveContext(t_disp, win, h_context, (caddr_t) info);
		return(0);
	}
	else{
		return 1;
	}
}

xtb_data xtb_lookup(win)
Window win;
/*
 * Returns the associated data with window `win'.
 */
{
	xtb_data data;

	if( !XFindContext(t_disp, win, h_context, (caddr_t*) &data) ) {
		return ((struct h_info *) data)->data.info;
	} else {
		return (xtb_data) 0;
	}
}

xtb_frame *xtb_lookup_frame(win)
Window win;
/*
 * Returns the associated data with window `win'.
 */
{
	xtb_data data;

	if( !XFindContext(t_disp, win, h_context, (caddr_t*) &data) ) {
		return ((struct h_info *) data)->frame;
	} else {
		return (xtb_frame*) 0;
	}
}

int xtb_disable( Window win )
{ xtb_frame *frame= xtb_lookup_frame(win);
  int r= 0;
	if( frame ){
		r= frame->enabled;
		frame->enabled= 0;
		switch( frame->info->type ){
			case xtb_BT:
				bt_draw( win, (b_info*) frame->info );
				break;
			case xtb_TI:
				ti_draw( win, (ti_info*) frame->info, True );
				break;
			case xtb_TI2:
				frame->framelist[0]->enabled= 0;
				frame->framelist[1]->enabled= 0;
				((ti2_info*)frame->info)->ti_info->clear_flag= True;
				xtb_ti2_redraw( win );
				break;
			case xtb_SR:
				sr_draw( win, (sr_info*) frame->info, True );
				break;
		}
	}
	return( r );
}

int xtb_enable( Window win )
{ xtb_frame *frame= xtb_lookup_frame(win);
  int r= 0;
	if( frame ){
		r= frame->enabled;
		frame->enabled= 1;
		switch( frame->info->type ){
			case xtb_BT:
				bt_draw( win, (b_info*) frame->info );
				break;
			case xtb_TI:
				ti_draw( win, (ti_info*) frame->info, True );
				break;
			case xtb_TI2:
				frame->framelist[0]->enabled= 1;
				frame->framelist[1]->enabled= 1;
				((ti2_info*)frame->info)->ti_info->clear_flag= True;
				xtb_ti2_redraw( win );
				break;
			case xtb_SR:
				sr_draw( win, (sr_info*) frame->info, True );
				break;
		}
	}
	return( r );
}

int xtb_enabled( Window win )
{ xtb_frame *frame= xtb_lookup_frame(win);
  int r= 0;
	if( frame ){
		r= frame->enabled;
	}
	return( r );
}

int xtb_disables( Window w, VA_DCL )
{ va_list ap;
  int r= 0;
	va_start(ap,w);
	do{
		r+= xtb_disable( w );
	}
	while( (w= va_arg( ap, Window))!= (Window) 0 );
	va_end(ap);
	return(r);
}

int xtb_enables( Window w, VA_DCL )
{ va_list ap;
  int r= 0;
	va_start(ap,w);
	do{
		r+= xtb_enable( w );
	}
	while( (w= va_arg( ap, Window))!= (Window) 0 );
	va_end(ap);
	return(r);
}

/* xtb_concat(): concats a number of strings    */
#ifdef CONCATBLABLA
char *xtb_concat( char *a, char *b, ...)
{}
#endif
char *xtb_concat(char *first, VA_DCL)
{ va_list ap;
   int n= 0, tlen= 0;
   char *c= first, *buf= NULL;
	va_start(ap,first);
	do{
		tlen+= strlen(c)+ 1;
		n+= 1;
	}
	while( (c= va_arg(ap, char*)) != NULL );
	va_end(ap);
	if( n== 0 || tlen== 0 ){
		if( debugFlag ){
			fprintf( StdErr, "concat(): %d strings, %d bytes: no action\n",
				n, tlen
			);
		}
		return( NULL );
	}
	if( (buf= calloc( tlen, sizeof(char) )) ){
		c= first;
		va_start(ap,first);
		do{
			strcat( buf, c);
		}
		while( (c= va_arg(ap, char*)) != NULL );
		va_end(ap);
		if( debugFlag ){
			fprintf( StdErr, "concat(): %d bytes to concat %d strings\n",
				tlen, n
			);
		}
	}
	else{
		if( debugFlag ){
			fprintf( StdErr, "concat(): can't get %d bytes to concat %d strings\n",
				tlen, n
			);
		}
	}
	return( buf );
}

int xtb_modifier_state= 0, xtb_button_state, xtb_clipboard_copy= False;

int xtb_Mod2toMod1(int mask)
{
	if( CheckMask( mask, Mod2Mask ) ){
#ifndef linux
		if( !CygwinXorXmingServer ){
			if( debugFlag && debugLevel > 1 ){
				fprintf( StdErr, "xtb_Mod2ToMod1(%s): Mod2Mask -> Mod1Mask\n",
					   xtb_modifiers_string(mask)
				);
			}
			mask|= Mod1Mask;
//			mask&= ~Mod2Mask;
		}
		else
#endif
		{
		  // 20110902: this is on Linux or an Xming or Cygwin/X server. They set Mod2Mask when Num_Lock is true, which isn't
		  // what we'd want.
		  // Downbeat: on those servers it is probably impossible to bind an event to the Mod2 modifier...
			mask&= ~Mod2Mask;
		}
	}
	return( mask );
}

static int win_hit( xtb_frame *list, Window win, xtb_data _info, xtb_frame **hit_frame )
{ int i= -1, hit= 0;
	if( !list ){
		return(0);
	}
	*hit_frame= NULL;
	hit= (list->win== win || list->info== _info );
	if( hit ){
		*hit_frame= list;
	}
	for( i= 0; !hit && list->framelist && i< list->frames; i++ ){
	 xtb_frame *frame= list->framelist[i];
		hit= (frame && (frame->win== win || frame->info== _info ));
		if( hit ){
			*hit_frame= frame;
		}
		else if( !frame && debugFlag ){
			fprintf( StdErr, "win_hit(win=0x%lx,_info=0x%lx) empty frame #%d in list=0x%lx \"%s\"\n",
				win, _info, i, list,
				(list->description)? list->description : ""
			);
			fflush( StdErr );
		}
	}
	if( hit && debugFlag && debugLevel== -2 ){
		fprintf( StdErr, "win_hit(win=0x%lx,_info=0x%lx) hit in frame #%d win=0x%lx,info=0x%lx \"%s\"\n",
			win, _info, i, (*hit_frame)->win, (*hit_frame)->info,
			((*hit_frame)->description)? (*hit_frame)->description : ""
		);
		fflush( StdErr );
	}
	return( hit );
}

xtb_hret xtb_dispatch( Display *disp, Window window, int frames, xtb_frame *framelist, XEvent *evt)
/*
 * Dispatches an event to a handler if its ours.  Returns one of
 * the return codes given above (XTB_NOTDEF, XTB_HANDLED, or XTB_STOP).
 */
{ struct h_info *info;
  Window pwin;
  xtb_frame *frame, *hit_frame= NULL;
  int frame_no= -1;
  Window root_rtn, child_rtn;
  int context_err, root_x, root_y, mask_rtn, newX, newY;
  xtb_hret hret= XTB_NOTDEF;

	if( frames && framelist ){
	 int hit;
	 xtb_data _info= xtb_lookup(evt->xany.window);
	 char msg[512]= "";
		frame_no= 0;
		hit= win_hit( &framelist[frame_no], evt->xany.window, _info, &hit_frame );
		while( frame_no< frames && !hit ){
			if( !(hit= win_hit( &framelist[frame_no], evt->xany.window, _info, &hit_frame )) ){
				frame_no+= 1;
			}
		}
		if( !hit || frame_no>= frames ){
			sprintf( msg, "xtb_dispatch(): window %ld not found in framelist\n", evt->xany.window );
		}
		else if( frame_no< frames ){
			if( framelist[frame_no].parent!= window ){
				sprintf( msg, "xtb_dispatch(): framelist[%d:%d].parent= 0x%lx != 0x%lx\n",
					frame_no, frames, framelist[frame_no].parent, evt->xany.window
				);
				hit= 0;
			}
		}
		if( msg[0] ){
			if( debugFlag && debugLevel== -2 ){
				fputs( msg, StdErr );
				fflush( StdErr );
			}
			  /* 20010722: give it a second change: who knows what the registry may come up with! */
/* 			return( XTB_NOTDEF );	*/
		}
		if( hit ){
			pwin= framelist[frame_no].parent;
			frame= &framelist[frame_no];
		}
		else{
			pwin= window;
			if( !xtb_dispatch_modal &&
				(frame= xtb_lookup_frame( evt->xany.window )) && frame->info && frame->info->type== xtb_User
			){
				if( debugFlag && debugLevel== -2 ){
					fprintf( StdErr, "xtb_dispatch(): window %ld: assuming parent=%ld, frame=0x%lx (window %ld %s)\n",
						evt->xany.window, pwin, frame, frame->win, (frame->description)? frame->description : ""
					);
				}
			}
			else{
				hret= XTB_NOTDEF;
				goto xtbdp_return;
			}
		}
	}
	else{
		pwin= window;
		frame= xtb_lookup_frame( evt->xany.window );
	}
	if( !hit_frame ){
		hit_frame= frame;
	}
	context_err= (int) XFindContext(disp, evt->xany.window, h_context, (caddr_t *) &info);
	if( !context_err && info ){
		if( info->frame && info->frame->info != info->data.info ){
			if( info->data.basic->frame!= info->frame ){
				fprintf( StdErr,
					"xtb_dispatch(%s): warning: potentially fatal error: info->data.info=0x%lx != info->frame->info=0x%lx",
					(info->frame->description)? info->frame->description : ((hit_frame->description)? hit_frame->description : ""),
					info->data.info, info->frame->info
				);
				fflush( StdErr );
				if( info->data.basic->type<= 0 || info->data.basic->type>= FrameTypes ){
					fprintf( StdErr, " and invalid frame type %d: ignoring event!\n",
						info->data.basic->type
					);
					frame= NULL;
					info= NULL;
				}
				else if( ((xtb_registry_info*)info->frame->info) && ((xtb_registry_info*)info->frame->info)->frame== info->frame ){
					fprintf( StdErr, ": corrected" );
					info->data.info= info->frame->info;
				}
				fputs( "\n", StdErr );
				fflush( StdErr );
			}
/*
			else{
				fprintf( StdErr, "xtb_dispatch(): info->frame->info=0x%lx!= info->data.info=0x%lx: ignoring event\n",
					info->frame->info, info->data.info
				);
				frame= NULL;
				context_err= 1;
			}
 */
		}
	}
	if( debugFlag && debugLevel== -2 ){
		fprintf( StdErr, "xtb_dispatch(%s): evt.xany.window=0x%lx window=0x%lx\n",
			event_name(evt->type), evt->xany.window, window
		);
	}
	if( frame && evt->xany.type== ButtonPress ){
	 static char active= 0;
		XQueryPointer(disp, evt->xany.window, &root_rtn, &child_rtn, &root_x, &root_y,
							 &newX, &newY, &mask_rtn);
		mask_rtn= xtb_Mod2toMod1(mask_rtn);
		if( CheckMask(mask_rtn, ControlMask) && !(CheckMask(mask_rtn, ShiftMask) || CheckMask(mask_rtn, Mod1Mask)) ){
			  /* Check if the event is for the current window, if that is not of type xtb_User, and display the description
			   \ if we're not already displaying one...
			   */
			if( !active && (hit_frame->win == evt->xany.window) && (!hit_frame->info || hit_frame->info->type!= xtb_User) ){
			 char *title= NULL, *description= (hit_frame->description)? hit_frame->description : "<No description>", *desc= NULL;
				if( info ){
					if( info->func== ti_h ){
					 char buf[256];
					 struct ti_info *ti= (struct ti_info*) info->data.info;
						sprintf( buf, "\n%d(%d) of %d, w=%d, curpos=%d\n",
							strlen(ti->text), ti->curlen, ti->maxlen, ti->maxwidth, ti->curidx
						);
						desc= xtb_concat( description, "\n", ti->text, buf, NULL );
					}
					else if( info->func== bt_h ){
						desc= xtb_concat( description, "\n", ((struct b_info*) info->data.info)->flag? "<ON>" : "<OFF>", "\n", NULL );
						title= ((struct b_info*)info->data.info)->text;
					}
					else if( info->func== sr_h ){
					 char buf[256];
					 struct sr_info *ri= (struct sr_info*) info->data.info;
						sprintf( buf, " [%s,%s] %s%%\n",
							d2str( ri->minvalue, "%g", NULL),
							d2str( ri->maxvalue, "%g", NULL),
							d2str( (ri->minvalue!= ri->maxvalue)? 100 * (ri->value - ri->minvalue)/ (ri->maxvalue - ri->minvalue) :
								100, "%g", NULL
							)
						);
						desc= xtb_concat( description,
							"\nDrag/click with mouse, click in arrows\n",
							(ri->vert_flag)? "or use use cursor up/down\n" : "or use cursor left/right\n",
							"2nd button in arrows sweeps; holding the\n",
							"shift key delays calling the callback handler\n\n",
							ri->text, buf, NULL
						);
					}
					else if( info->func== to_h ){
						title= ((struct to_info*)info->data.info)->text;
					}
					if( !info->frame->enabled ){
					  char *c= desc;
						desc= xtb_concat( c, "Widget is disabled\n\n", NULL );
						xfree( c );
					}
				}
				active= 1;
				if( desc ){
					if( info && info->func== ti_h ){
					  struct ti_info *ti= (struct ti_info*) info->data.info;
					  int id;
					  char *sel= NULL;
					  xtb_frame *menu= NULL;
/* 						xtb_box_ign_first_brelease= 1;	*/
						id= xtb_popup_menu( hit_frame->win, desc, "Description & Current Textbuffer (click in text to move cursor)",
							&sel, &menu
						);
						if( sel && *sel && ti->text ){
						  char *c= strstr( ti->text, sel);
						  int n= 0;
							while( c ){
								n+= 1;
								c= strstr( &c[1], sel );
							}
							if( n== 1 ){
								c= strstr( ti->text, sel );
							}
							else if( n> 1 ){
							  int i= 0, f= 0, diff;
							  err_info *minfo= (err_info*) menu->info;
								  /* Find exactly which occurence was selected: this can be done
								   \ using the id of the menu-entry returned.
								   */
								while( i<= id && i< minfo->num_lines ){
									if( (diff= strcmp( xtb_popup_get_itemtext( menu, i), sel))== 0 ){
										f+= 1;
									}
									i+= 1;
								}
#ifdef FEEDBACK_ON_SELECTION
								{ char mbuf[128], nbuf[128];
									sprintf( mbuf, "Which occurence of the above pattern do you want (%d total)\n", n );
									sprintf( nbuf, "%d", f );
									if( xtb_input_dialog( ti->frame->win, nbuf, 16, 127,
											mbuf, sel, NULL, NULL, NULL, NULL
										)
									){
										i= atoi( nbuf );
										CLIP( i, 0, n );
										if( i> 0 ){
											n= i;
											c= strstr( ti->text, sel );
											for( i= 1; i< n && c; i++ ){
												c= strstr( &c[1], sel );
											}
										}
										else{
											Boing(10);
										}
									}
									else{
										c= NULL;
									}
								}
#else
								if( f> 0 ){
									n= f;
									c= strstr( ti->text, sel );
									for( i= 1; i< n && c; i++ ){
										c= strstr( &c[1], sel );
									}
								}
								else{
									Boing(10);
								}
#endif
							}
							if( c ){
								ti->curidx= (int) (c- ti->text);
								ti->curxval= text_width(norm_font, ti->text, ti->curidx, ti);
								ti->startxval= 0;
								ti->startidx= 0;
								xtb_ti_scroll_left( ti->frame->win, ti->curidx );
								ti_draw( ti->frame->win, ti, True );
							}
						}
						xtb_popup_delete( &menu );
					}
					else{
						xtb_error_box( hit_frame->win, desc, (title)? title : "Description & State:");
					}
					xfree(desc);
				}
				else{
					xtb_error_box( hit_frame->win, description, (title)? title : "Description:");
				}
				active= 0;
				return( XTB_HANDLED);
			}
			hret= XTB_NOTDEF;
			goto xtbdp_return;
		}
	}
	if( evt->xany.type== KeyPress || evt->xany.type== KeyRelease || evt->xany.type== ButtonPress ||
		evt->xany.type== ButtonRelease || evt->xany.type== MotionNotify
	){
		xtb_modifier_state= xtb_Mod2toMod1( 0xFF & evt->xbutton.state );
		xtb_button_state= 0xFF00 & evt->xbutton.state;
	}
	else{
		xtb_modifier_state= 0;
		xtb_button_state= 0;
	}
	if( !context_err && info ){
		if( info->func){
			if( debugFlag && debugLevel== -2 ){
				fprintf( StdErr, "xtb_dispatch(\"%s\",w=%ld,fr=0x%lx[%d],0x%lx(%s #%lu))\n\tpwin=0x%lx win=0x%lx func=0x%lx info=0x%lx\n",
					DisplayString(disp), window, framelist, frames, evt, event_name(evt->type), evt->xany.serial,
					pwin, evt->xany.window,
					info->func, info->data.info
				);
				fprintf( StdErr, "\tframe=0x%lx %s \"%s\"\n", frame,
					(frame && frame->mapped)? "[mapped]" : "[NOT-MAPPED]",
					(frame && frame->description)? frame->description : ""
				);
				fflush( StdErr );
			}
			if( evt->type== Expose && evt->xexpose.count> 0 ){
				if( debugFlag && debugLevel== -2 ){
					fprintf( StdErr, "\t\tEXPOSE event with count==%d skipped\n", evt->xexpose.count );
				}
				return( XTB_HANDLED );
			}
			hret= (*info->func)(evt, info->data.info, pwin );
		}
		else{
			hret= XTB_NOTDEF;
		}
	} else hret= XTB_NOTDEF;
xtbdp_return:;
#ifdef XGRAPH
{ extern int CursorCross;
	if( hret!= XTB_HANDLED && evt->xany.type== MotionNotify && CursorCross ){
		  /* 20041217: see the corresponding comment in _Handle_An_Event()! */
		DrawCursorCrosses( evt, NULL );
	}
}
#endif
	return( hret );
}

/* 20020429: the escaping principle for the greek toggle opcode is complex, and cannot really
 \ be done by only looking ahead in the buffer. Therefore, add support for escaping with the
 \ real ASCII escape character (which probably isn't used in any font), and make an attempt anyway:
 \ NB: there is support for looking backwards. Currently, this is used only to check whether or not
 \ the preceding character is an ESC. Checking backwards for sequences of \ characters is too complex -- there!
 */
int xtb_toggle_greek( char *text, char *first )
{ int toggle= 0;
	if( text && *text ){
		if( text[0]== '\\' && (text== first || text[-1]!= 0x7f) ){
			if( text[1] && text[1]== '\\' ){
				if( text[2] ){
					  /* If the char following the next (text[2]) is a \ too, suppose that
					   \ text[1] is an escape for text[2], and toggle here (text[0]). Else,
					   \ toggle rather (maybe) at the next position.
					   */
					toggle= (text[2]== '\\')? True : False;
				}
				else{
				  /* last char is a \ too; toggle at that position */
					toggle= 0;
				}
			}
			else{
				toggle= True;
			}
		}
	}
	return(toggle);
}

/* Check if a string has the escape code for selecting greek, \.
 * A \ followed by another one is not the same!
 \ This routine returns a pointer to the last greek toggle in the string.
 */
char *xtb_has_greek_0( char *text )
{  char *c= index( text, '\\');
	while( c && *c && c[1]== '\\' ){
	  /* if a \ is followed by another, keep looking for a single one!	*/
		c= index( &c[2], '\\');
	}
	if( c && *c== '\\' ){
		return( c );
	}
	else{
		return( NULL );
	}
}

char *xtb_has_greek( char *text )
{  char *c= index( text, '\\');
	while( c && *c && !xtb_toggle_greek(c, text) ){
	  /* if a \ is followed by another, keep looking for a single one!	*/
/* 		c= (c[1]== '\\' )? index( &c[2], '\\') : index( &c[1], '\\');	*/
		c= index( &c[1], '\\');
	}
	if( c && *c== '\\' && xtb_toggle_greek(c, text) ){
		return( c );
	}
	else{
		return( NULL );
	}
}

/* Check if a string has the escape code for a backslash, \\.
 * This is the inverse of xtb_has_greek(), in a sense.
 \ 20020429: maybe we should use xtb_toggle_greek() here!
 */
char *xtb_has_backslash_0( char *text )
{  char *c= index( text, '\\');
	while( c && *c && c[1]!= '\\' ){
		c= index( &c[2], '\\');
	}
	if( c && *c== '\\' && c[1]== '\\' ){
		return( c );
	}
	else{
		return( NULL );
	}
}

char *xtb_has_backslash( char *text )
{  char *c= index( text, '\\');
	while( c && *c && xtb_toggle_greek(c, text) ){
		c= (c[1]== '\\')? &c[1] : index( &c[2], '\\');
	}
	if( c && *c== '\\' && !xtb_toggle_greek(c, text) ){
		return( c );
	}
	else{
		return( NULL );
	}
}

int xtb_TextExtents( XFontStruct *font1, XFontStruct *alt_font, char *text, int Len, int *dir, int *ascent, int *descent,
	XCharStruct *bb, Boolean escape_backslash)
{
	XCharStruct BB;
	int len, Ascent= 0, Descent= 0, width= 0;
	int loop= 0;
	char *Text= text, *greek_rest;
	XFontStruct *lfont;

	if( !text ){
		return(0);
	}
	if( text[0]== 0x01 ){
		  /* RJVB 20020918: skip an initial ^A that selects reverse video mode: */
		text++;
		Text++;
		Len-= 1;
	}
	lfont= font1;
	memset( bb, 0, sizeof(XCharStruct) );
	*ascent= 0;
	*descent= 0;
	while( Text && *Text ){
	  char *_Text;
		if( (greek_rest= xtb_has_greek( &Text[1] )) && alt_font ){
			*greek_rest= '\0';
		}
		if( Text[0]== '\\' && Text[1]!= '\\' && alt_font ){
			Text++;
			if( lfont== font1){
				lfont= alt_font;
			}
			else{
				lfont= font1;
			}
		}
		_Text= xtb_textfilter( Text, lfont, escape_backslash );
		len = strlen(_Text);
		XTextExtents( lfont, _Text, len, dir, &Ascent, &Descent, &BB);
		*ascent= (!loop)? BB.ascent : MAX( *ascent, BB.ascent );
		*descent= (!loop)? BB.descent : MAX( *descent, BB.descent );
		width+= BB.rbearing - BB.lbearing;
		bb->width+= BB.width;
		bb->lbearing= (!loop)? BB.lbearing - bb->width : MIN( bb->lbearing, BB.lbearing - bb->width);
		bb->rbearing= (!loop)? BB.rbearing - bb->width : MAX( bb->rbearing, BB.rbearing - bb->width);
		bb->ascent= (!loop)? BB.ascent : MAX( bb->ascent, BB.ascent);
		bb->descent= (!loop)? BB.descent : MAX( bb->descent, BB.descent);
		if( greek_rest && alt_font ){
			Text= greek_rest;
			*Text= '\\';
		}
		else{
			Text= NULL;
		}
		loop+=1;
	}
	bb->lbearing+= bb->width;
	bb->rbearing+= bb->width;
	if( debugFlag && debugLevel== -3 ){
		fprintf( StdErr, "xtb_TextExtents(\"%s\"): width=%d ascent=%d descent=%d\n\
		bb->width=%d bb->lbearing=%d bb->rbearing=%d (%d)\n\
		bb->ascent=%d bb->descent=%d\n",
			text, width, *ascent, *descent,
			bb->width, bb->lbearing, bb->rbearing,
				bb->rbearing - bb->lbearing,
			bb->ascent, bb->descent
		);
		fflush( StdErr );
	}
/* 	xtb_textfilter_free();	*/
	return( bb->rbearing - bb->lbearing );
}

int xtb_TextWidth( char *text, XFontStruct *font1, XFontStruct *alt_font )
/* Determine the width of an X11 string. The character '\' is used to
 \ switch between font1 and alt_font, iff alt_font!=NULL
 */
{
	XCharStruct bb;
	int dir;
	int ascent, descent;

	if( !text ){
		return(0);
	}
	else{
		return( xtb_TextExtents( font1, alt_font, text, strlen(text), &dir, &ascent, &descent, &bb, False ) );
	}
}

int XFontWidth(XFontStruct *font)
{
	if( font ){
	  int w= (int) ((
				+ (double) XTextWidth( font, "8", 1)
				+ (double) XTextWidth( font, "m", 1)
				  /* 991105:	*/
				+ (double) XTextWidth( font, "M", 1)
				+ (double) XTextWidth( font, " ", 1)
/* 				+ (double) XTextWidth( font, "O", 1)   	*/
/* 				+ (double) XTextWidth( font, "@", 1)   	*/
			) / 4.0 + 0.5 );
		if( w> 0 ){
			return( w );
		}
		else{
			return( MAX( 0, font->max_bounds.rbearing - font->max_bounds.lbearing ) );
		}
	}
	else{
		if( debugFlag ){
			fprintf( StdErr, "XFontWidth() called with a NULL font pointer!\n" );
		}
		return(0);
	}
}

static XFontStruct *t_font;
static GC textGC(Window t_win, GC gc, XFontStruct *T_font)
/*
 * Sets the fields above in a global graphics context.  If
 * the graphics context does not exist,  it is created.
 */
{
	XGCValues gcvals;
	unsigned long gcmask;

	t_font= T_font;
	gcvals.font = T_font->fid;
	gcmask = GCFont;
	XChangeGC(t_disp, gc, gcmask, &gcvals);
	return gc;
}

/* Make a local, static copy of a printing string, which does not
 \ contain characters which are not in either the specified font, or in
 \ the current font (some servers seem to crash if these are passed..!)
 \ If no copy can be obtained, the original buffer is passed without modifications.
 \ If xtb_escape_backslash=True, escaped backslashes are preserved.
 \ 980617: must free after use..! (outcommented caching lines)
 */
char *xtb_textfilter( char *_text, XFontStruct *font, Boolean escape_backslash )
{ int bsl= 0, range= 0, slen= strlen(_text);
	if( slen> xtb_textfilter_buflen || !textalloc ){
		if( xtb_textfilter_buf && textalloc ){
			if( textalloc== 2 ){
				fprintf( StdErr, "xtb_textfilter(): realloc'ing initial buffer[%d] for buffer of size %d\n",
					xtb_textfilter_buflen, slen
				);
			}
			xfree(xtb_textfilter_buf);
			textalloc= False;
		}
		if( !(xtb_textfilter_buf= calloc( (xtb_textfilter_buflen=slen+ 1), sizeof(char))) ){
			if( debugFlag ){
				fprintf( StdErr,
					"xtb_textfilter(\"%s\"): can't realloc buffer (%s) -"
					"some X servers might crash on characters not in font used\n",
					_text, serror()
				);
			}
			xtb_textfilter_buflen= 0;
			textalloc= False;
			return( _text );
		}
		else{
			textalloc= True;
		}
	}
	if( textalloc ){
	  char *c= xtb_textfilter_buf, *d= _text, dc;
	  unsigned int min, max;
		if( font ){
			min= (unsigned int) font->min_char_or_byte2, max= (unsigned int) font->max_char_or_byte2;
			dc= font->default_char;
		}
		else if( t_font ){
			min= (unsigned int) t_font->min_char_or_byte2, max= (unsigned int) t_font->max_char_or_byte2;
			dc= t_font->default_char;
		}
		else{
			min= max= 0;
		}
		if( (min || max) || xtb_has_backslash(xtb_textfilter_buf) ){
			if( !dc ){
				dc= ' ';
			}
			while( *d ){
			  unsigned int D= ((unsigned int) *d) & 0x000000ff;
				if( D>= min && D<= max ){
					if( !escape_backslash && (*d== '\\' && d[1]== '\\') ){
						*c++ = *d++;
						bsl+= 1;
					}
					else if( *d== 0x7f ){
					  /* 20020429: skip escape characters and output the next one verbatim: */
						*c++ = *++d;
					}
					  /* 20020313 */
					else if( *d== '\t' || *d== '\n' ){
						*c++ = ' ';
					}
					else{
						*c++ = *d;
					}
				}
				else{
					*c++ = dc;
					range+= 1;
				}
				d++;
			}
			*c= '\0';
		}
		else{
			strcpy( c, d );
		}
		if( (range || bsl) && debugFlag && debugLevel== -3 ){
			fprintf( StdErr,
				"xtb_textfilter(\"%s\")=\"%s\" ; "
				"subst. %d chars outside ASCII-range [%u,%u] with '%c' and/or %d backslash patterns\n",
				_text, xtb_textfilter_buf,
				range, min, max, dc, bsl
			);
		}
	}
	return(xtb_textfilter_buf);
}

void xtb_textfilter_free()
{
	if( textalloc && xtb_textfilter_buf && xtb_textfilter_buflen ){
		xfree( xtb_textfilter_buf );
		xtb_textfilter_buflen= 0;
	}
}

int xtb_DrawString( Display *t_disp, Window win, GC gc, int x, int y,
	char *text, int Len, XFontStruct *font1, XFontStruct *alt_font,
	int text_v_info, Boolean escape_backslash
)
/* Draw a string. The character '\' is used to
 \ switch between font1 and alt_font, iff alt_font!=NULL
 \ (x,y) specify the upper-left corner of the image-string!
 \ if text_v_info== True, vertical dimension information from the string
 \ is used for the vertical placement, otherwise the max_ascent of font1 and
 \ alt_font is used.
 */
{
	XCharStruct bb;
	int rx, len, dir, rich= 0;
	int _max_ascent, ascent, descent;
/*  extern char *index();   */
	char *Text, *greek_rest;
	XFontStruct *lfont;
	int ma= max_ascent, md= max_descent;

	if( !text ){
		return(0);
	}
	if( text[0]== 0x01 ){
		  /* RJVB 20020918: skip an initial ^A that selects reverse video mode: */
		text++;
		Len-= 1;
	}
	if( !font1 && !alt_font ){
	  char *tt;
	  /* In this case, we just call XDrawImageString, but we do filter out characters
	   \ that are not defined in the current font.
	   */
		XDrawImageString( t_disp, win, gc, x, y, (tt= xtb_textfilter( text, NULL, escape_backslash)), Len );
/* 		xtb_textfilter_free();	*/
		return(0);
	}

	xtb_TextExtents( font1, alt_font, text, strlen(text), &dir, &ascent, &descent, &bb, escape_backslash );
	if( alt_font ){
		max_ascent= MAX( font1->ascent, alt_font->ascent );
		max_descent= MAX( font1->descent, alt_font->descent );
	}
	else{
		max_ascent= font1->ascent;
		max_descent= font1->descent;
	}
	if( text_v_info ){
		y+= (int) (((double)(max_ascent + max_descent) - (bb.ascent + bb.descent))/ 2.0 + 0.5);
		_max_ascent= bb.ascent;
	}
	else{
		_max_ascent= max_ascent;
	}

	rx = x;
	lfont= font1;
	Text= text;
	while( Text && *Text ){
	  char *_Text;
		if( (greek_rest= xtb_has_greek( &Text[1] )) && alt_font ){
			*greek_rest= '\0';
		}
/* 		if( Text[0]== '\\' && Text[1]!= '\\' && alt_font )	*/
		if( alt_font && xtb_toggle_greek(Text,text) )
		{
			Text++;
			if( lfont== font1){
				lfont= alt_font;
				rich+= 1;
			}
			else{
				lfont= font1;
			}
		}
		_Text= xtb_textfilter( Text, lfont, escape_backslash );
		len = strlen(_Text);
		XTextExtents( lfont, _Text, len, &dir, &ascent, &descent, &bb);
		XDrawImageString(t_disp, win, textGC( win, gc, lfont),
				rx, y+ _max_ascent, _Text, len
		);
		rx+= bb.rbearing - bb.lbearing;
		if( greek_rest && alt_font ){
			Text= greek_rest;
			*Text= '\\';
		}
		else{
			Text= NULL;
		}
	}
/* 	xtb_textfilter_free();	*/
	max_ascent= ma;
	max_descent= md;
	return( rich );
}

#define BT_HPAD (3+1)
#define BT_VPAD (2+1)
#define BT_LPAD 3
#define BT_BRDR 1

/* #define BT_WIDTH(bb) (bb.width+ 2*BT_HPAD)   */
	 /* This works better with cursorfont (beats me!): */
#define BT_WIDTH(bb)	(bb.rbearing-bb.lbearing+ 2*BT_HPAD)

static void bt_line(win, ri, pix)
Window win;
struct b_info *ri;
unsigned long pix;
/*
 * Draws a status line beneath the text to indicate the
 * user has moved into the button.
 */
{
	XDrawLine(t_disp, win,
		 xtb_set_gc(win, pix, (ri->flag)? ri->light_pix : ri->middle_pix, (ri->font)? ri->font->fid : norm_font->fid ),
		 BT_HPAD, ri->line_y, BT_HPAD+ri->line_w, ri->line_y
	);
}

static void bt_draw( Window win, struct b_info *ri)
/*
 * Draws a button window
 */
{
	XCharStruct bb;
	int dir, ascent, descent, text_v_info= False, centre= True;
	XPoint line[3];
	GC lineGC;
/*  char line_style[16]= { 1, 1 };  */

	if( !ri->frame || !ri->frame->mapped ){
		return;
	}

#if DEBUG == 2
	{ char *c= index( ri->text, 0x7f );
		if( c && c[1]== '\\' ){
			;
		}
	}
#endif
	xtb_TextExtents( ri->font, ri->alt_font, ri->text, strlen(ri->text), &dir, &ascent, &descent, &bb, False);

	switch( ri->pos ){
		default:
		case XTB_TOP_LEFT:
			ri->x_text_offset= 0;
			ri->y_text_offset= 0;
			break;
		case XTB_TOP_RIGHT:
			ri->x_text_offset= -bb.lbearing+ BT_HPAD+ BT_BRDR;
			ri->y_text_offset= 0;
			break;
		case XTB_CENTERED:
			 /* This works best with cursorfont (what about others?! - never tried!)   */
			ri->x_text_offset= -bb.lbearing+ (ri->frame->width - BT_WIDTH(bb))/2.0 ;
			ri->y_text_offset= 0;
			text_v_info= True;
			centre= False;
			break;
	}
	if( centre ){
		  /* If allowed, position text in centered mode, *with respect to the offset already determined* */
		ri->x_text_offset+= (ri->frame->width - (bb.rbearing-bb.lbearing))/2- BT_HPAD- BT_BRDR;
	}

	if( ri->flag) {
		if( ri->frame->enabled ){
			XSetWindowBackgroundPixmap( t_disp, win, None );
			XSetWindowBackground( t_disp, win, ri->lighter_pix);
		}
		else{
			XSetWindowBackground( t_disp, win, ri->lighter_pix);
			XSetWindowBackgroundPixmap( t_disp, win, disabled_bg_light );
		}
		XClearWindow( t_disp, win);

/* 		{ int ys= ri->frame->height/2;	*/
/* 			line[0].x= 0, line[0].y= ys/2;	*/
/* 			line[1].x= ys*2, line[1].y= 0;	*/
/* 			line[2].x= 0, line[2].y= 0;	*/
/* 			XFillPolygon( t_disp, win, xtb_set_gc(win, xtb_red_pix, ri->light_pix, ri->font->fid),	*/
/* 				line, 3, Convex, CoordModeOrigin	*/
/* 			);	*/
/* 		}	*/

		xtb_DrawString(t_disp, win,
				(ri->focus_flag)? xtb_set_gc(win, ri->lighter_pix, ri->norm_pix, ri->font->fid) :
					xtb_set_gc(win, ri->norm_pix, ri->lighter_pix, ri->font->fid),
				ri->x_text_offset+ BT_HPAD, ri->y_text_offset+ BT_VPAD,
				ri->text, strlen(ri->text), ri->font, ri->alt_font, text_v_info, False
		);

		line[0].x= 0;
		line[0].y= (short) ri->line_y+ 1;
		line[1].x= (short) ri->line_w+ 2* BT_HPAD- 1;
		line[1].y= (short) ri->line_y+ 1;
		line[2].x= (short) ri->line_w+ 2* BT_HPAD- 1;
		line[2].y= 0;
		lineGC= xtb_set_gc(win, ri->back_pix, ri->lighter_pix, ri->font->fid);
		XSetLineAttributes( t_disp, lineGC, 2, LineSolid, CapButt, JoinMiter);
		XDrawLines(t_disp, win, lineGC, line, 3, CoordModeOrigin);

		line[0].x= line[1].x= line[1].y= line[2].y= 1;
		line[0].y= (short) ri->line_y+ 1;
		line[2].x= (short) ri->line_w+ 2* BT_HPAD- 1;
		lineGC= xtb_set_gc(win, (ri->flag2)? xtb_red_pix : ri->norm_pix, ri->lighter_pix, ri->font->fid);
		XSetLineAttributes( t_disp, lineGC, 2, LineSolid, CapButt, JoinMiter);
		XDrawLines(t_disp, win, lineGC, line, 3, CoordModeOrigin);

		if( ri->enter_flag ){
			bt_line(win, ri, ri->norm_pix );
		}
	} else {
		if( ri->frame->enabled ){
			XSetWindowBackgroundPixmap( t_disp, win, None );
			XSetWindowBackground( t_disp, win, ri->middle_pix);
		}
		else{
			XSetWindowBackground( t_disp, win, ri->middle_pix);
			XSetWindowBackgroundPixmap( t_disp, win, disabled_bg_middle );
		}
		XClearWindow( t_disp, win);
		xtb_DrawString(t_disp, win,
				(ri->focus_flag)? xtb_set_gc(win, ri->middle_pix, ri->back_pix, ri->font->fid) :
					xtb_set_gc(win, ri->back_pix, ri->middle_pix, ri->font->fid),
				  /* shift text 1 pixel left to "avoid" the dark right border shadow */
				ri->x_text_offset+ BT_HPAD- 1, ri->y_text_offset+ BT_VPAD,
				ri->text, strlen(ri->text), ri->font, ri->alt_font, text_v_info, False
		);

		line[0].x= 0;
		line[0].y= (short) ri->line_y+ 1;
		line[1].x= (short) ri->line_w+ 2* BT_HPAD- 1;
		line[1].y= (short) ri->line_y+ 1;
		line[2].x= (short) ri->line_w+ 2* BT_HPAD- 1;
		line[2].y= 0;
		lineGC= xtb_set_gc(win, (ri->flag2)? xtb_red_pix : ri->norm_pix, ri->back_pix, ri->font->fid);
		XSetLineAttributes( t_disp, lineGC, 2, LineSolid, CapButt, JoinMiter);
		XDrawLines(t_disp, win, lineGC, line, 3, CoordModeOrigin);

		line[0].x= line[1].x= line[1].y= line[2].y= 1;
		line[0].y= (short) ri->line_y+ 1;
		line[2].x= (short) ri->line_w+ 2* BT_HPAD- 1;
		lineGC= xtb_set_gc(win, ri->light_pix, ri->back_pix, ri->font->fid);
		XSetLineAttributes( t_disp, lineGC, 2, LineSolid, CapButt, JoinMiter);
		XDrawLines(t_disp, win, lineGC, line, 3, CoordModeOrigin);

		if( ri->enter_flag ){
			bt_line(win, ri, ri->back_pix );
		}
	}

}

XEvent *xtb_bt_event= NULL;

static xtb_hret bt_h(evt, info, parent)
XEvent *evt;
xtb_data info;
Window parent;
/*
 * Handles button events.
 */
{
	Window win = evt->xany.window;
	static Window PressWin;
	struct b_info *ri = (struct b_info *) info;
	  /* 990519: default return value	*/
	xtb_hret rtn= XTB_HANDLED;

	xtb_bt_event= evt;
	switch (evt->type) {
		case UnmapNotify:
			if( ri->frame ) ri->frame->mapped= 0;
			break;
		case MapNotify:
			if( ri->frame ) ri->frame->mapped= 1;
			break;
		case VisibilityNotify:
		case ConfigureNotify:
			if( ri->frame ) ri->frame->mapped= 1;
		case Expose:
			bt_draw(win, ri);
			break;
		case FocusIn:
			if (evt->xfocus.detail != NotifyPointer) {
				if( !ri->focus_flag ){
					ri->focus_flag = 1;
					bt_draw(win, ri);
				}
			}
			break;
		case FocusOut:
			if (evt->xfocus.detail != NotifyPointer) {
				if( ri->focus_flag ){
					ri->focus_flag = 0;
					bt_draw(win, ri);
				}
			}
			break;
		case EnterNotify:
			bt_line(win, ri, (ri->flag)? ri->norm_pix : ri->back_pix );
			ri->enter_flag= 1;
			break;
		case LeaveNotify:
			bt_line(win, ri, (ri->flag)? ri->back_pix : ri->norm_pix );
			ri->enter_flag= 0;
			break;
		case KeyPress:
		case ButtonPress:
			 /* Nothing - just wait for button up */
			PressWin= win;
			break;
		case KeyRelease:{
		  int nbytes;
		  char keys[2];
		  KeySym keysyms[2];
			nbytes = XLookupString(&evt->xkey, keys, 2,
						   (KeySym *) keysyms, (XComposeStatus *) 0);
			keysyms[0]= XLookupKeysym( &evt->xkey, 0);
			switch( keysyms[0] ){
				case XK_Return:
				case XK_KP_Enter:
					if( (ri->enter_flag || ri->focus_flag) && win== PressWin ){
						goto HandleBTRelease;
					}
					break;
				default:
					break;
			}
			break;
		}
		case ButtonRelease:
			if( win!= PressWin ){
				return( XTB_HANDLED );
			}
			else{
			 Window root_rtn, child_rtn;
			 int root_x, root_y, curX, curY, mask_rtn;
			 extern FILE *StdErr;
			 xtb_frame *cframe, *hit_frame= NULL;
				XQueryPointer(t_disp, parent, &root_rtn, &child_rtn, &root_x, &root_y,
					 &curX, &curY, &mask_rtn
				);
				mask_rtn= xtb_Mod2toMod1(mask_rtn);
				cframe= xtb_lookup_frame( child_rtn );
				win_hit( cframe? cframe : ri->frame, child_rtn, info, &hit_frame );
				if( debugFlag && debugLevel== -2 ){
					fprintf( StdErr, "bt_h(): parent=0x%lx PressedWin=0x%lx win=0x%lx InWin=0x%lx info->frame->win=0x%lx (%d,%d) (%d,%d)\n",
						parent, PressWin, win, child_rtn, ri->frame->win,
						root_x, root_y, curX, curY
					);
					fflush( StdErr );
				}
				if( !ri->enter_flag ){
					 /* failed. We don't accept this event as a state-switcher.    */
					return( XTB_HANDLED );
				}
				else if( cframe!= ri->frame ){
					 /* button-row. Somehow, we can't distinguish the different buttons
                       \ in it. child_rtn is either the window containing the buttons, or 0
                       \ So we see if the window has focus.
                       */
					if( ri->enter_flag== 0 || child_rtn== 0 ){
						return( XTB_HANDLED );
					}
				}
				else if( child_rtn!= PressWin && parent!= win ){
					 /* cframe==ri->frame: single button. Check if the pointer was within it
                       \ when the mousebutton was released.
                       */
					return( XTB_HANDLED );
				}
			}
HandleBTRelease:;
			if( ri->frame->enabled && ri->func ){
				rtn = (*ri->func)(win, ri->flag, ri->val);
			}
			else{
				if( !ri->frame->enabled ){
					Boing(5);
				}
				rtn= XTB_HANDLED;
			}
			break;
		default:
			rtn = XTB_NOTDEF;
	}
	xtb_bt_event= NULL;
	return rtn;
}

Window xtb_XCreateSimpleWindow(Display *disp, Window parent,
				int x_loc, int y_loc,
				unsigned int width, unsigned int height,
				unsigned int border, unsigned long background, unsigned long foreground
)
{ XSetWindowAttributes wattr;
  unsigned long wamask;
	  /* 20021101: For some reason, XCreateSimpleWindow was called with both a background and a foreground
	   \ colour pixel. The foreground colour is redundant according to the manpage and the prototype, but
	   \ it does appear to have an effect (e.g. an empty xtb_ti field will be white with the default colourscheme
	   \ instead of black). We mimick this here by setting CWBackPixel to foreground!
	   */
	wamask = ux11_fill_wattr(&wattr, CWBackPixel, foreground,
				CWBorderPixel, background,
				  /* Redundant? */
				CWOverrideRedirect, True,
				  /* let's try a (simple?) form of backingstore: nicer for our buttons when we get uncovered! */
				CWBackingStore, WhenMapped,
				CWSaveUnder, True,
				CWColormap, cmap, UX11_END
	);
	return( XCreateWindow(t_disp, parent, x_loc, y_loc, width, height, border,
			   depth, InputOutput, vis, wamask, &wattr)
	);
}

void xtb_BackingStore( Display *disp, Window win, int setting )
{ XSetWindowAttributes wattr;
  unsigned long wamask;
	if( setting ){
		wattr.backing_store= WhenMapped;
		wattr.save_under= True;
	}
	else{
		wattr.backing_store= NotUseful;
		wattr.save_under= True;
	}
	wamask|= CWBackingStore|CWSaveUnder;
 	XChangeWindowAttributes( disp, win, wamask, &wattr );
}

static void _xtb_bt_new( Window win, char *text, int pos,
	FNPTR( func, xtb_hret, (Window, int, xtb_data) ),
	xtb_data val, xtb_frame *frame, char *caller)
{ XCharStruct bb;
  struct b_info *info;
  int dir, ascent, descent;
  XEvent evt;
  Cursor curs= bt_cursor;

	if( !(frame->framelist= (xtb_frame**) malloc(sizeof(xtb_frame*))) ){
		return;
	}
	if( SetParentBackground ){
		XSetWindowBackground( t_disp, win, xtb_light_pix );
	}
	xtb_TextExtents(norm_font, greek_font, text, strlen(text), &dir, &ascent, &descent, &bb, False);
	frame->description= NULL;
	frame->parent= win;
	frame->width = BT_WIDTH(bb);
/*     frame->height = norm_font->ascent + norm_font->descent + BT_VPAD + BT_LPAD;  */
	frame->height = max_ascent + max_descent + BT_VPAD + BT_LPAD;
	frame->x_loc = frame->y_loc = 0;
	frame->win = xtb_XCreateSimpleWindow(t_disp, win,
					frame->x_loc, frame->y_loc,
					frame->width, frame->height,
					BT_BRDR, xtb_norm_pix, xtb_light_pix
	);
	XSetWindowColormap( t_disp, frame->win, cmap );
	if( !curs ){
		curs = XCreateFontCursor(t_disp, XC_hand1 );
		bt_cursor= curs;
	}
	fg_color.pixel = xtb_norm_pix;
	XQueryColor(t_disp, cmap, &fg_color);
	bg_color.pixel = xtb_back_pix;
	XQueryColor(t_disp, cmap, &bg_color);
	XRecolorCursor(t_disp, curs, &fg_color, &bg_color);
	XDefineCursor(t_disp, frame->win, curs);

	frame->frames= 1;
	frame->framelist[0]= frame;
	XSelectInput(t_disp, frame->win,
		VisibilityChangeMask|StructureNotifyMask|ExposureMask|ButtonPressMask|ButtonReleaseMask|
		FocusChangeMask|EnterWindowMask|LeaveWindowMask|KeyPressMask|KeyReleaseMask
	);
	info = (struct b_info *) calloc( 1, sizeof(struct b_info));
	info->type= xtb_BT;
	info->func = func;
	info->text = STRDUP(text);
	XStoreName( t_disp, frame->win, info->text );
	info->font = norm_font;
	info->alt_font = greek_font;
	info->flag = 0;
	info->val = val;
	info->norm_pix = xtb_norm_pix;
	info->lighter_pix= xtb_lighter_pix;
	info->light_pix= xtb_light_pix;
	info->middle_pix= xtb_middle_pix;
	info->back_pix = xtb_back_pix;
	info->line_y = frame->height - 2;
	info->line_w = frame->width - 2*BT_HPAD;
	info->pos= pos;
	info->enter_flag= 0;
	info->focus_flag= 0;
	info->frame= frame;
	info->frame->enabled= True;
	frame->info= (xtb_registry_info*) info;
	xtb_register(frame, frame->win, bt_h, (xtb_registry_info*) info);
	XMapWindow(t_disp, frame->win);
	 /* Wait for the first event and dispatch it.
       \ Otherwise (it seems) the text is sometimes
       \ rendered in background colour?!
       */
	XNextEvent( t_disp, &evt);
	xtb_dispatch( t_disp, frame->win, 0, NULL, &evt);
	frame->width += (2*BT_BRDR);
	frame->height += (2*BT_BRDR);
	frame->border= BT_BRDR;
/*  frame->mapped= 1;   */
	frame->redraw= xtb_bt_redraw;
	frame->destroy= xtb_bt_del;
	frame->chfnt= xtb_bt_chfnt;
	if( debugFlag && debugLevel== -2 ){
		fprintf( StdErr, "%s(0x%lx,\"%s\",0x%lx,0x%lx)\n",
			caller, win, text, func, val, frame->win
		);
		fflush( StdErr );
	}
}

void xtb_bt_new( Window win, char *text,
	FNPTR( func, xtb_hret, (Window, int, xtb_data) ),
	xtb_data val, xtb_frame *frame)
/*
 * Makes a new button under `win' with the text `text'.  The
 * window, size, and position of the button are returned in `frame'.
 * The initial position is always (0, 0).  When the
 * button is pressed,  `func' will be called with the button
 * window,  the current state of the button,  and `val'.
 * It is up to `func' to change the state of the button (if desired).
 * The routine should return XTB_HANDLED normally and XTB_STOP if the
 * dialog should stop.  The window will be automatically mapped.
 */
{
#ifdef __GNUC__
	return( _xtb_bt_new( win, text, XTB_TOP_LEFT, func, val, frame, "xtb_bt_new" ) );
#else
	_xtb_bt_new( win, text, XTB_TOP_LEFT, func, val, frame, "xtb_bt_new" );
#endif
}

void xtb_bt_new2( Window win, char *text, int pos,
	FNPTR( func, xtb_hret, (Window, int, xtb_data) ),
	xtb_data val, xtb_frame *frame)
{
#ifdef __GNUC__
	return( _xtb_bt_new( win, text, pos, func, val, frame, "xtb_bt_new2" ) );
#else
	_xtb_bt_new( win, text, pos, func, val, frame, "xtb_bt_new2" );
#endif
}

int xtb_bt_get(Window win, xtb_data *stuff)
/*
 * Returns the state of button `win'.  If provided,  the button
 * specific info is returned in `info'.
 */
{
	struct b_info *info = (struct b_info *) xtb_lookup(win);

	if (stuff) *stuff = info->val;
	return info->flag;
}


int xtb_bt_chfnt( Window win, XFontStruct *norm_font, XFontStruct *greek_font )
{ struct b_info *info = (struct b_info *) xtb_lookup(win);
	if( info && norm_font && greek_font ){
		info->font= norm_font;
		info->alt_font= greek_font;
		bt_draw(win, info);
		return(0);
	}
	return(1);
}

int xtb_bt_set( Window win, int val, xtb_data stuff)
/*
 * Changes the value of a button and returns the new state.
 * The button is drawn.  If set,  the button specific info
 * will be set to `info'.  This doesn't allow you to set
 * the state to zero, but that may not be important.  The
 * change in button appearance will be immediate.
 */
{ struct b_info *info = (struct b_info *) xtb_lookup(win);

	if( !win || !info ){
		return 0;
	}
	if (stuff)
		info->val = stuff;
	if( info->flag!= val ){
		info->flag = val;
		bt_draw(win, info);
		if( !RemoteConnection ){
			XSync( t_disp, False );
		}
	}
	return info->flag;
}

int xtb_bt_swap(win)
Window win;
{
	struct b_info *info = (struct b_info *) xtb_lookup(win);

	if( !win || !info ){
		return 0;
	}
	info->flag = ! info->flag;
	if( info->frame && info->frame->mapped ){
		bt_draw(win, info);
		if( !RemoteConnection ){
			XSync( t_disp, False );
		}
	}
	return info->flag;
}

char* xtb_bt_get_text(win)
Window win;
/*
 * Changes the value of a button and returns the new state.
 * The button is drawn.  If set,  the button specific info
 * will be set to `info'.  This doesn't allow you to set
 * the state to zero, but that may not be important.  The
 * change in button appearance will be immediate.
 */
{
	struct b_info *info = (struct b_info *) xtb_lookup(win);

	return( (info)? info->text : NULL );
}

int xtb_bt_set2( Window win, int val, int val2, xtb_data stuff)
/*
 * Changes the value of a button and returns the new state.
 * The button is drawn.  If set,  the button specific info
 * will be set to `info'.  This doesn't allow you to set
 * the state to zero, but that may not be important.  The
 * change in button appearance will be immediate.
 */
{ struct b_info *info = (struct b_info *) xtb_lookup(win);

	if( !win || !info ){
		return 0;
	}
	if (stuff)
		info->val = stuff;
	if( info->flag!= val || info->flag2!= val2 ){
		info->flag = val;
		info->flag2 = val2;
		bt_draw(win, info);
		if( !RemoteConnection ){
			XSync( t_disp, False );
		}
	}
	return info->flag;
}

int xtb_bt_set_text(win, val, text, stuff)
Window win;
int val;
char *text;
xtb_data stuff;
/*
 * Changes the value of a button and returns the new state.
 * The button is drawn.  If set,  the button specific info
 * will be set to `info'.  This doesn't allow you to set
 * the state to zero, but that may not be important.  The
 * change in button appearance will be immediate.
 */
{ struct b_info *info = (struct b_info *) xtb_lookup(win);
   int change;

	if( !win || !info ){
		return 0;
	}
	if( stuff){
		info->val = stuff;
	}
	if( info->text && text ){
		change= strcmp( info->text, text );
	}
	else{
		change= (int) info->text;
	}
	if( change || info->flag!= val ){
		if( change ){
			if( info->text){
				xfree( info->text);
			}
			info->text= (text)? STRDUP(text) : NULL;
		}
		info->flag = val;
		XStoreName( t_disp, win, info->text );
		bt_draw(win, info);
	}
	return( info->flag);
}

/* Buttons have the nasty habit of not being drawn (remaining empty).
 \ Hence this function;
 */
void xtb_bt_redraw( win)
Window win;
{   struct b_info *ri= (struct b_info*) xtb_lookup(win);
	bt_draw( win, ri);
	return;
}

void xtb_bt_del(Window win, xtb_data *info)
/*
 * Deletes the button `win' and returns the user defined information
 * in `info' for destruction.
 */
{
	struct b_info *bi;

	if( !win ){
		return;
	}
	if (xtb_unregister(win, (xtb_registry_info* *) &bi)) {
		if( bi ){
			if( info ){
				*info = bi->val;
			}
			xfree( bi->text);
			if( bi->frame ){
				xfree( bi->frame->framelist );
			}
			xfree( bi);
		}
		else if( info ){
			*info= NULL;
		}
		XDestroyWindow(t_disp, win);
	}
}

typedef struct br_info {
	xtb_frame *frame;
	xtb_data val;		/* User data          */
	FNPTR( func, xtb_hret, (Window win, int prev, int this, xtb_data val) );
					/* Function to call   */
	FrameType type;
	Window main_win;		/* Main button row    */
	int which_one;		/* Which button is on */
	int btn_cnt;		/* How many buttons   */
	unsigned long norm_pix, light_pix, middle_pix, back_pix;
	Window *btns;		/* Button windows     */
} br_info;

/*ARGSUSED*/
static xtb_hret br_h(win, val, info)
Window win;
int val;
xtb_data info;
/*
 * This handles events for button rows.  When a button is pressed,
 * it turns off the button selected in `which_one' and turns itself
 * on.
 */
{
	struct br_info *real_info = (struct br_info *) info;
	int i, prev, Val= 1;

	prev = real_info->which_one;
	if ((prev >= 0) && (prev < real_info->btn_cnt)) {
		(void) xtb_bt_set(real_info->btns[prev], 0, (xtb_data) 0);
	}
	for (i = 0;   i < real_info->btn_cnt;   i++) {
		if (win == real_info->btns[i]) {
			if( i== real_info->which_one ){
				real_info->which_one= -1;
				Val= 0;
			}
			else{
				real_info->which_one = i;
				Val= 1;
			}
			break;
		}
	}
	(void) xtb_bt_set(win, Val, (xtb_data) 0);
	/* Callback */
	if (real_info->func) {
		return (*real_info->func)(real_info->main_win,
					 prev, real_info->which_one,
					 real_info->val);
	} else {
		return XTB_HANDLED;
	}
}

static xtb_hret brr_h(evt, info, parent)
XEvent *evt;
xtb_data info;
Window parent;
/*
 * Handles button events.
 */
{
/*     Window win = evt->xany.window;   */
	struct br_info *ri = (struct br_info *) info;
	xtb_hret rtn= XTB_NOTDEF;
	xtb_data data;
	int i;

	switch (evt->type) {
		case UnmapNotify:
			if( ri->frame ) ri->frame->mapped= 0;
			break;
		case MapNotify:
			if( ri->frame ) ri->frame->mapped= 1;
			break;
		case VisibilityNotify:
		case ConfigureNotify:
			if( ri->frame ) ri->frame->mapped= 1;
			break;
	}
	 /* Pass on some events to all buttons.
       \ This is maybe redundant. But then maybe not..
       */
	for( i= 0; i< ri->btn_cnt; i++ ){
		switch (evt->type) {
			case UnmapNotify:
			case MapNotify:
			case VisibilityNotify:
			case ConfigureNotify:
			case Expose:
/*              rtn = XTB_HANDLED;  */
				evt->xany.window= ri->btns[i];
				if( (data= xtb_lookup(evt->xany.window)) ){
					bt_h( evt, data, ri->main_win );
				}
				break;
		}
	}
	return( rtn );
}

void xtb_br_new( Window win, int cnt, char *lbls[], int initNr,
	FNPTR( func, xtb_hret, (Window, int, int, xtb_data) ),
	xtb_data val, xtb_frame *frame)
/*
 * This routine makes a new row of buttons in the window `win'
 * and returns a frame containing all of these buttons.  These
 * buttons are designed so that only one of them will be active
 * at a time.  Initially,  button `initNr' will be activated (if
 * initNr is less than zero,  none will be initially activated).
 * Whenever a button is pushed, `func' will be called with the
 * button row window,  the index of the previous button (-1 if
 * none),  the index of the current button,  and the user data, `val'.
 * The function is optional (if zero,  no function will be called).
 * The size of the row is returned in `frame'.  The window
 * will be automatically mapped.  Initially,  the window containing
 * the buttons will be placed at 0,0 in the parent window.
 */
{
	struct br_info *info;
	xtb_frame *sub_frame;
	int i, x, y;
	char name[128];

	if( !(info = (struct br_info *) calloc( 1, sizeof(struct br_info))) ){
		return;
	}
// 	if( !(sub_frame= (xtb_frame*) malloc(sizeof(xtb_frame))) ){
// 		xfree(info);
// 		return;
// 	}
	if( !(frame->framelist= (xtb_frame**) malloc(cnt*sizeof(xtb_frame*))) ){
		xfree(info);
// 		xfree(sub_frame);
		return;
	}
	if( SetParentBackground ){
		XSetWindowBackground( t_disp, win, xtb_light_pix );
	}
	frame->description= NULL;
	frame->frames= cnt;
	frame->parent= win;
	frame->width = frame->height = 0;
	frame->x_loc = frame->y_loc = 0;
	frame->win = xtb_XCreateSimpleWindow(t_disp, win, 0, 0, 1, 1,
					BT_BRDR*2, xtb_norm_pix, xtb_middle_pix);
	XSetWindowColormap( t_disp, frame->win, cmap );
	sprintf( name, "xtb_br frame #%d", xtb_br_serial++ );
	XStoreName( t_disp, frame->win, name );
	info->type= xtb_BR;
	info->main_win = frame->win;
	info->btns = (Window *) malloc((unsigned) (sizeof(Window) * cnt));
	info->btn_cnt = cnt;
	info->which_one = initNr;
	info->func = func;
	info->val = val;
	info->norm_pix = xtb_norm_pix;
	info->light_pix= xtb_light_pix;
	info->middle_pix= xtb_middle_pix;
	info->back_pix = xtb_back_pix;
	frame->info= (xtb_registry_info*) info;
	info->frame= frame;
	info->frame->enabled= True;

	if( !br_cursor ){
		br_cursor = XCreateFontCursor(t_disp, XC_hand2 );
	}
	fg_color.pixel = xtb_norm_pix;
	XQueryColor(t_disp, cmap, &fg_color);
	bg_color.pixel = xtb_back_pix;
	XQueryColor(t_disp, cmap, &bg_color);
	XRecolorCursor(t_disp, br_cursor, &fg_color, &bg_color);
	XDefineCursor(t_disp, frame->win, br_cursor);

	XSelectInput(t_disp, frame->win, ExposureMask|ButtonPressMask|ButtonReleaseMask|EnterWindowMask|LeaveWindowMask);
	 /* the handler is used simply to get information out */
/*     xtb_register(frame, frame->win, (xtb_hret (*)()) 0, (xtb_registry_info*) info);    */
	xtb_register(frame, frame->win, brr_h, (xtb_registry_info*) info);
	x = BR_XPAD;   y = BR_YPAD;
	for (i = 0;   i < cnt;   i++) {
		if( !(sub_frame= (xtb_frame*) malloc(sizeof(xtb_frame))) ){
			return;
		}
		xtb_bt_new( frame->win, lbls[i], br_h, (xtb_data) info, sub_frame);
/* 		XSetTransientForHint( t_disp, frame->win, sub_frame->win );	*/
		info->btns[i] = sub_frame->win;
		frame->framelist[i]= sub_frame;
		sub_frame->parent= win;
		XMoveWindow(t_disp, info->btns[i], x, y);
		x += (BR_INTER + sub_frame->width);
		if( sub_frame->height > frame->height){
			frame->height = sub_frame->height;
		}
		if( i == initNr){
			xtb_bt_set(info->btns[i], 1, (xtb_data) 0);
		}
		else{
			xtb_bt_set(info->btns[i], 0, (xtb_data) 0);
		}
	}
	frame->width = x - BR_INTER + BR_XPAD;
	frame->height += (2 * BR_YPAD);
	frame->border= 2* BT_BRDR;
	XResizeWindow(t_disp, frame->win, frame->width, frame->height);
	XMapWindow(t_disp, frame->win);
	XSetWindowBackground( t_disp, frame->win, xtb_middle_pix );
	frame->mapped= 1;
	frame->redraw= xtb_br_redraw;
	frame->destroy= (void*) xtb_br_del;
	frame->chfnt= xtb_br_chfnt;
	if( debugFlag && debugLevel== -2 ){
		fprintf( StdErr, "xtb_br_new(0x%lx,%d,{\"%s\",..},%d,0x%lx,0x%lx)\n",
			win, cnt, lbls[0], initNr, func, val, frame->win
		);
		fflush( StdErr );
	}
}

void xtb_br2D_new( Window win, int cnt, char *lbls[], int columns, int rows,
	FNPTR( format_fun, xtb_fmt*, (xtb_frame *br_frame, int cnt, xtb_frame **buttons, xtb_data val) ),
	int initNr,
	FNPTR( func, xtb_hret, (Window, int, int, xtb_data) ),
	xtb_data val, xtb_frame *frame)
/*
 * This routine makes a new row of buttons in the window `win'
 * and returns a frame containing all of these buttons.  These
 * buttons are designed so that only one of them will be active
 * at a time.  Initially,  button `initNr' will be activated (if
 * initNr is less than zero,  none will be initially activated).
 * Whenever a button is pushed, `func' will be called with the
 * button row window,  the index of the previous button (-1 if
 * none),  the index of the current button,  and the user data, `val'.
 * The function is optional (if zero,  no function will be called).
 * The size of the row is returned in `frame'.  The window
 * will be automatically mapped.  Initially,  the window containing
 * the buttons will be placed at 0,0 in the parent window.
 */
{
	struct br_info *info;
	xtb_frame *sub_frame;
	int i, x, y, col= 0;
	char name[128];
	xtb_fmt **frows= NULL, *vertical, *format;

	if( columns< 0 && rows> 0 ){
		columns= cnt/ rows;
		while( columns* rows< cnt ){
			columns+= 1;
		}
	}
	else if( rows< 0 && columns> 0 ){
		rows= cnt/ columns;
		while( columns* rows< cnt ){
			rows+= 1;
		}
	}
	else if( !format_fun ){
		fprintf( StdErr, "xtb_br2D_new() columns==%d, rows==%d and format_fun==NULL: impossible to proceed!\n", columns, rows );
		return;
	}
	if( !format_fun ){
		if( !(frows= (xtb_fmt**) calloc( rows, sizeof(xtb_fmt*))) ){
			fprintf( StdErr, "xtb_br2D_new(): error allocating a %d element array (%s)\n", rows );
			return;
		}
		memset( frows, 0, rows* sizeof(xtb_fmt*) );
	}
	if( !(info = (struct br_info *) calloc( 1, sizeof(struct br_info))) ){
		return;
	}
	if( !(frame->framelist= (xtb_frame**) malloc(cnt*sizeof(xtb_frame*))) ){
		return;
	}
	if( SetParentBackground )
		XSetWindowBackground( t_disp, win, xtb_light_pix );
	frame->description= NULL;
	frame->frames= cnt;
	frame->parent= win;
	frame->width = frame->height = 0;
	frame->x_loc = frame->y_loc = 0;
	frame->win = xtb_XCreateSimpleWindow(t_disp, win, 0, 0, 1, 1,
					BT_BRDR*2, xtb_norm_pix, xtb_middle_pix);
	XSetWindowColormap( t_disp, frame->win, cmap );
	sprintf( name, "xtb_br frame #%d", xtb_br_serial++ );
	XStoreName( t_disp, frame->win, name );
	info->type= xtb_BR;
	info->main_win = frame->win;
	info->btns = (Window *) malloc((unsigned) (sizeof(Window) * cnt));
	info->btn_cnt = cnt;
	info->which_one = initNr;
	info->func = func;
	info->val = val;
	info->norm_pix = xtb_norm_pix;
	info->light_pix= xtb_light_pix;
	info->middle_pix= xtb_middle_pix;
	info->back_pix = xtb_back_pix;
	frame->info= (xtb_registry_info*) info;
	info->frame= frame;
	info->frame->enabled= True;

	if( !br_cursor ){
		br_cursor = XCreateFontCursor(t_disp, XC_hand2 );
	}
	fg_color.pixel = xtb_norm_pix;
	XQueryColor(t_disp, cmap, &fg_color);
	bg_color.pixel = xtb_back_pix;
	XQueryColor(t_disp, cmap, &bg_color);
	XRecolorCursor(t_disp, br_cursor, &fg_color, &bg_color);
	XDefineCursor(t_disp, frame->win, br_cursor);

	XSelectInput(t_disp, frame->win, ExposureMask|ButtonPressMask|ButtonReleaseMask|EnterWindowMask|LeaveWindowMask);
	 /* the handler is used simply to get information out */
/*     xtb_register(frame, frame->win, (xtb_hret (*)()) 0, (xtb_registry_info*) info);    */
	xtb_register(frame, frame->win, brr_h, (xtb_registry_info*) info);
	x = BR_XPAD;   y = BR_YPAD;
	for (i = 0;   i < cnt;   i++) {
		if( !(sub_frame= (xtb_frame*) malloc(sizeof(xtb_frame))) ){
			return;
		}
		xtb_bt_new( frame->win, lbls[i], br_h, (xtb_data) info, sub_frame);
/* 		XSetTransientForHint( t_disp, frame->win, sub_frame->win );	*/
		info->btns[i] = sub_frame->win;
		frame->framelist[i]= sub_frame;
		sub_frame->parent= win;
		if( i == initNr){
			xtb_bt_set(info->btns[i], 1, (xtb_data) 0);
		}
		else{
			xtb_bt_set(info->btns[i], 0, (xtb_data) 0);
		}
		sub_frame->mapped= 1;
		if( frows ){
			if( i> 0 && (i % columns)== 0 ){
				frows[col]= xtb_hort_cum( XTB_CENTER, 0, BR_INTER, NULL );
				col+= 1;
			}
			frows[col]= xtb_hort_cum( XTB_CENTER, BR_XPAD, BR_INTER, xtb_w(sub_frame) );
		}
	}
	if( frows ){
		frows[col]= xtb_hort_cum( XTB_CENTER, 0, BR_INTER, NULL );

		for( i= 0; i< rows; i++ ){
			vertical= xtb_vert_cum( XTB_CENTER, 0, 0, frows[i] );
		}
		vertical= xtb_vert_cum( XTB_CENTER, 0, BR_INTER, NULL );
		format= xtb_fmt_do( vertical, &frame->width, &frame->height );
	}
	else if( format_fun ){
		format= (*format_fun)( frame, cnt, frame->framelist, val );
	}
	for( i= 0; i< cnt; i++ ){
		xtb_mv_frame( frame->framelist[i] );
	}
	xtb_fmt_free(format);
/* 	frame->width+= 2* BR_XPAD;	*/
/* 	frame->height+= 2* BR_YPAD;	*/

	frame->border= 2* BT_BRDR;
	XResizeWindow(t_disp, frame->win, frame->width, frame->height);
	XMapWindow(t_disp, frame->win);
	XSetWindowBackground( t_disp, frame->win, xtb_middle_pix );
	frame->mapped= 1;
	frame->redraw= xtb_br_redraw;
	frame->destroy= (void*) xtb_br_del;
	frame->chfnt= xtb_br_chfnt;
	if( debugFlag && debugLevel== -2 ){
		fprintf( StdErr, "xtb_br2Dr_new(0x%lx,%d,{\"%s\",..},%d,%d,0x%lx,0x%lx)\n",
			win, cnt, lbls[0], rows, initNr, func, val, frame->win
		);
		fflush( StdErr );
	}
	xfree( frows );
}

void xtb_br2Dr_new( Window win, int cnt, char *lbls[], int rows, int initNr,
	FNPTR( func, xtb_hret, (Window, int, int, xtb_data) ),
	xtb_data val, xtb_frame *frame)
{
	if( rows<= 0 || cnt<= 0 ){
		fprintf( StdErr, "xtb_br2Dr_new(): called with invalid rows (%d) or button count number (%d)\n", rows, cnt );
		return;
	}
	xtb_br2D_new( win, cnt, lbls, -1, rows, NULL, initNr, func, val, frame );
}

void xtb_br2Dc_new( Window win, int cnt, char *lbls[], int columns, int initNr,
	FNPTR( func, xtb_hret, (Window, int, int, xtb_data) ),
	xtb_data val, xtb_frame *frame)
{
	if( columns<= 0 || cnt<= 0 ){
		fprintf( StdErr, "xtb_br2Dc_new(): called with invalid columns (%d) or button count number (%d)\n", columns, cnt );
		return;
	}
	xtb_br2D_new( win, cnt, lbls, columns, -1, NULL, initNr, func, val, frame );
}

int xtb_br_chfnt( Window win, XFontStruct *norm_font, XFontStruct *greek_font )
{ struct br_info *info = (struct br_info *) xtb_lookup(win);
	if( info && norm_font && greek_font ){
	  int i;
		for( i= 0; i< info->btn_cnt; i++ ){
			xtb_bt_chfnt( info->btns[i], norm_font, greek_font );
		}
		return(0);
	}
	return(1);
}

int xtb_br_set(win, button)
Window win;
int button;
{
	struct br_info *info = (struct br_info *) xtb_lookup(win);

	if( info ){
		if( button< 0 ){
			if( info->which_one>= 0 ){
				xtb_bt_set( info->btns[info->which_one], 0, NULL);
			}
		}
		else if( button< info->btn_cnt ){
			if( button!= info->which_one ){
				if( info->which_one>= 0 ){
					xtb_bt_set( info->btns[info->which_one], 0, NULL);
				}
				xtb_bt_set( info->btns[button], 1, NULL);
			}
			else if( !xtb_bt_get( info->btns[button], NULL) ){
				xtb_bt_set( info->btns[button], 1, NULL);
			}
		}
		return( info->which_one= button );
	}
	return( 0 );
}

int xtb_br_get(win)
Window win;
/*
 * This routine returns the index of the currently selected item of
 * the button row given by the window `win'.  Note:  no checking
 * is done to make sure `win' is a button row.
 */
{
	struct br_info *info = (struct br_info *) xtb_lookup(win);

	if( info ){
		return info->which_one;
	}
	else{
		return(-1);
	}
}

void xtb_br_redraw( Window win)
{   struct br_info *ri= (struct br_info*) xtb_lookup(win);
    int i;
    xtb_data data;

	for( i= 0; i< ri->btn_cnt; i++ ){
		if( (data= xtb_lookup(ri->btns[i])) ){
			bt_draw( ri->btns[i], data );
		}
	}
	return;
}

void xtb_br_del(Window win)
/*
 * Deletes a button row.  All resources are reclaimed.
 */
{
	struct br_info *info;
	int i;

	if (xtb_unregister(win, (xtb_registry_info* *) &info)) {
		if( info ){
			for (i = 0;   i < info->btn_cnt;   i++) {
				xtb_bt_del(info->btns[i], NULL );
			}
			xfree( info->btns);
			if( info->frame ){
				for (i = 0;   i < info->btn_cnt;   i++) {
					xfree( info->frame->framelist[i] );
				}
				xfree( info->frame->framelist );
			}
			xfree( info);
		}
		XDestroyWindow(t_disp, win);
	}
}

/* Text widget */

#define TO_HPAD 1
#define TO_VPAD 2

static void to_draw(win, ri)
Window win;
struct to_info *ri;
/*
 * Draws the text for a widget
 */
{
	XCharStruct bb;
	int dir, ascent, descent, text_v_info= False,
		x_text_offset, y_text_offset;

#define TO_WIDTH(bb)	(bb.rbearing-bb.lbearing+ 2*TO_HPAD)

	if( !ri || !ri->frame || !ri->frame->mapped ){
		return;
	}

	xtb_TextExtents( *ri->ft, *ri->ft2, ri->text, strlen(ri->text), &dir, &ascent, &descent, &bb, False);

	switch( ri->pos ){
		default:
		case XTB_TOP_LEFT:
			x_text_offset= 0;
			y_text_offset= 0;
			break;
		case XTB_TOP_RIGHT:
			x_text_offset= -bb.lbearing+ TO_HPAD;
			y_text_offset= 0;
			break;
		case XTB_CENTERED:
			 /* This works best with cursorfont (what about others?! - never tried!)   */
			x_text_offset= -bb.lbearing+ (ri->frame->width - TO_WIDTH(bb))/2.0 ;
			y_text_offset= 0;
			text_v_info= True;
			break;
	}

	XClearWindow(t_disp, win);
	xtb_DrawString(t_disp, win,
			xtb_set_gc(win, ri->norm_pix, ri->light_pix, (*ri->ft)->fid),
/* 			x_text_offset+ TO_HPAD, y_text_offset+ TO_VPAD,	*/
			x_text_offset+ TO_HPAD, y_text_offset+ TO_VPAD- (*ri->ft)->descent,
			ri->text, strlen(ri->text), *ri->ft, *ri->ft2, text_v_info, False
	);
/*
    xtb_DrawString(t_disp, win,
             xtb_set_gc(win, ri->norm_pix, ri->light_pix, (*ri->ft)->fid),
             TO_HPAD, TO_VPAD,
             ri->text, strlen(ri->text), *ri->ft, *ri->ft2, False, False
    );
 */
}

static xtb_hret to_h(evt, info)
XEvent *evt;
xtb_data info;
/*
 * Handles text widget events
 */
{
	Window win = evt->xany.window;
	struct to_info *ri = (struct to_info *) info;
	// 20080711: if new, unwanted behaviour, set hret to default value XTB_NOTDEF!
	xtb_hret hret= XTB_HANDLED;

	switch (evt->type) {
		case UnmapNotify:
			if( ri->frame ) ri->frame->mapped= 0;
			break;
		case MapNotify:
			if( ri->frame ) ri->frame->mapped= 1;
			break;
		case EnterNotify:
			ri->entered= 1;
			if( ri->text_return ){
				*(ri->text_return)= ri->text;
			}
			if( ri->text_id_return ){
				*(ri->text_id_return)= ri->id;
			}
			XSetWindowBorder( t_disp, win, ri->norm_pix );
			to_draw( win, ri );
			xtb_XSync( t_disp, False );
			hret= XTB_HANDLED;
			break;
		case LeaveNotify:
			ri->entered= 0;
			if( ri->text_return ){
				*(ri->text_return)= NULL;
			}
			if( ri->text_id_return ){
				*(ri->text_id_return)= ri->id;
			}
			XSetWindowBorder( t_disp, win, ri->light_pix );
			to_draw( win, ri );
			xtb_XSync( t_disp, False );
			hret= XTB_HANDLED;
			break;
		case VisibilityNotify:
			if( ri->frame ) ri->frame->mapped= 1;
			  /* FALL THROUGH */
		case Expose:
			to_draw(win, ri);
			hret= XTB_HANDLED;
			break;
		case ButtonPress:
		/* if Button3, copy contents into cutbuffer 0 */
		{   Window focus;
			int revert;
			XGetInputFocus( t_disp, &focus, &revert);
			if( focus!= win && focus!= PointerRoot ){
			  /* Send the event to the window that *has* the focus.  */
				evt->xany.window= focus;
				XSendEvent( t_disp, focus, 0,
					VisibilityChangeMask|StructureNotifyMask|ExposureMask|KeyPressMask|EnterWindowMask|LeaveWindowMask|
					FocusChangeMask|ButtonPressMask|PointerMotionMask|PointerMotionHintMask|ButtonReleaseMask,
					evt
				);
			}
			else{
			  XButtonEvent *bev= (XButtonEvent*) evt;
				switch( bev->button){
					case Button3:
						if( *(ri->text) ){
							XStoreBuffer( t_disp, ri->text, strlen(ri->text), 0);
						}
						break;
				}
			}
			break;
		}
		default:
			hret= XTB_NOTDEF;
			break;
	}
	return hret;
}

void xtb_to_new( Window win, char *text, int pos, XFontStruct **ft, XFontStruct **ft2, xtb_frame *frame)
/*
 * Makes a new text widget under `win' with the text `text'.
 * The size of the widget is returned in `w' and `h'.  The
 * window is created and mapped at 0,0 in `win'.  The font
 * used for the text is given in `ft'.
 */
{
	struct to_info *info;
	XCharStruct bb;
	int dir, ascent, descent;
	Cursor curs= to_cursor;
	char name[128];

	xtb_TextExtents( (ft && *ft)? *ft : norm_font, (ft2 && *ft2)? *ft2 : greek_font, text, strlen(text),
		&dir, &ascent, &descent, &bb, False);
	frame->description= NULL;
	frame->parent= win;
/*     frame->width = bb.width + 2*TO_HPAD; */
	frame->width = TO_WIDTH(bb);
/*     frame->height = max_ascent + max_descent + 2* TO_VPAD;   */
	frame->height = ascent + descent + 2* TO_VPAD;
	frame->border= 1;
	frame->x_loc = frame->y_loc = 0;
	frame->win = xtb_XCreateSimpleWindow(t_disp, win, 0, 0,
					frame->width, frame->height, 1,
					xtb_light_pix, xtb_light_pix);
	XSetWindowColormap( t_disp, frame->win, cmap );
	sprintf( name, "xtb_to frame #%d", xtb_to_serial++ );
	XStoreName( t_disp, frame->win, name );
	frame->frames= 1;
	if( !(frame->framelist= (xtb_frame**) malloc(sizeof(xtb_frame*))) ){
		return;
	}
	frame->framelist[0]= frame;
	XSelectInput(t_disp, frame->win, VisibilityChangeMask|StructureNotifyMask|ButtonReleaseMask|ButtonPressMask|ExposureMask|EnterWindowMask|LeaveWindowMask);
	info = (struct to_info *) calloc(1, sizeof(struct to_info));
	info->type= xtb_TO;
	info->text = STRDUP(text);
	info->len= strlen(info->text);
	info->pos= pos;
	info->ft = (ft && *ft)? ft : &norm_font;
	info->ft2= (ft2 && *ft2)? ft2 : &greek_font;
	if( text[0]== 0x01 ){
		info->norm_pix= xtb_back_pix;
		info->light_pix= xtb_norm_pix;
		info->back_pix= xtb_light_pix;
	}
	else{
		info->norm_pix= xtb_norm_pix;
		info->light_pix= xtb_light_pix;
		info->back_pix= xtb_back_pix;
	}
	info->frame= frame;
	info->frame->enabled= True;
	frame->info= (xtb_registry_info*) info;
	if( !curs ){
		curs = XCreateFontCursor(t_disp, XC_center_ptr );
		to_cursor= curs;
	}
	fg_color.pixel = xtb_norm_pix;
	XQueryColor(t_disp, cmap, &fg_color);
	bg_color.pixel = xtb_back_pix;
	XQueryColor(t_disp, cmap, &bg_color);
	XRecolorCursor(t_disp, curs, &fg_color, &bg_color);
	XDefineCursor(t_disp, frame->win, curs);

	xtb_register(frame, frame->win, to_h, (xtb_registry_info*) info);
	XMapWindow(t_disp, frame->win);
	XNextEvent( t_disp, &evt);
	xtb_dispatch( t_disp, frame->win, 0, NULL, &evt);
	frame->redraw= xtb_to_redraw;
	frame->destroy= (void*) xtb_to_del;
	frame->chfnt= NULL;
/*  frame->mapped= 1;   */
}

void xtb_to_new2( Window win, char *text, int maxwidth, int pos, XFontStruct **ft, XFontStruct **ft2, xtb_frame *frame)
/*
 * Makes a new text widget under `win' with the text `text'.
 * The size of the widget is returned in `w' and `h'.  The
 * window is created and mapped at 0,0 in `win'.  The font
 * used for the text is given in `ft'.
 */
{
	struct to_info *info;
	XCharStruct bb;
	int dir, ascent, descent;
	Cursor curs= to_cursor;
	char name[128];

	xtb_TextExtents( (ft && *ft)? *ft : norm_font, (ft2 && *ft2)? *ft2 : greek_font, text, strlen(text),
		&dir, &ascent, &descent, &bb, False);
	frame->description= NULL;
	frame->parent= win;
	if( maxwidth ){
		frame->width= XFontWidth((ft && *ft)? *ft : norm_font)* maxwidth;
	}
	else{
/*     frame->width = bb.width + 2*TO_HPAD; */
		frame->width = TO_WIDTH(bb);
	}
/*     frame->height = max_ascent + max_descent + 2* TO_VPAD;   */
	frame->height = ascent + descent + 2* TO_VPAD;
	frame->x_loc = frame->y_loc = 0;
	frame->win = xtb_XCreateSimpleWindow(t_disp, win, 0, 0,
					frame->width, frame->height, 1,
					xtb_light_pix, xtb_light_pix);
	XSetWindowColormap( t_disp, frame->win, cmap );
	sprintf( name, "xtb_to frame #%d", xtb_to_serial++ );
	XStoreName( t_disp, frame->win, name );
	frame->frames= 1;
	if( !(frame->framelist= (xtb_frame**) malloc(sizeof(xtb_frame*))) ){
		return;
	}
	frame->framelist[0]= frame;
	XSelectInput(t_disp, frame->win, VisibilityChangeMask|StructureNotifyMask|ButtonReleaseMask|ButtonPressMask|ExposureMask|EnterWindowMask|LeaveWindowMask);
	info = (struct to_info *) calloc( 1, sizeof(struct to_info));
	info->type= xtb_TO;
	info->text = STRDUP(text);
	info->len= strlen(info->text);
	info->pos= pos;
	info->ft = (ft && *ft)? ft : &norm_font;
	info->ft2= (ft2 && *ft2)? ft2 : &greek_font;
	if( text[0]== 0x01 ){
		info->norm_pix= xtb_back_pix;
		info->light_pix= xtb_norm_pix;
		info->back_pix= xtb_light_pix;
	}
	else{
		info->norm_pix= xtb_norm_pix;
		info->light_pix= xtb_light_pix;
		info->back_pix= xtb_back_pix;
	}
	info->frame= frame;
	info->frame->enabled= True;
	frame->info= (xtb_registry_info*) info;
	if( !curs ){
		curs = XCreateFontCursor(t_disp, XC_center_ptr );
		to_cursor= curs;
	}
	fg_color.pixel = xtb_norm_pix;
	XQueryColor(t_disp, cmap, &fg_color);
	bg_color.pixel = xtb_back_pix;
	XQueryColor(t_disp, cmap, &bg_color);
	XRecolorCursor(t_disp, curs, &fg_color, &bg_color);
	XDefineCursor(t_disp, frame->win, curs);

	xtb_register(frame, frame->win, to_h, (xtb_registry_info*) info);
	XMapWindow(t_disp, frame->win);
	XNextEvent( t_disp, &evt);
	xtb_dispatch( t_disp, frame->win, 0, NULL, &evt);
/*  frame->mapped= 1;   */
	frame->redraw= xtb_to_redraw;
	frame->destroy= (void*) xtb_to_del;
	frame->chfnt= NULL;
}

void xtb_to_redraw( win)
Window win;
{   struct to_info *ri= (struct to_info*) xtb_lookup(win);
	to_draw( win, ri);
	return;
}

void xtb_to_del(Window win)
/*
 * Deletes an output only text widget.
 */
{
	struct to_info *info;

	if (xtb_unregister(win, (xtb_registry_info* *) &info)) {
		if( info ){
			xfree( info->text);
			if( info->frame ){
				xfree( info->frame->framelist );
			}
			xfree( info);
		}
		XDestroyWindow(t_disp, win);
	}
}

void xtb_to_set(win, text)
Window win;
char *text;
{ struct to_info *ri = (struct to_info *) xtb_lookup(win);
   int change, len= (text)? strlen(text) : 0;
	if( !ri ){
		return;
	}
	if( text && ri->text ){
		change= strcmp( text, ri->text );
	}
	else{
		change= 1;
	}
	if( !ri->text || len> ri->len ){
		if( (ri->text= realloc( ri->text, (len+ 2)* sizeof(char))) ){
			ri->len= len+ 1;
		}
	}
	if( change ){
		if( ri->text && text ){
			strcpy( ri->text, text );
		}
		to_draw(win, ri);
	}
}

/* For debugging */
void focus_evt(XEvent *evt)
{
	switch (evt->xfocus.mode) {
		case NotifyNormal:
			printf("NotifyNormal");
			break;
		case NotifyGrab:
			printf("NotifyGrab");
			break;
		case NotifyUngrab:
			printf("NotifyUngrab");
			break;
	}
	printf(", detail = ");
	switch (evt->xfocus.detail) {
		case NotifyAncestor:
			printf("NotifyAncestor");
			break;
		case NotifyVirtual:
			printf("NotifyVirtual");
			break;
		case NotifyInferior:
			printf("NotifyInferior");
			break;
		case NotifyNonlinear:
			printf("NotifyNonLinear");
			break;
		case NotifyNonlinearVirtual:
			printf("NotifyNonLinearVirtual");
			break;
		case NotifyPointer:
			printf("NotifyPointer");
			break;
		case NotifyPointerRoot:
			printf("NotifyPointerRoot");
			break;
		case NotifyDetailNone:
			printf("NotifyDetailNone");
			break;
	}
	printf("\n");
}

/*
 * Input text widget
 */

static int text_width( XFontStruct *font, char *str, int len, ti_info *info)
/*
 * Returns the width of a string using XTextExtents.
 */
{
	XCharStruct bb;
	int dir, ascent, descent;
	char *text;

	  /* 20020313:
	   \ Determine width using the displayed version of the string...!
	   */
	if( info->enter_flag && info->rich_text ){
		text= xtb_textfilter( str, font, True );
	}
	else{
		text= xtb_textfilter( str, font, False );
	}
	XTextExtents(font, text, len, &dir, &ascent, &descent, &bb);
	return bb.width;
}

static void ti_cursor_on(win, ri)
Window win;
struct ti_info *ri;
/*
 * Draws the cursor for the window.  Uses pixel `pix'.
 */
{
	ri->ti_crsp= (ri->curidx== ri->curlen)? TI_CRSP : -TI_CRSP;
	XFillRectangle(t_disp, win,
		   xtb_set_gc(win, ri->norm_pix, ri->back_pix, ri->font->fid),
/* 		   ri->curxval- ri->startxval + TI_HPAD + ri->ti_crsp, TI_VPAD,	*/
		   ri->curxval- ri->startxval + TI_HPAD + ri->ti_crsp,
		   ri->frame->height- TI_VPAD- TI_LPAD- max_descent- ri->cursor_height- 1,
		   (ri->focus_flag ? 2 : 1),
		   ri->cursor_height
	 );
}

static void ti_cursor_off(win, ri)
Window win;
struct ti_info *ri;
/*
 * Draws the cursor for the window.  Uses pixel `pix'.
 */
{
	XFillRectangle(t_disp, win,
		   xtb_set_gc(win, ri->back_pix, ri->back_pix, ri->font->fid),
		   ri->curxval- ri->startxval + TI_HPAD + ri->ti_crsp,
		   ri->frame->height- TI_VPAD- TI_LPAD- max_descent- ri->cursor_height- 1,
		   (ri->focus_flag ? 2 : 1),
		   ri->cursor_height
	);
}

static void ti_line(win, ri, pix)
Window win;
struct ti_info *ri;
unsigned long pix;
/*
 * Draws a status line beneath the text in a text widget to indicate
 * the user has moved into the text field.
 */
{   Window focus;
    int revert;
	XGetInputFocus( t_disp, &focus, &revert);
	if( focus== win || focus== PointerRoot || focus== ri->frame->parent ){
	 /* If the focus is on another (ti?) window, we don't
       \ show/remove the line indicating that we will get/loose
       \ input events.
       */
		XDrawLine(t_disp, win,
/*            xtb_set_gc(win, pix, ri->_back_pix, ri->font->fid),   */
			 xtb_set_gc(win, pix, ri->light_pix, ri->font->fid),
			 TI_HPAD, ri->line_y, TI_HPAD+ri->line_w, ri->line_y);
	}
}

static void ti_draw( Window win, struct ti_info *ri, int c_flag)
/*
 * Draws the indicated text widget.  This includes drawing the
 * text and cursor.  If `c_flag' is set,  the window will
 * be cleared first.
 */
{ XPoint line[3];
   GC lineGC;
   xtb_frame *frame= ri->frame;
   int height, width;

	if( !frame || !frame->mapped ){
		return;
	}
	height= frame->height- 2* TI_BRDR;
	width= frame->width- 2* TI_BRDR;

	if( ri->frame->enabled ){
		XSetWindowBackgroundPixmap( t_disp, win, None );
		XSetWindowBackground( t_disp, win, ri->back_pix );
	}
	else{
		XSetWindowBackground( t_disp, win, ri->light_pix);
		XSetWindowBackgroundPixmap( t_disp, win, disabled_bg_light );
	}
	if( c_flag){
		XClearWindow(t_disp, win);
		if( ri->clear_flag ){
			ri->clear_flag= 0;
		}
	}

		 /* lower  */
		line[0].x= 1;
		line[0].y= (short) height- 1;
		line[1].x= (short) width- 1;
		line[1].y= (short) height- 1;
		line[2].x= (short) width- 1;
		line[2].y= 0;
		lineGC= xtb_set_gc(win, ri->light_pix, ri->back_pix, ri->font->fid),
		XSetLineAttributes( t_disp, lineGC, 2, LineSolid, CapButt, JoinMiter);
		XDrawLines(t_disp, win, lineGC, line, 3, CoordModeOrigin);

		 /* upper  */
		line[0].x= line[1].x= line[1].y= line[2].y= 1;
		line[0].y= (short) height- 1;
		line[2].x= (short) width- 1;
		lineGC= xtb_set_gc(win, ri->middle_pix, ri->back_pix, ri->font->fid),
		XSetLineAttributes( t_disp, lineGC, 2, LineSolid, CapButt, JoinMiter);
		XDrawLines(t_disp, win, lineGC, line, 3, CoordModeOrigin);

	 /* Text */
	if( ri->enter_flag && ri->rich_text ){
		 /* (re)draw the text so that the codes become visible:    */
		t_font= ri->font;
		xtb_DrawString(t_disp, win,
				xtb_set_gc(win, ri->norm_pix, ri->back_pix, ri->font->fid),
				TI_HPAD, TI_VPAD+ ri->max_ascent,
				&ri->text[ri->startidx], strlen(&ri->text[ri->startidx]), NULL, NULL, False, True
		);
	}
	else{
		 /* Draw the text as it will be shown. */
		ri->rich_text= xtb_DrawString(t_disp, win,
				xtb_set_gc( win, ri->norm_pix, ri->back_pix, ri->font->fid),
				TI_HPAD, TI_VPAD,
				&ri->text[ri->startidx], strlen(&ri->text[ri->startidx]), ri->font, ri->alt_font, False, False
		);
	}

/*  ti_line( win, ri, (ri->enter_flag)? ri->_norm_pix : ri->_back_pix );    */
	ti_line( win, ri, (ri->enter_flag)? ri->_norm_pix : ri->light_pix );
	 /* Cursor */
	ti_cursor_on(win, ri);
}



char *CharToString( int c)
{  static char string[2]= {'\0','\0'};
	string[0]= c;
	return( string );
}

static void ti_move_right( Window win, ti_info *ri )
{
	if( ri->curidx< ri->curlen ){
	  int inspos;
		ri->curidx += 1;
		ri->curxval = text_width(norm_font, ri->text, ri->curidx, ri);
		if( (inspos= ri->curxval- ri->startxval+ TI_HPAD)> ri->line_w ){
			xtb_ti_scroll_left( win, (inspos- ri->line_w- 1)+ ri->maxwidth/2 );
		}
	}
	else{
		Boing(10);
	}
}

static void ti_move_left( Window win, ti_info *ri )
{
	if( ri->curidx> 0 ){
	  int inspos;
		ri->curidx -= 1;
		ri->curxval = text_width(norm_font, ri->text, ri->curidx, ri);
		if( (inspos= ri->curxval- ri->startxval+ TI_HPAD)< 0 ){
			xtb_ti_scroll_right( win, (0- inspos- 1)+ ri->maxwidth/2 );
		}
	}
	else{
		Boing(10);
	}
}

static int ri_CopyClipboard(struct ti_info *ri, Window win, char *buffer, xtb_hret *rtn)
{ int i= 0, nbytes= 0;
	if( ri->frame->enabled ){
	  int stop= False;
	  char *clipboard= XFetchBuffer( t_disp, &nbytes, 0);
		for( i= 0; i< nbytes && !stop; i++ ){
		  int xcc= xtb_clipboard_copy;
			xtb_clipboard_copy= True;
			if( ri->text ){
				strcpy( buffer, ri->text);
			}
			else{
				buffer[0]= '\0';
			}
			if( (*rtn = (*ri->func)(win, (int) clipboard[i],
						   buffer, ri->val
						)
				)== XTB_STOP
			){
				stop= True;
			}
			xtb_clipboard_copy= xcc;
		}
		  /* 20040922 */
		XFree(clipboard);
	}
	return(i);
}

static xtb_hret ti_h( XEvent *evt, xtb_data info, Window parent )
/*
 * Handles text input events.
 */
{
	Window win = evt->xany.window;
	struct ti_info *ri = (struct ti_info *) info;
	__ALLOCA( keys, char, ri->maxlen, keyslen);
	__ALLOCA( textcopy, char, ri->maxlen, copylen);
	__ALLOCA( keysyms, KeySym, ri->maxlen, symlen);
	// 20080711: if new, unwanted behaviour, set rtn to default value XTB_NOTDEF!
	xtb_hret rtn= XTB_HANDLED;
	int nbytes, i;
	XButtonEvent *bev= (XButtonEvent*) evt;

	switch (evt->type) {
		case UnmapNotify:
			if( ri->frame ) ri->frame->mapped= 0;
			break;
		case MapNotify:
			if( ri->frame ) ri->frame->mapped= 1;
			break;
		case ConfigureNotify:
		case VisibilityNotify:
			if( ri->frame ) ri->frame->mapped= 1;
		case Expose:
			ti_draw(win, ri, ri->clear_flag );	rtn = XTB_HANDLED;
			if( ri->_entered ){
				ri->enter_flag= 0;
			}
			break;
		case KeyPress:{
			if( !ri->frame->enabled ){
				rtn= XTB_HANDLED;
				Boing(5);
				break;
			}
			nbytes = XLookupString(&evt->xkey, keys, MAXKEYS,
						   (KeySym *) keysyms, (XComposeStatus *) 0);
/* 			keysyms[0]&= 0x0000FFFF;	*/
			keysyms[0]= XLookupKeysym( &evt->xkey, 0);
					 /* 20050115: */
					xtb_modifier_state= xtb_Mod2toMod1( 0xFF & evt->xkey.state );
			if( nbytes== 0 &&
				(keysyms[0]!= XK_Shift_L && keysyms[0]!= XK_Shift_R &&
					keysyms[0]!= XK_Control_L && keysyms[0]!= XK_Control_R &&
					keysyms[0]!= XK_Caps_Lock && keysyms[0]!= XK_Mode_switch
				)
			){
				strcpy(textcopy, ri->text);
				if( debugFlag && debugLevel== -2 ){
					fprintf( StdErr, "xtb_ti_ins(\"%s\",%s=0x%x)\n", textcopy,
						XKeysymToString(keysyms[0]), keysyms[0]
					);
					fflush( StdErr );
				}
				 /* 9811: rudimentary "real" editing: move cursor left/right with up/down
				  \ keys.. because xgraph already uses the left/right keys for accepting
				  \ input (possibly decrementing/incrementing numerical values first).
				  \ Could be XGRAPH conditional code, using the more logical left/right
				  \ keys otherwise ;-)
				  \ 981202: Changed into the more intuitive version... ;-)
				  */
				if( keysyms[0]== XK_Right ){
					ti_move_right( win, ri );
					ti_draw(win, ri, True );
					rtn= XTB_HANDLED;
				}
				else if( keysyms[0]== XK_Left ){
					ti_move_left( win, ri );
					ti_draw(win, ri, True );
					rtn= XTB_HANDLED;
				}
				else if( keysyms[0]== XK_Home ){
				  int inspos;
					ri->curidx= 0;
					ri->curxval= text_width(norm_font, ri->text, ri->curidx, ri);
					if( (inspos= ri->curxval- ri->startxval+ TI_HPAD)< 0 ){
						xtb_ti_scroll_right( win, (0- inspos- 1)+ ri->maxwidth/2 );
					}
					ti_draw(win, ri, True );
					rtn= XTB_HANDLED;
				}
				else if( keysyms[0]== XK_End ){
				  int inspos;
					ri->curidx= ri->curlen;
					ri->curxval= text_width(norm_font, ri->text, ri->curidx, ri);
					if( (inspos= ri->curxval- ri->startxval+ TI_HPAD)> ri->line_w ){
						xtb_ti_scroll_left( win, (inspos- ri->line_w- 1)+ ri->maxwidth/2 );
					}
					ti_draw(win, ri, True );
					rtn= XTB_HANDLED;
				}
				  /* 20050115: */
				else if( keysyms[0]== XK_Insert || (keysyms[0]=='v' && CheckMask(xtb_modifier_state,ControlMask)) ){
					if( !ri_CopyClipboard(ri, win, textcopy, &rtn) ){
						Boing(5);
					}
					if( !rtn ){
						rtn= XTB_HANDLED;
					}
				}
				else if( (keysyms[0]=='c' && CheckMask(xtb_modifier_state,ControlMask)) ){
					if( *(ri->text) ){
						XStoreBuffer( t_disp, ri->text, strlen(ri->text), 0);
					}
					if( !rtn ){
						rtn= XTB_HANDLED;
					}
				}
				else if( (rtn = (*ri->func)(win, (int) keysyms[0],
						   textcopy, ri->val)) == XTB_STOP
				){
					break;
				}
			}
			for (i = 0;   i < nbytes;   i++) {
				strcpy(textcopy, ri->text);
				keysyms[i]= 0x0000FFFF & XLookupKeysym( (XKeyPressedEvent*) &evt->xkey, 0);
				if( !(keysyms[i]!= XK_Shift_L && keysyms[i]!= XK_Shift_R &&
						keysyms[i]!= XK_Control_L && keysyms[i]!= XK_Control_R &&
						keysyms[0]!= XK_Caps_Lock && keysyms[0]!= XK_Mode_switch
					)
				){
					keysyms[i]= 0;
				}
				if( keys[i] || keysyms[i] ){
					if( debugFlag && debugLevel== -2 ){
						fprintf( StdErr, "xtb_ti_ins(\"%s\"(%s),%s=0x%x[%d])\n", textcopy,
							CharToString(keys[i]), XKeysymToString(keysyms[0]),
							(keys[i])? keys[i] : keysyms[0], i
						);
						fflush( StdErr );
					}
					if( keysyms[0]== XK_Right ){
						ti_move_right( win, ri );
						ti_draw(win, ri, True );
						rtn= XTB_HANDLED;
					}
					else if( keysyms[0]== XK_Left ){
						ti_move_left( win, ri );
						ti_draw(win, ri, True );
						rtn= XTB_HANDLED;
					}
					else if( keysyms[0]== XK_Home ){
					  int inspos;
						ri->curidx= 0;
						ri->curxval= text_width(norm_font, ri->text, ri->curidx, ri);
						if( (inspos= ri->curxval- ri->startxval+ TI_HPAD)< 0 ){
							xtb_ti_scroll_right( win, (0- inspos- 1)+ ri->maxwidth/2 );
						}
						ti_draw(win, ri, True );
						rtn= XTB_HANDLED;
					}
					else if( keysyms[0]== XK_End ){
					  int inspos;
						ri->curidx= ri->curlen;
						ri->curxval= text_width(norm_font, ri->text, ri->curidx, ri);
						if( (inspos= ri->curxval- ri->startxval+ TI_HPAD)> ri->line_w ){
							xtb_ti_scroll_left( win, (inspos- ri->line_w- 1)+ ri->maxwidth/2 );
						}
						ti_draw(win, ri, True );
						rtn= XTB_HANDLED;
					}
					  /* 20050115: */
					else if( keysyms[0]== XK_Insert || (keysyms[0]=='v' && CheckMask(xtb_modifier_state,ControlMask)) ){
						if( !ri_CopyClipboard(ri, win, textcopy, &rtn) ){
							Boing(5);
						}
						if( !rtn ){
							rtn= XTB_HANDLED;
						}
					}
					else if( (keysyms[0]=='c' && CheckMask(xtb_modifier_state,ControlMask)) ){
						if( *(ri->text) ){
							XStoreBuffer( t_disp, ri->text, strlen(ri->text), 0);
						}
						if( !rtn ){
							rtn= XTB_HANDLED;
						}
					}
					else if( (rtn = (*ri->func)(win, (int) ((keys[i])? keys[i] : keysyms[i]),
							   textcopy, ri->val)) == XTB_STOP
					){
					 break;
					}
				}
			}
			break;
		}
		case FocusIn:
			if( debugFlag && debugLevel== -2 ){
				fprintf( StdErr, "xtb::ti_h(%d): focus into window %d: _entered=%d enter_flag=%d focus_flag=%d\n",
					__LINE__, win, ri->_entered, ri->enter_flag, ri->focus_flag
				);
				focus_evt(evt);
			}
			if( ri->_entered ){
				ri->enter_flag= 0;
			}
			if (evt->xfocus.detail != NotifyPointer) {
				if( !ri->focus_flag ){
					ti_cursor_off(win, ri);
					ri->clear_flag= 1;
					ri->focus_flag = 1;
					SWAP( ri->norm_pix, ri->back_pix, unsigned long);
					ti_draw(win, ri, ri->clear_flag);
				}
			}
			break;
		case FocusOut:
			if( debugFlag && debugLevel== -2 ){
				fprintf( StdErr, "xtb::ti_h(%d): focus out off window %d: _entered=%d enter_flag=%d focus_flag=%d\n",
					__LINE__, win, ri->_entered, ri->enter_flag, ri->focus_flag
				);
				focus_evt(evt);
			}
			if( ri->_entered ){
				ri->enter_flag= 0;
			}
			if (evt->xfocus.detail != NotifyPointer) {
				if( ri->focus_flag ){
					ti_cursor_off(win, ri);
					ri->clear_flag= 1;
					ri->focus_flag = 0;
					SWAP( ri->norm_pix, ri->back_pix, unsigned long);
					ti_draw(win, ri, ri->clear_flag);
				}
			}
			break;
		case EnterNotify:
			if( debugFlag && debugLevel== -2 ){
			  int dummy;
			  Window w= win;
				XGetInputFocus( t_disp, &w, &dummy );
				fprintf( StdErr, "xtb::ti_h(%d): enter into window %d; focus %d: _entered=%d enter_flag=%d focus_flag=%d\n",
					__LINE__, win, w, ri->_entered, ri->enter_flag, ri->focus_flag
				);
			}
			ri->enter_flag= 1;
			ri->_entered= 0;
			ri->clear_flag= ri->rich_text;
			ti_draw( win, ri, ri->clear_flag );
/*          ti_line(win, ri, ri->_norm_pix);    */
			rtn = XTB_HANDLED;
			break;
		case LeaveNotify:
			if( debugFlag && debugLevel== -2 ){
			  int dummy;
			  Window w= win;
				XGetInputFocus( t_disp, &w, &dummy );
				fprintf( StdErr, "xtb::ti_h(%d): enter out off window %d; focus %d: _entered=%d enter_flag=%d focus_flag=%d\n",
					__LINE__, win, w, ri->_entered, ri->enter_flag, ri->focus_flag
				);
			}
			ri->enter_flag= 0;
			ri->_entered= 0;
			ri->clear_flag= ri->rich_text;
			ti_draw( win, ri, ri->clear_flag );
/*          ti_line(win, ri, ri->_back_pix);    */
			rtn = XTB_HANDLED;
			break;
		case ButtonPress:
		/* If Button2, insert cutbuffer 0,
         \ if Button3, copy contents into cutbuffer 0
         \ else wait for release and set input focus
         */
		{   Window focus;
			int revert;
			XGetInputFocus( t_disp, &focus, &revert);
			if( debugFlag && debugLevel== -2 ){
				fprintf( StdErr, "xtb::ti_h(%d): buttonpress into window %d; focus %d/%d: _entered=%d enter_flag=%d focus_flag=%d\n",
					__LINE__, win, focus, revert, ri->_entered, ri->enter_flag, ri->focus_flag
				);
			}
			if( focus!= win && focus!= PointerRoot && focus!= ri->frame->parent ){
			  /* Send the event to the window that *has* the focus.  */
				evt->xany.window= focus;
				ri->button_handled= False;
				ri->button_pressed= False;
				XSendEvent( t_disp, focus, 0,
					VisibilityChangeMask|StructureNotifyMask|ExposureMask|KeyPressMask|EnterWindowMask|LeaveWindowMask|
					FocusChangeMask|ButtonPressMask|PointerMotionMask|PointerMotionHintMask|ButtonReleaseMask,
					evt
				);
			}
			else{
				ri->button_pressed= True;
				switch( bev->button){
					case Button3:
						if( *(ri->text) ){
							XStoreBuffer( t_disp, ri->text, strlen(ri->text), 0);
						}
						ri->button_handled= True;
						break;
					case Button2:
						if( !ri_CopyClipboard(ri, win, textcopy,&rtn) ){
							Boing(5);
						}
						break;
						ri->button_handled= True;
					default:
						ri->button_handled= False;
						break;
				}
			}
			break;
		}
		case ButtonRelease:{
		  Window focus;
		  int revert;
			XGetInputFocus( t_disp, &focus, &revert);
		/* Set input focus */
			if( debugFlag && debugLevel== -2 ){
				fprintf( StdErr, "xtb::ti_h(%d): buttonrelease into window %d; focus %d/%d: _entered=%d enter_flag=%d focus_flag=%d\n",
					__LINE__, win, focus, revert, ri->_entered, ri->enter_flag, ri->focus_flag
				);
			}
			if( ri->button_pressed ){
				ri->button_pressed= False;
				if( !ri->button_handled ){
					XSetInputFocus(t_disp, win, RevertToParent, CurrentTime);
				}
			}
			break;
		}
		case MotionNotify:
			if( ri->button_pressed ){
			 Window root_win, win_win;
			 int root_x, root_y, win_x, win_y, mask;
				XQueryPointer( t_disp, win, &root_win, &win_win,
					&root_x, &root_y, &win_x, &win_y, &mask
				);
				mask= xtb_Mod2toMod1(mask);
				if( CheckMask(mask, ShiftMask) ){
					if( win_x > ri->button_xpos ){
						xtb_ti_scroll_right( win, 1 );
					}
					else if( win_x< ri->button_xpos ){
						xtb_ti_scroll_left( win, 1 );
					}
					ri->button_xpos= win_x;
					ri->button_handled= True;
				}
			}
			break;
		default:
			rtn = XTB_NOTDEF;
			break;
	}
	GCA();
	return rtn;
}

void xtb_ti_scroll_left( Window win, int places )
{ struct ti_info *info = (struct ti_info *) xtb_lookup(win);
   int i, maxlen;
	if( info && info->frame && info->type== xtb_TI2 ){
		info= ((ti2_info*)info->frame->info)->ti_info;
		win= info->frame->win;
	}
	maxlen= strlen(info->text)-1;
	for( i= 0; i< places && info && info->startidx< maxlen; i++ ){
		info->startxval+= text_width(info->font, &(info->text[ (info->startidx)++ ]), 1, info);
	}
	if( i ){
		ti_cursor_off( win, info);
		ti_draw( win, info, 1 );
	}
	else{
		Boing(5);
	}
}

void xtb_ti_scroll_right( Window win, int places )
{ struct ti_info *info = (struct ti_info *) xtb_lookup(win);
   int i;
	if( info && info->frame && info->type== xtb_TI2 ){
		info= ((ti2_info*)info->frame->info)->ti_info;
		win= info->frame->win;
	}
	for( i= 0; i< places && info && info->startidx> 0; i++ ){
		info->startxval-= text_width(info->font, &(info->text[ --(info->startidx) ]), 1, info);
	}
	if( i ){
		ti_cursor_off( win, info);
		ti_draw( win, info, 1 );
	}
	else{
		Boing(5);
	}
	if( !info->startidx || !info->curidx ){
		if( info->startxval ){
			info->startxval= 0;
			ti_draw( win, info, 1);
		}
		info->startidx= 0;
	}
}

void _xtb_ti_new( Window win, char *text, int maxwidth, int maxchar,
	FNPTR( func, xtb_hret, (Window, int, char *, xtb_data) ),
	xtb_data val, xtb_frame *frame,
	int ti_hpad, int ti_vpad, int ti_lpad, int ti_brdr,
	char *caller
)
/*
 * This routine creates a new editable text widget under `win'
 * with the initial text `text'.  The widget contains only
 * one line of text which cannot exceed `maxchar' characters.
 * The size of the widget is returned in `frame'.  Each
 * time a key is pressed in the window,  `func' will be called
 * with the window, the character, a copy of the text, and `val'.
 * The state of the widget can be changed by the routines below.
 * May set window to zero if the maximum overall character width
 * (MAXCHBUF) is exceeded.
 \ 931109 RJB: maxchar is truncated to MAXCHBUF-1
 \ 970124 RJB: buffer is allocated dynamically.
 */
{
	struct ti_info *info;
	Cursor curs= ti_cursor;
	char name[128];

#ifdef STATIC_TI_BUFLEN
	if (maxchar >= MAXCHBUF) {
/*
        frame->win = (Window) 0;
        return;
 */
		maxchar= MAXCHBUF-1;
		if( strlen(text) >= MAXCHBUF ){
			text[MAXCHBUF-1]= '\0';
		}
	}
#endif
	if( !(info = (struct ti_info *) calloc(1, sizeof(struct ti_info))) ){
		return;
	}
	info->type= xtb_TI;
	if( !(info->text= calloc( maxchar, sizeof(char))) ){
		fprintf( StdErr, "%s(): can't (re)allocate textbuf of %d\n", caller, maxchar );
		xfree( info );
		return;
	}
	info->maxlen= maxchar;

	if( !maxwidth ){
		if( strlen(text) ){
			frame->width = XTextWidth(norm_font, text, maxchar) + 4*ti_hpad;
		}
		else{
/*          frame->width = XTextWidth(norm_font, "8", 1) * maxchar + 2*ti_hpad; */
			frame->width = XFontWidth(norm_font) * maxchar + 2*ti_hpad;
		}
		maxwidth= frame->width;
	}
	else{
		frame->width= XFontWidth( norm_font)* maxwidth;
	}
	if( frame->width> 0.75* DisplayWidth(t_disp, t_screen) ){
		frame->width= 0.75* DisplayWidth(t_disp, t_screen);
	}
	frame->height = max_ascent + max_descent + ti_vpad + ti_lpad;
	frame->border= ti_brdr;
	frame->x_loc = frame->y_loc = 0;
	frame->description= NULL;
	frame->win = xtb_XCreateSimpleWindow(t_disp, win, 0, 0,
					frame->width, frame->height, ti_brdr,
					xtb_norm_pix, xtb_back_pix);
	XSetWindowColormap( t_disp, frame->win, cmap );
	XSelectInput(t_disp, frame->win,
		VisibilityChangeMask|StructureNotifyMask|ExposureMask|KeyPressMask|EnterWindowMask|LeaveWindowMask|
		FocusChangeMask|ButtonPressMask|PointerMotionMask|PointerMotionHintMask|ButtonReleaseMask
	);
	if( !curs ){
		ti_cursor= curs = XCreateFontCursor(t_disp, XC_pencil );
	}
	fg_color.pixel = xtb_norm_pix;
	XQueryColor(t_disp, cmap, &fg_color);
	bg_color.pixel = xtb_back_pix;
	XQueryColor(t_disp, cmap, &bg_color);
	XRecolorCursor(t_disp, curs, &fg_color, &bg_color);
	XDefineCursor(t_disp, frame->win, curs);
	sprintf( name, "xtb_ti frame #%d", xtb_ti_serial++ );
	XStoreName( t_disp, frame->win, name );

	frame->parent= win;
	frame->frames= 1;
	if( !(frame->framelist= (xtb_frame**) malloc(sizeof(xtb_frame*))) ){
		return;
	}
	frame->framelist[0]= frame;

	info->func = func;
	info->val = val;
	info->_norm_pix= info->norm_pix = xtb_norm_pix;
	info->light_pix= xtb_light_pix;
	info->middle_pix= xtb_middle_pix;
	info->_back_pix= info->back_pix = xtb_back_pix;
	info->maxwidth= maxwidth;
	info->font = norm_font;
	info->alt_font= greek_font;
	if( greek_font ){
		info->max_ascent= MAX( norm_font->ascent, greek_font->ascent );
	}
	else{
		info->max_ascent = norm_font->ascent;
	}
	info->cursor_height= norm_font->ascent+ norm_font->descent- 2;
	if (text)
		(void) strcpy(info->text, text);
	else
		info->text[0] = '\0';
	info->curlen= info->curidx = strlen(info->text);
	info->curxval = text_width(norm_font, info->text, info->curidx, info);
	info->startidx= info->startxval= 0;
	info->line_y = frame->height - 2;
	info->line_w = frame->width - 2 * ti_hpad;
	info->focus_flag = 0;
	info->frame= frame;
	info->frame->enabled= True;
	frame->info= (xtb_registry_info*) info;
	xtb_register(frame, frame->win, ti_h, (xtb_registry_info*) info);
	XMapWindow(t_disp, frame->win);
	XNextEvent( t_disp, &evt);
	xtb_dispatch( t_disp, frame->win, 0, NULL, &evt);
	frame->width += (2 * ti_brdr);
	frame->height += (2 * ti_brdr);
/*  frame->mapped= 1;   */
	frame->redraw= xtb_ti_redraw;
	frame->destroy= xtb_ti_del;
	frame->chfnt= xtb_ti_chfnt;
}

void xtb_ti_new( Window win, char *text, int maxwidth, int maxchar,
	FNPTR( func, xtb_hret, (Window, int, char *, xtb_data) ),
	xtb_data val, xtb_frame *frame
)
{
	_xtb_ti_new( win, text, maxwidth, maxchar, func, val, frame,
		TI_HPAD, TI_VPAD, TI_LPAD, TI_BRDR, "xtb_ti_new"
	);
}

static xtb_hret ti2_h(XEvent *evt, xtb_data info, Window parent)
{
/*     Window win = evt->xany.window;   */
	struct ti2_info *ri = (struct ti2_info *) info;
	xtb_hret rtn= XTB_NOTDEF;
	xtb_data data;
	int i;

	switch (evt->type) {
		case UnmapNotify:
			if( ri->frame ) ri->frame->mapped= 0;
			break;
		case MapNotify:
			if( ri->frame ) ri->frame->mapped= 1;
			break;
		case VisibilityNotify:
		case ConfigureNotify:
			if( ri->frame ) ri->frame->mapped= 1;
			break;
	}
	 /* Pass on some events to all buttons.
       \ This is maybe redundant. But then maybe not..
       */
	for( i= 0; i< 2; i++ ){
		switch (evt->type) {
			case UnmapNotify:
			case MapNotify:
			case VisibilityNotify:
			case ConfigureNotify:
			case Expose:
				evt->xany.window= ri->subwin[i];
				if( (data= xtb_lookup(evt->xany.window)) ){
					  /* Could be more elegant here... the callback handler is registered! */
					switch( i ){
						case 0:
							ti_h( evt, data, ri->main_win );
							break;
						case 1:
							bt_h( evt, data, ri->main_win );
							break;
					}
				}
				break;
		}
	}
	return( rtn );
}

void xtb_ti2_redraw( Window win)
{   struct ti2_info *ri= (struct ti2_info*) xtb_lookup(win);
    int i;
    xtb_data data;

	for( i= 0; i< 2; i++ ){
		if( (data= xtb_lookup(ri->subwin[i])) ){
			switch( i ){
				case 0:
					ti_draw( ri->subwin[i], ri->ti_info, ri->ti_info->clear_flag );
					break;
				case 1:
					bt_draw( ri->subwin[i], data );
					break;
			}
		}
	}
	return;
}

void xtb_ti2_del(Window win)
{
	struct ti2_info *info;
	int i;

	if (xtb_unregister(win, (xtb_registry_info* *) &info)) {
		if( info ){
			for( i = 0; i < 2; i++ ){
				switch( i ){
					case 0:
						xtb_ti_del(info->subwin[i], NULL );
						break;
					case 1:
						xtb_bt_del(info->subwin[i], NULL );
						break;
				}
			}
			if( info->frame ){
				xfree( info->frame->framelist );
			}
			xfree( info);
		}
		XDestroyWindow(t_disp, win);
	}
}

int xtb_ti2_chfnt( Window win, XFontStruct *norm_font, XFontStruct *greek_font )
{ struct ti2_info *info = (struct ti2_info *) xtb_lookup(win);
	if( info && norm_font && greek_font ){
	  int i;
		for( i= 0; i< 2; i++ ){
			switch( i ){
				case 0:
					xtb_ti_chfnt( info->subwin[i], norm_font, greek_font );
					break;
				case 1:
					xtb_bt_chfnt( info->subwin[i], norm_font, greek_font );
					break;
			}
		}
		return(0);
	}
	return(1);
}

/* 20020429: xtb_TI2: an additional class of text input fields. These widgets are composite; they
 \ consist of a standard xtb_TI field, and an xtb_BT button. This button is supposed to trigger
 \ an action relevant to the text that is to be input through the TI field, e.g. to open a popup
 \ menu to select from different values, a history, etc.
 \ This class of widgets can largely be used as the standard xtb_TI widgets; most xtb_ti_... routines
 \ will call the appropriate handler when called with an xtb_TI2 widget. Likely, xtb_describe() will
 \ add the description of an xtb_TI2 widget to both the parent (composite) frame, and the underlying
 \ xtb_TI frame. The xtb_ti2_ti_frame() routine can be used to get at the TI frame from within user
 \ code. This routine should not fail when called with a frame->win for which frame->info->type==xtb_TI2.
 */
void xtb_ti2_new( Window win, char *text, int maxwidth, int maxchar,
	FNPTR( tifunc, xtb_hret, (Window, int, char *, xtb_data) ),
	char *bttext,
	FNPTR( btfunc, xtb_hret, (Window, int, xtb_data) ),
	xtb_data val, xtb_frame *frame
)
{
	struct ti2_info *info;
	xtb_frame *sub_frame;
	int i, x, y;
	char name[128];

	if( !(info = (struct ti2_info *) calloc( 1, sizeof(struct ti2_info))) ){
		return;
	}
	if( !(sub_frame= (xtb_frame*) malloc(sizeof(xtb_frame))) ){
		return;
	}
	if( !(frame->framelist= (xtb_frame**) malloc(2*sizeof(xtb_frame*))) ){
		return;
	}
	if( SetParentBackground )
		XSetWindowBackground( t_disp, win, xtb_light_pix );
	frame->description= NULL;
	frame->frames= 2;
	frame->parent= win;
	frame->width = frame->height = 0;
	frame->x_loc = frame->y_loc = 0;
	frame->win = xtb_XCreateSimpleWindow(t_disp, win, 0, 0, 1, 1,
					BT_BRDR, xtb_norm_pix, xtb_middle_pix);
	XSetWindowColormap( t_disp, frame->win, cmap );
	sprintf( name, "xtb_ti2 frame #%d", xtb_ti2_serial++ );
	XStoreName( t_disp, frame->win, name );
	info->type= xtb_TI2;
	info->main_win = frame->win;
	info->val = val;
	info->norm_pix = xtb_norm_pix;
	info->light_pix= xtb_light_pix;
	info->middle_pix= xtb_middle_pix;
	info->back_pix = xtb_back_pix;
	frame->info= (xtb_registry_info*) info;
	info->frame= frame;
	info->frame->enabled= True;

	if( !br_cursor ){
		br_cursor = XCreateFontCursor(t_disp, XC_hand2 );
	}
	fg_color.pixel = xtb_norm_pix;
	XQueryColor(t_disp, cmap, &fg_color);
	bg_color.pixel = xtb_back_pix;
	XQueryColor(t_disp, cmap, &bg_color);
	XRecolorCursor(t_disp, br_cursor, &fg_color, &bg_color);
	XDefineCursor(t_disp, frame->win, br_cursor);

	XSelectInput(t_disp, frame->win, ExposureMask|ButtonPressMask|ButtonReleaseMask|EnterWindowMask|LeaveWindowMask);
	 /* the handler is used simply to get information out */
	xtb_register(frame, frame->win, ti2_h, (xtb_registry_info*) info);
	x= BR_XPAD-1; y= BR_YPAD-1;
	for( i = 0; i< 2; i++ ){
		if( !(sub_frame= (xtb_frame*) malloc(sizeof(xtb_frame))) ){
			return;
		}
		switch( i ){
			case 0:
				_xtb_ti_new( frame->win, text, maxwidth, maxchar, tifunc, val, sub_frame,
					TI_HPAD, TI_VPAD, TI_LPAD, TI_BRDR, "xtb_ti2_new"
				);
				info->ti_info= (ti_info*) sub_frame->info;
				break;
			case 1:
				xtb_bt_new( frame->win, bttext, btfunc, val, sub_frame);
				break;
		}
		info->subwin[i] = sub_frame->win;
		frame->framelist[i]= sub_frame;
		sub_frame->parent= win;
		XMoveWindow(t_disp, sub_frame->win, (sub_frame->x_loc= x), (sub_frame->y_loc= y) );
		x+= (1 + sub_frame->width);
		if( sub_frame->height > frame->height){
			frame->height = sub_frame->height;
		}
	}
	for( i= 0; i< 2; i++ ){
		sub_frame= frame->framelist[i];
		sub_frame->y_loc-= (frame->height- sub_frame->height)/ 2;
		XMoveWindow(t_disp, sub_frame->win, sub_frame->x_loc, sub_frame->y_loc );
	}
	frame->width = x - 1 + BR_XPAD-1;
	frame->height += (2 * (BR_YPAD-1));
	frame->border= BT_BRDR;
	XResizeWindow(t_disp, frame->win, frame->width, frame->height);
	XMapWindow(t_disp, frame->win);
	XSetWindowBackground( t_disp, frame->win, xtb_middle_pix );
	frame->mapped= 1;
	frame->redraw= xtb_ti2_redraw;
	frame->destroy= (void*) xtb_ti2_del;
	frame->chfnt= xtb_ti2_chfnt;
}

xtb_frame *xtb_ti2_ti_frame( Window win)
{   struct ti2_info *ri= (struct ti2_info*) xtb_lookup(win);
	return( (ri)? ri->frame->framelist[0] : NULL );
}

int xtb_ti_length(Window win, xtb_data *val)
{ struct ti_info *info = (struct ti_info *) xtb_lookup(win);

	if( info && info->frame && info->type== xtb_TI2 ){
		info= ((ti2_info*)info->frame->info)->ti_info;
		win= info->frame->win;
	}
#ifndef DEBUG
	if( info )
#endif
	{
		if (val) *val = info->val;
		return( info->curlen );
	}
}

void xtb_ti_get(Window win, char *text, xtb_data *val)
/*
 * This routine returns the information associated with text
 * widget `win'.  The text is filled into the passed buffer
 * `text' which should be MAXCHBUF characters in size.  If
 * `val' is non-zero,  the user supplied info is returned there.
 */
{
	struct ti_info *info = (struct ti_info *) xtb_lookup(win);

	if( info && info->frame && info->type== xtb_TI2 ){
		info= ((ti2_info*)info->frame->info)->ti_info;
		win= info->frame->win;
	}
#ifndef DEBUG
	if( info )
#endif
	{
		if (val) *val = info->val;
		(void) strcpy(text, info->text);
	}
}

int xtb_ti_chfnt( Window win, XFontStruct *font, XFontStruct *alt_font)
{   struct ti_info *ri= (struct ti_info*) xtb_lookup(win);
	if( ri ){
		ri->font= font;
		ri->alt_font= alt_font;
		ti_draw( win, ri, True );
		return(0);
	}
	else{
		return(1);
	}
}

int xtb_ti_set(Window win, char *text, xtb_data val)
/*
 * This routine sets the text of a text widget.  The widget
 * will be redrawn.  Note:  for incremental changes,  ti_ins and
 * ti_dch should be used.  If `val' is non-zero,  it will replace
 * the user information for the widget.  The widget is redrawn.
 * Will return zero if `text' is too long.
 */
{
	struct ti_info *info = (struct ti_info *) xtb_lookup(win);
	int newlen, change= 0;

	if( info && info->frame && info->type== xtb_TI2 ){
		info= ((ti2_info*)info->frame->info)->ti_info;
		win= info->frame->win;
	}
	if( !info ){
		return(-1);
	}
	if (text) {
		if ((newlen = strlen(text)) >= info->maxlen){
		 char *new;
			if( !(new= realloc( info->text, (newlen+2)* sizeof(char))) ){
				fprintf( StdErr, "xtb_ti_set(): can't (re)allocate textbuf of %d\n", newlen );
				return(0);
			}
			else{
				info->text= new;
				info->maxlen= newlen+ 1;
			}
		}
	}
	else{
		newlen = 0;
	}
	if( text ){
		if( strcmp( info->text, text) ){
			change= 1;
			strcpy(info->text, text);
		}
		else if( info->startidx!= 0 ){
		 /* 970423: if we re-set the current string, reset the display offset. */
			change= 1;
		}
	}
	else{
		change= info->text[0];
		info->text[0] = '\0';
	}
	if( !info->font ){
		info->font= norm_font;
	}
	if (val)
		info->val = val;
	if( change ){
		info->curlen= info->curidx = newlen;
		info->curxval = text_width(info->font, info->text, info->curidx, info);
		info->startidx= info->startxval= 0;
		info->rich_text= (xtb_has_greek( info->text) || xtb_has_backslash(info->text) )? 1 : 0;
		ti_draw(win, info, 1);
		return(1);
	}
	return -1;
}

int xtb_ti_ins(Window win, int ch)
/*
 * Inserts the character `ch' onto the end of the text for `win'.
 * Will return zero if there isn't any more room left.  Does
 * all appropriate display updates.
 */
{
	struct ti_info *info = (struct ti_info *) xtb_lookup(win);
	char lstr[1];

	if( info && info->frame && info->type== xtb_TI2 ){
		info= ((ti2_info*)info->frame->info)->ti_info;
		win= info->frame->win;
	}
	if( info->curlen >= info->maxlen-1){
		if( debugFlag ){
			fprintf( StdErr, "Maximal %d characters allowed\n", info->maxlen );
		}
		return 0;
	}

	if( !info->enter_flag ){
		info->enter_flag= info->_entered= 1;
		ti_draw( win, info, 1);
	}

	 /* Text */
	lstr[0] = (char) ch;

	if( info->text[info->curidx]== '\0' ){
		info->text[info->curidx] = ch;
		info->text[info->curidx+1] = '\0';
	}
	else{
	  /* really insert..	*/
	  char *c= &info->text[info->curidx], oc;
		do{
			  /* replace the old char. at this point:	*/
			oc= *c;
			*c++= ch;
			ch= oc;
		} while( *c );
		if( ch ){
			*c++= ch;
		}
		*c++= '\0';
		info->clear_flag= True;
	}
	info->curlen+= 1;
	 /* Turn off cursor */
	ti_cursor_off(win, info);
	if( !info->font ){
		info->font= norm_font;
	}
	if( ch== '\\' ){
		info->rich_text= 1;
	}
	info->curidx += 1;
	if( info->clear_flag ){
		info->curxval += text_width(info->font, lstr, 1, info);
		ti_draw( win, info, 1);
	}
	else{
	  int inspos= info->curxval- info->startxval+ TI_HPAD;
		if( inspos< 0 ){
			info->curxval += text_width(info->font, lstr, 1, info);
			xtb_ti_scroll_right( win, (0- inspos- 1)+ info->maxwidth/2 );
		}
		else if( inspos< info->line_w ){
			t_font= info->font;
			xtb_DrawString(t_disp, win,
					xtb_set_gc(win, info->norm_pix, info->back_pix, info->font->fid),
					info->curxval- info->startxval+TI_HPAD, TI_VPAD+ info->max_ascent,
					lstr, 1, NULL, NULL, False, True
			);
			info->curxval += text_width(info->font, lstr, 1, info);
			ti_cursor_on(win, info);
		}
		else{
/* 981120: ???!!
			info->curidx += 1;
	 */
			info->curxval += text_width(info->font, lstr, 1, info);
			xtb_ti_scroll_left( win, (inspos- info->line_w- 1)+ info->maxwidth/2 );
		}
	}
	return 1;
}

int xtb_ti_delchar(Window win, struct ti_info *info)
/*
 * Deletes (backspaces) the character at the end of the text for `win'.  Will
 * return zero if there aren't any characters to delete.  Does
 * all appropriate display updates.
 */
{
	int chw;

	if( info->curidx == 0)
		return 0;
	ti_draw( win, info, 1);
	 /* Wipe out cursor */
	ti_cursor_off(win, info);
	info->curidx -= 1;
	if( !info->font ){
		info->font= norm_font;
	}
	chw = text_width(info->font, &(info->text[info->curidx]), 1, info);
	info->curxval -= chw;
	if( info->text[info->curidx+1]== '\0' ){
	  /* at end of text	*/
		info->text[info->curidx] = '\0';
	}
	else{
	  char *c= &info->text[info->curidx], *d= &c[1];
		while( *c ){
			*c++= *d++;
		}
		info->clear_flag= True;
	}
	info->curlen-= 1;
	/* Wipe out character */
/*  if( info->curxval< info->line_w + TI_HPAD && !info->startxval ){    */
	if( info->curxval- info->startxval+ TI_HPAD> 0 ){
		XClearArea(t_disp, win, info->curxval- info->startxval+TI_HPAD, TI_VPAD,
			   (unsigned int) chw + 1,
			   (unsigned int) info->font->ascent + info->font->descent,
			   False
		);
	}
	else{
		if( info->startxval> 0 ){
			xtb_ti_scroll_right( win, info->maxwidth );
		}
	}
	if( !info->startidx || !info->curidx ){
		if( info->startxval ){
			info->startxval= 0;
			ti_draw( win, info, 1);
		}
		info->startidx= 0;
	}
	if( info->clear_flag ){
		ti_draw( win, info, 1);
	}
	else{
		ti_cursor_on(win, info);
	}
	return 1;
}

int xtb_ti_dch(Window win)
/*
 * Deletes (backspaces) the character at the end of the text for `win'.  Will
 * return zero if there aren't any characters to delete.  Does
 * all appropriate display updates.
 */
{
	struct ti_info *info = (struct ti_info *) xtb_lookup(win);
	if( info && info->frame && info->type== xtb_TI2 ){
		info= ((ti2_info*)info->frame->info)->ti_info;
		win= info->frame->win;
	}
	return( xtb_ti_delchar( win, info ) );
}

int xtb_ti_dch_right( Window win)
{ struct ti_info *info = (struct ti_info *) xtb_lookup(win);
	if( info && info->frame && info->type== xtb_TI2 ){
		info= ((ti2_info*)info->frame->info)->ti_info;
		win= info->frame->win;
	}
	ti_move_right( win, info );
	return( xtb_ti_delchar( win, info ) );
}

void xtb_ti_redraw( Window win)
{   struct ti_info *ri= (struct ti_info*) xtb_lookup(win);
	if( ri && ri->frame && ri->type== xtb_TI2 ){
		xtb_ti2_redraw( ri->frame->win );
	}
	else{
		ti_draw( win, ri, True);
	}
	return;
}

void xtb_ti_del(Window win, xtb_data *info)
/*
 * Deletes an input text widget.  User defined data is returned in `info'.
 */
{
	struct ti_info *ti= (ti_info*) xtb_lookup(win);

	if( ti && ti->frame && ti->type== xtb_TI2 ){
		ti= ((ti2_info*)ti->frame->info)->ti_info;
		win= ti->frame->win;
	}
	if (xtb_unregister(win, (xtb_registry_info* *) &ti)) {
		if( ti ){
			if( info ){
				*info = ti->val;
			}
			if( ti->text ){
				xfree( ti->text );
			}
			if( ti->frame ){
				xfree( ti->frame->framelist );
			}
			xfree( ti);
		}
		else if( info ){
			*info= NULL;
		}
		XDestroyWindow(t_disp, win);
	}
}

/*
 * Simple colored output frame - usually used for drawing lines
 */

void xtb_bk_new(win, width, height, frame)
Window win;   		/* Parent window */
unsigned width, height;   	/* Size          */
xtb_frame *frame;		/* Returned size */
/*
 * This routine creates a new frame that displays a block
 * of color whose size is given by `width' and `height'.
 * It is usually used to draw lines.  No user interaction
 * is defined for the frame.  The color used is the default
 * foreground color set in xtb_init().
 \ 20010722: block windows are always sent to the bottom of the
 \ window stack, to ensure they don't obscure other windows.
 */
{ Cursor curs= bk_cursor;
   char name[128];
   xtb_registry_info *info= NULL;

	frame->x_loc = frame->y_loc = 0;
	frame->width = width;
	frame->height = height;
	frame->border= 0;
	frame->win = xtb_XCreateSimpleWindow(t_disp, win,
					frame->x_loc, frame->y_loc,
					frame->width, frame->height,
					0, xtb_norm_pix, xtb_norm_pix);
	XSetWindowColormap( t_disp, frame->win, cmap );
	sprintf( name, "xtb_bk frame #%d", xtb_bk_serial++ );
	XStoreName( t_disp, frame->win, name );
	frame->description= NULL;
	frame->parent= win;
	frame->frames= 1;
	if( !(frame->framelist= (xtb_frame**) malloc(sizeof(xtb_frame*))) ){
		return;
	}
	frame->framelist[0]= frame;
	if( (info= (xtb_registry_info*) calloc(1, sizeof(xtb_registry_info))) ){
		info->val= 0;
		info->func= NULL;
		info->frame= frame;
		info->type= xtb_BK;
		frame->info= info;
		xtb_register(frame, frame->win, NULL, (xtb_registry_info*) info);
	}
	else{
		frame->info= NULL;
	}
	if( !curs ){
		bk_cursor= curs = XCreateFontCursor(t_disp, XC_boat );
	}
	fg_color.pixel = xtb_norm_pix;
	XQueryColor(t_disp, cmap, &fg_color);
	bg_color.pixel = xtb_back_pix;
	XQueryColor(t_disp, cmap, &bg_color);
	XRecolorCursor(t_disp, curs, &fg_color, &bg_color);
	XDefineCursor(t_disp, frame->win, curs);

	XMapWindow(t_disp, frame->win);
	XLowerWindow( t_disp, frame->win );
	XSelectInput( t_disp, frame->win, ButtonReleaseMask|ButtonPressMask);
	frame->mapped= 1;
	frame->redraw= NULL;
	frame->destroy= (void*)xtb_bk_del;
	frame->chfnt= NULL;
}


void xtb_bk_del(Window win)
/*
 * Deletes a block frame.
 */
{  xtb_registry_info *info;
	if( xtb_unregister( win, &info ) ){
		if( info->frame ){
			xfree( info->frame->framelist );
		}
		xfree( info );
		XDestroyWindow(t_disp, win);
	}
}


#define SLIDERULER
#ifdef SLIDERULER

/* A sliderule widget
 \ (C) RJB 1996
 */

#define SR_HPAD 2
#define SR_VPAD 2
#define SR_LPAD 3
#define SR_BRDR 2
#define SR_SBRDR 1

static void sr_draw_slide( Window win, sr_info *ri)
{ int len;
	if( ri->redraw ){
	 void sr_draw();
		sr_draw( win, ri, ri->clear_flag );
		return;
	}
	len= xtb_TextWidth( ri->text, ri->font, ri->alt_font );
	if( len>= ri->slide_w ){
		ri->slide_w= len+ 2* SR_HPAD;
		XResizeWindow( t_disp, ri->slide, ri->slide_w+ 1, ri->slide_h );
	}
	XSetWindowBackground( t_disp, ri->slide, ri->norm_pix );
	XClearWindow( t_disp, ri->slide );
	XClearWindow( t_disp, ri->slide_ );
	xtb_DrawString(t_disp, ri->slide,
			xtb_set_gc( ri->slide, ri->back_pix, ri->norm_pix, ri->font->fid),
/* 			1+ (ri->slide_w- len)/2, SR_VPAD+ 1,	*/
			1+ (ri->slide_w- len)/2, SR_VPAD+ 1- ri->font->descent,
			ri->text, strlen(ri->text), ri->font, ri->alt_font, True, False
	);
	if( ri->vert_flag ){
	 int pos= ri->frame->height- 2* SR_BRDR- (ri->position+ ri->minposition);
		XMoveWindow( t_disp, ri->slide_, ri->slide_x+ 1,
			pos+ ri->slide_h/2+ 1
		);
		XMoveWindow( t_disp, ri->slide, ri->slide_x,
			pos- ri->slide_h/2- 1
		);
	}
	else{
	 int pos= ri->position+ ri->minposition;
		XMoveWindow( t_disp, ri->slide_, pos+ ri->slide_w/2+ 1, ri->slide_y+ 3 );
		XMoveWindow( t_disp, ri->slide, pos- ri->slide_w/2- 1, ri->slide_y );
	}
}

static void sr_draw( Window win, sr_info *ri, int c_flag)
{   GC gc;
    char *buf;
    int len, shift= (ri->font->ascent+ ri->font->descent)/ 2;
    xtb_frame *frame= ri->frame;
    XPoint line[3];
    int height= frame->height- 2* SR_BRDR,
		width= frame->width- 2* SR_BRDR;

	if( !frame || !frame->mapped ){
		return;
	}
	if( ri->frame->enabled ){
		XSetWindowBackgroundPixmap( t_disp, win, None );
		XSetWindowBackground( t_disp, win, ri->light_pix);
	}
	else{
		XSetWindowBackground( t_disp, win, ri->light_pix);
		XSetWindowBackgroundPixmap( t_disp, win, disabled_bg_light );
	}
	if( c_flag){
		XClearWindow(t_disp, win);
		if( ri->clear_flag ){
			ri->clear_flag= 0;
		}
	}
	if( ri->vert_flag ){
	 int less_y= frame->height- ri->lr_h- 2* SR_BRDR- SR_SBRDR;
		 /* lower  */
		line[0].x= 1;
		line[0].y= (short) less_y- 1;
		line[1].x= (short) width- 1;
		line[1].y= (short) less_y- 1;
		line[2].x= (short) width- 1;
		line[2].y= ri->lr_h+ SR_BRDR+ 1;
		gc= xtb_set_gc(win, ri->back_pix, ri->light_pix, ri->font->fid),
		XSetLineAttributes( t_disp, gc, 2, LineSolid, CapButt, JoinMiter);
		XDrawLines(t_disp, win, gc, line, 3, CoordModeOrigin);

		 /* upper  */
		line[0].x= line[1].x= 1;
		line[1].y= line[2].y= ri->lr_h+ SR_BRDR+ 1;
		line[0].y= (short) less_y- 1;
		line[2].x= (short) width- 1;
		gc= xtb_set_gc(win, ri->norm_pix, ri->light_pix, ri->font->fid),
		XSetLineAttributes( t_disp, gc, 2, LineSolid, CapButt, JoinMiter);
		XDrawLines(t_disp, win, gc, line, 3, CoordModeOrigin);

		XDrawLine(t_disp, win,
			 (gc= xtb_set_gc(win, ri->norm_pix, ri->light_pix, ri->font->fid)),
			 ri->line_x, ri->minposition, ri->line_x, ri->maxposition
		);

		buf= d2str( ri->minvalue, "%g", NULL );
		len= xtb_TextWidth( buf, ri->font, ri->alt_font);
		xtb_DrawString(t_disp, win,
				xtb_set_gc( win, ri->norm_pix, ri->light_pix, ri->font->fid),
				ri->line_x- len/2, ri->maxposition- shift,
				buf, strlen(buf), ri->font, ri->alt_font, True, False
		);
		buf= d2str( ri->maxvalue, "%g", NULL );
		len= xtb_TextWidth( buf, ri->font, ri->alt_font);
		xtb_DrawString(t_disp, win,
				xtb_set_gc( win, ri->norm_pix, ri->light_pix, ri->font->fid),
				ri->line_x- len/2, ri->minposition- shift+ 1,
				buf, strlen(buf), ri->font, ri->alt_font, True, False
		);

		XDrawLine( t_disp, ri->more, gc, 0, ri->lr_h, ri->lr_w/2, 0 );
		XDrawLine( t_disp, ri->more, gc, ri->lr_w/2, 0, ri->lr_w, ri->lr_h );
		XDrawLine( t_disp, ri->less, gc, 0, 0, ri->lr_w/2, ri->lr_h );
		XDrawLine( t_disp, ri->less, gc, ri->lr_w/2, ri->lr_h, ri->lr_w, 0 );
		XMoveWindow( t_disp, ri->more, 0, 0 );
		XMoveWindow( t_disp, ri->less, 0, less_y );
	}
	else{
	 int more_x= frame->width- ri->lr_w- 2* SR_BRDR- SR_SBRDR;
		 /* lower  */
		line[0].x= ri->lr_w+ SR_BRDR+ 1;
		line[0].y= (short) height- 1;
		line[1].x= (short) more_x- 1;
		line[1].y= (short) height- 1;
		line[2].x= (short) more_x- 1;
		line[2].y= 0;
		gc= xtb_set_gc(win, ri->back_pix, ri->light_pix, ri->font->fid),
		XSetLineAttributes( t_disp, gc, 2, LineSolid, CapButt, JoinMiter);
		XDrawLines(t_disp, win, gc, line, 3, CoordModeOrigin);

		 /* upper  */
		line[0].x= line[1].x= ri->lr_w+ SR_BRDR+ 1;
		line[1].y= line[2].y= 1;
		line[0].y= (short) height- 1;
		line[2].x= (short) more_x- 1;
		gc= xtb_set_gc(win, ri->norm_pix, ri->light_pix, ri->font->fid),
		XSetLineAttributes( t_disp, gc, 2, LineSolid, CapButt, JoinMiter);
		XDrawLines(t_disp, win, gc, line, 3, CoordModeOrigin);

		XDrawLine(t_disp, win,
			 (gc= xtb_set_gc(win, ri->norm_pix, ri->light_pix, ri->font->fid)),
			 ri->minposition, ri->line_y, ri->maxposition, ri->line_y
		);

		buf= d2str( ri->minvalue, "%g", NULL );
		len= xtb_TextWidth( buf, ri->font, ri->alt_font);
		xtb_DrawString(t_disp, win,
				xtb_set_gc( win, ri->norm_pix, ri->light_pix, ri->font->fid),
/* 				ri->minposition- len/2, ri->line_y- shift,	*/
				ri->minposition- len/2, ri->line_y- shift- ri->font->descent+ 1,
				buf, strlen(buf), ri->font, ri->alt_font, False, False
		);
		buf= d2str( ri->maxvalue, "%g", NULL );
		len= xtb_TextWidth( buf, ri->font, ri->alt_font);
		xtb_DrawString(t_disp, win,
				xtb_set_gc( win, ri->norm_pix, ri->light_pix, ri->font->fid),
/* 				ri->maxposition- len/2, ri->line_y- shift,	*/
				ri->maxposition- len/2, ri->line_y- shift- ri->font->descent+ 1,
				buf, strlen(buf), ri->font, ri->alt_font, False, False
		);

		XDrawLine( t_disp, ri->less, gc, ri->lr_w, ri->lr_h, 0, ri->lr_h/2);
		XDrawLine( t_disp, ri->less, gc, 0, ri->lr_h/2, ri->lr_w, 0);
		XDrawLine( t_disp, ri->more, gc, 0, ri->lr_h, ri->lr_w, ri->lr_h/2);
		XDrawLine( t_disp, ri->more, gc, ri->lr_w, ri->lr_h/2, 0, 0);
		XMoveWindow( t_disp, ri->less, 0, 0);
		XMoveWindow( t_disp, ri->more, more_x, 0);
	}
	ri->redraw= 0;
	sr_draw_slide( win, ri);
}

/* Lowlevel function that sets a new value and updates position information */
static void xtb_sr_set_( sr_info *info, double value )
{
	if( !info ){
		return;
	}
	if( NaN(value) ){
		info->position= 0;
		info->value= value;
	}
	else{
		info->value= (info->int_flag)? floor( value /* + 0.5*/ ) : value;
		if( value<= info->_minvalue ){
			info->position= (info->minvalue< info->maxvalue)? 0 : info->maxposition- info->minposition;
		}
		else if( value>= info->_maxvalue ){
			info->position= (info->minvalue< info->maxvalue)? info->maxposition- info->minposition : 0;
		}
		else{
			info->position = (int) ( (info->delta)? (value- info->minvalue)/ info->delta/* + 0.5*/ : info->minposition );
		}
		CLIP( info->position, 0, info->maxposition- info->minposition);
	}
	d2str( value, "%g", info->text);
}


char *sr_win( sr_info *ri, Window evwin )
{ char *w;
	if( evwin== ri->frame->win ){
		w= "main";
	}
	else if( evwin== ri->slide_){
		w= "slide_";
	}
	else if( evwin== ri->slide){
		w= "slide";
	}
	else if( evwin== ri->less){
		w= "decrease";
	}
	else if( evwin== ri->more ){
		w= "increase";
	}
	else if( evwin== ri->frame->parent ){
		w= "slide parent";
	}
	else{
		w= "unknown";
	}
	return( w );
}

#define maskcat(dest,source)	if(*dest && dest[strlen(dest)-1] != '='){strcat( dest, "|" source);}else{strcat( dest, source);};

char *xtb_modifiers_string( int mask )
{   static char buf[]= "0xffffffff Button1Mask|Button2Mask|Button3Mask|Button4Mask|Button5Mask|ShiftMask|ControlMask|LockMask|Mod1Mask|Mod2Mask|Mod3Mask|Mod4Mask|Mod5Mask|||";
	sprintf( buf, "0x%x=", mask );
	if( CheckMask( mask, Button1Mask) ){
		maskcat( buf, "Button1Mask");
	}
	if( CheckMask( mask, Button2Mask) ){
		maskcat( buf, "Button2Mask");
	}
	if( CheckMask( mask, Button3Mask) ){
		maskcat( buf, "Button3Mask");
	}
	if( CheckMask( mask, Button4Mask) ){
		maskcat( buf, "Button4Mask");
	}
	if( CheckMask( mask, Button5Mask) ){
		maskcat( buf, "Button5Mask");
	}
	if( CheckMask( mask, ShiftMask) ){
		maskcat( buf, "ShiftMask");
	}
	if( CheckMask( mask, ControlMask) ){
		maskcat( buf, "ControlMask");
	}
	if( CheckMask( mask, LockMask) ){
		maskcat( buf, "LockMask");
	}
	if( CheckMask( mask, Mod1Mask) ){
		maskcat( buf, "Mod1Mask");
	}
	if( CheckMask( mask, Mod2Mask) ){
		maskcat( buf, "Mod2Mask");
	}
	if( CheckMask( mask, Mod3Mask) ){
		maskcat( buf, "Mod3Mask");
	}
	if( CheckMask( mask, Mod4Mask) ){
		maskcat( buf, "Mod4Mask");
	}
	if( CheckMask( mask, Mod5Mask) ){
		maskcat( buf, "Mod5Mask");
	}
	return( buf );
}

static xtb_hret sr_h(evt, info)
XEvent *evt;
xtb_data info;
/*
 * Handles text input events.
 */
{
	Window evwin = evt->xany.window;
	sr_info *ri = (sr_info *) info;
	xtb_frame *frame= ri->frame;
	Window win = frame->win;
	// 20080711: if new, unwanted behaviour, set rtn to default value XTB_NOTDEF!
	xtb_hret rtn= XTB_NOTDEF;
/*  XButtonEvent *bev= (XButtonEvent*) evt; */
	double value;
	Window root_win, win_win;
	int root_x, root_y, win_x, win_y, mask, nbytes;
	KeySym keysym;
	char keys[1];

	if( !frame || !ri ){
		return( XTB_NOTDEF );
	}
	switch (evt->type) {
		case UnmapNotify:
			if( ri->frame ) ri->frame->mapped= 0;
			break;
		case MapNotify:
			if( ri->frame ) ri->frame->mapped= 1;
			break;
		case ConfigureNotify:
		case VisibilityNotify:
			if( ri->frame ) ri->frame->mapped= 1;
			ri->redraw= 1;
			break;
		case Expose:
			if( evt->xexpose.count<= 0 ){
				sr_draw(win, ri, ri->clear_flag );
			}
			else{
				ri->redraw= 1;
			}
			rtn = XTB_HANDLED;
			if( debugFlag ){
				fprintf( StdErr, "xtb::sr_h(%d): slide at (%s@%d); %s event for %s window\n",
					__LINE__, ri->text, ri->position, event_name(evt->type), sr_win( ri, evwin)
				);
			}
			break;
		case KeyRelease:
		 /* This probably never does anything. */
		case ButtonRelease:
			ri->button_pressed= False;
			if( !ri->button_handled ){
				ri->button_handled= True;
				if( ri->frame->enabled ){
					value= ri->value;
					rtn = (*ri->func)(win, ri->position, ri->value, ri->val);
					if( debugFlag ){
						fprintf( StdErr, "xtb::sr_h(%d): button/keyrelease in %s window; slide callback left slide at (%s@%d) (%s->%s)\n",
							__LINE__, sr_win(ri, evwin), ri->text, ri->position,
							d2str( value, "%g", NULL), d2str( ri->value, "%g", NULL)
						);
					}
				}
				else{
					Boing(5);
					rtn= XTB_HANDLED;
				}
				if( rtn == XTB_STOP ){
					break;
				}
			}
			break;
		case KeyPress:
			nbytes = XLookupString(&evt->xkey, keys, 1,
						   (KeySym *) &keysym, (XComposeStatus *) 0
			);
			keysym&= 0x0000FFFF;
			if( keysym== ((ri->vert_flag)? XK_Down : XK_Left) ){
			 /* Left-cursor: decrement. Set the eventwindow to ri->less
               \ (the decrement-window), and fall through to the button-press
               \ handler.
               */
				evwin= ri->less;
			}
			else if( keysym== ((ri->vert_flag)? XK_Up : XK_Right) ){
				evwin= ri->more;
			}
			else{
			 /* Not a valid key. Until next event...   */
				break;
			}
			/* fall through to ButtonPress  */
		case ButtonPress:{
			if( !ri->frame->enabled ){
				Boing(5);
				break;
			}
			ri->button_pressed= True;
			if( evwin!= ri->less && evwin!= ri->more ){
				ri->prev_value= ri->value;
				ri->prev_position= ri->position;
				ri->button_handled= False;
			}
			else if( evwin== ri->less && !NaN(ri->value) ){
			 unsigned long continue_mask=(Button2Mask | Button3Mask | Button4Mask | Button5Mask);
				while( ri->button_pressed ){
					ri->prev_position= ri->position;
					ri->prev_value= ri->value;
					if( ri->int_flag ){
						ri->value-= 1;
					}
					else{
						ri->position-= 1;
						ri->value= ri->position* ri->delta+ ri->minvalue;
					}
					CLIP( ri->value, ri->_minvalue, ri->_maxvalue );
					if( ri->value!= ri->prev_value ){
						xtb_sr_set_( ri, ri->value );
						if( debugFlag ){
							fprintf( StdErr, "xtb::sr_h(%d): button/keypress in %s window; decreasing slide from (%s@%d) to (%s@%d)\n",
								__LINE__, sr_win(ri, evwin),
								d2str( ri->prev_value, "%g", NULL), ri->prev_position,
								ri->text, ri->position
							);
						}
						sr_draw_slide( win, ri);
					}
					 /* QueryPointer to determine button and modifier states:  */
					XQueryPointer( t_disp, evwin, &root_win, &win_win,
						&root_x, &root_y, &win_x, &win_y, &mask
					);
					mask= xtb_Mod2toMod1(mask);
					ri->button_handled= False;
					if( mask& continue_mask ){
						ri->button_pressed= True;
					}
					else{
						ri->button_pressed= False;
					}
					if( !CheckMask(mask, ShiftMask) ){
						ri->button_handled= True;
						if( (rtn = (*ri->func)(win, ri->position, ri->value, ri->val)) == XTB_STOP ){
							break;
						}
					}
					if( debugFlag && debugLevel ){
						fprintf( StdErr, "xtb::sr_h(%d): press in %s window, slide at (%s@%d), button=%d, handled=%d, mask=%s\n",
							__LINE__, sr_win(ri, evwin), ri->text, ri->position, ri->button_pressed, ri->button_handled,
							xtb_modifiers_string(mask)
						);
					}
				}
			}
			else if( evwin== ri->more && !NaN(ri->value) ){
			 unsigned long continue_mask=(Button2Mask | Button3Mask | Button4Mask | Button5Mask);
				while( ri->button_pressed ){
					ri->prev_position= ri->position;
					ri->prev_value= ri->value;
					if( ri->int_flag ){
						ri->value+= 1;
					}
					else{
						ri->position+= 1;
						ri->value= ri->position* ri->delta+ ri->minvalue;
					}
					CLIP( ri->value, ri->_minvalue, ri->_maxvalue );
					if( ri->value!= ri->prev_value ){
						xtb_sr_set_( ri, ri->value );
						if( debugFlag ){
							fprintf( StdErr, "xtb::sr_h(%d): button/keypress in %s window, increasing slide from (%s@%d) to (%s@%d)\n",
								__LINE__, sr_win(ri, evwin),
								d2str( ri->prev_value, "%g", NULL), ri->prev_position,
								ri->text, ri->position
							);
						}
						sr_draw_slide( win, ri);
					}
					 /* QueryPointer to determine button and modifier states:  */
					XQueryPointer( t_disp, evwin, &root_win, &win_win,
						&root_x, &root_y, &win_x, &win_y, &mask
					);
					mask= xtb_Mod2toMod1(mask);
					ri->button_handled= False;
					if( mask& continue_mask ){
						ri->button_pressed= True;
					}
					else{
						ri->button_pressed= False;
					}
					if( !CheckMask(mask, ShiftMask) ){
						ri->button_handled= True;
						if( (rtn = (*ri->func)(win, ri->position, ri->value, ri->val)) == XTB_STOP ){
							break;
						}
					}
					if( debugFlag && debugLevel ){
						fprintf( StdErr, "xtb::sr_h(%d): press in %s window, slide at (%s@%d), button=%d, handled=%d, mask=%s\n",
							__LINE__, sr_win(ri,evwin),
							ri->text, ri->position, ri->button_pressed, ri->button_handled,
							xtb_modifiers_string(mask)
						);
					}
				}
			}
		}
		 /* Fall-through to set new position when dragging:    */
		case MotionNotify:
			if( ri->frame->enabled && evwin!= ri->less && evwin!= ri->more ){
			 unsigned long buttons= (Button1Mask | Button2Mask | Button3Mask | Button4Mask | Button5Mask);
				XQueryPointer( t_disp, win, &root_win, &win_win,
					&root_x, &root_y, &win_x, &win_y, &mask
				);
				mask= xtb_Mod2toMod1(mask);
				if( !ri->button_pressed ){
					if( mask & buttons ){
						ri->button_pressed= True;
					}
				}
				if( ri->button_pressed ){
				 int pos;
					if( ri->vert_flag ){
						pos= (ri->frame->height- 2*SR_BRDR- win_y)- ri->minposition;
					}
					else{
						pos= win_x- ri->minposition;
					}
					value= pos* ri->delta+ ri->minvalue;
					CLIP( value, ri->_minvalue, ri->_maxvalue );
					if( ri->int_flag ){
						value= floor( value /* + 0.5*/ );
						pos = (int) ( (ri->delta)? (value- ri->minvalue)/ ri->delta : ri->minposition );
					}
					CLIP( pos, 0, ri->maxposition- ri->minposition);
					if( debugFlag ){
						fprintf( StdErr, "xtb::sr_h(%d): Motion in %s window, dragging slide from (%s@%d) to (%s@%d)\n",
							__LINE__, sr_win(ri, evwin), ri->text, ri->position,
							d2str( value, "%g", NULL), pos
						);
					}
					if( pos!= ri->prev_position ){
						ri->prev_position= ri->position;
						ri->prev_value= ri->value;
						ri->position= pos;
						ri->value= value;
						d2str( value, "%g", ri->text );
						sr_draw_slide( win, ri);
						if( debugFlag && debugLevel ){
							fprintf( StdErr, "xtb::sr_h(%d): slide at (%s@%d), button=%d\n",
								__LINE__, ri->text, ri->position, ri->button_pressed
							);
						}
						if( !CheckMask(mask, ShiftMask) ){
							ri->button_handled= True;
							if( (rtn = (*ri->func)(win, ri->position, ri->value, ri->val)) == XTB_STOP ){
								break;
							}
						}
					}
				}
				else if( debugFlag && debugLevel== -2 ){
						fprintf( StdErr, "xtb::sr_h(%d): \"unpressed\" MotionNotify in %s window, button=%d, handled=%d, mask=%s\n",
							__LINE__, sr_win(ri, evwin), ri->button_pressed, ri->button_handled, xtb_modifiers_string(mask)
						);
				}
			}
			else if( !ri->frame->enabled ){
				Boing(5);
			}
			break;
		default:
			rtn = XTB_NOTDEF;
			break;
	}
	return rtn;
}

/* Userfunction that does the same, and redraws the ruler   */
void xtb_sr_set(Window win, double value, xtb_data *val)
{ sr_info *info = (sr_info *) xtb_lookup(win);

	if( !info ){
		return;
	}
	if( val ){
		info->val= *val;
	}
	if( debugFlag ){
		fprintf( StdErr, "xtb::xtb_sr_set(%d): setting slide from %s to %s\n",
			__LINE__, info->text, d2str( value, "%g", NULL)
		);
	}
	xtb_sr_set_( info, value );
	sr_draw( win, info, 0);
}

/* Userfunction that returs the slider's value returns the current position of the ruler in *position   */
double xtb_sr_get(Window win, int *position, xtb_data *val)
{ sr_info *info = (sr_info *) xtb_lookup(win);

	if( !info ){
		return 0;
	}
	if( val ){
		*val = info->val;
	}
	*position= info->position;
	return( info->value );
}

/* Lowlevel function that changes the min/max and scale settings of the slideruler  */
static void xtb_sr_set_scale_( sr_info *info, double minval, double maxval )
{
	if( !info || NaN(minval) || NaN(maxval) ){
		return;
	}
	info->minvalue = (info->int_flag)? floor( minval /* + 0.5*/ ) : minval;
	info->maxvalue= (info->int_flag)? floor( maxval /* + 0.5*/ ) : maxval;
	info->_minvalue= MIN( info->minvalue, info->maxvalue );
	info->_maxvalue= MAX( info->minvalue, info->maxvalue );
	info->delta= (maxval- minval)/ info->line_w;
}

/* Userfunction that does the same, updates the position of the slider, and redraws */
void xtb_sr_set_scale(Window win, double minval, double maxval, xtb_data *val)
{ sr_info *info = (sr_info *) xtb_lookup(win);

	if( !info ){
		return;
	}
	if( val ){
		info->val= *val;
	}
	xtb_sr_set_scale_( info, minval, maxval );
	xtb_sr_set_( info, info->value );
	sr_draw( win, info, 1);
}

/* Userfunction that does the same, updates the position of the slider, and redraws */
void xtb_sr_set_scale_integer(Window win, double minval, double maxval, int int_flag, xtb_data *val)
{ sr_info *info = (sr_info *) xtb_lookup(win);

	if( !info ){
		return;
	}
	if( val ){
		info->val= *val;
	}
	info->int_flag= int_flag;
	xtb_sr_set_scale_( info, minval, maxval );
	xtb_sr_set_( info, info->value );
	sr_draw( win, info, 1);
}

/* Userfunction that returns the current low and/or high bounds in *minval and/or *maxval,
 \ depending on whether valid pointers where passed.
 */
void xtb_sr_get_scale(Window win, double *minval, double *maxval, xtb_data *val )
{ sr_info *info = (sr_info *) xtb_lookup(win);

	if( !info ){
		return;
	}
	if( val ){
		info->val= *val;
	}
	if( minval ) *minval= info->minvalue;
	if( maxval ) *maxval= info->maxvalue;
}

/* Userfunction that can change the int_flag    */
void xtb_sr_set_integer(Window win, int int_flag, xtb_data *val )
{ sr_info *info = (sr_info *) xtb_lookup(win);

	if( !info ){
		return;
	}
	if( val ){
		info->val= *val;
	}
	info->int_flag= int_flag;
	xtb_sr_set_scale_( info, info->minvalue, info->maxvalue );
	xtb_sr_set_( info, info->value );
	sr_draw( win, info, 1);
}

/* Userfunction that returns the int_flag   */
int xtb_sr_get_integer(Window win, xtb_data *val )
{ sr_info *info = (sr_info *) xtb_lookup(win);

	if( !info ){
		return 0;
	}
	if( val ){
		info->val= *val;
	}
	return( info->int_flag );
}

/* Create a new slideruler, specifying its bounds, initial position, size
 \ and the callback routine. The size can be specified as a positive or negative
 \ number. In the former case, it is interpreted as the size. In the latter case,
 \ the absolute value is multiplied with the width of the normal font (norm_font)
 \ to obtain the size of the ruler. This allows the ruler to be matched in size
 \ with a ti_info widget.
 */
static sr_info *xtb_sr_new_(win, minval, maxval, init_val, size, func, val, frame, int_flag, vert_flag )
Window win;   		/* Parent window      */
double init_val, minval, maxval;
int size;
FNPTR( func, xtb_hret, (Window, int, double, xtb_data) ); /* Callback */
xtb_data val;			/* User data          */
xtb_frame *frame;		/* Returned size      */
int int_flag, vert_flag;
{
	struct sr_info *info;
	char mintext[64], maxtext[64], name[128];
	int min_w, max_w;

	if( size== 0 ){
		fprintf( StdErr, "xtb_sr_new_(%s,%s,%s,...%s,%s): zero size requested\n",
			d2str( minval, "%g", NULL),
			d2str( maxval, "%g", NULL),
			d2str( init_val, "%g", NULL),
			(int_flag)? "integer" : "float",
			(vert_flag)? "vertical" : "horizontal"
		);
		return(NULL);
	}
	if( !(frame->framelist= (xtb_frame**) malloc(sizeof(xtb_frame*))) ){
		return(NULL);
	}
	if( !(info = (sr_info *) calloc(1, sizeof(sr_info))) ){
		return(NULL);
	}
	info->type= xtb_SR;

	 /* The size of the slider (a separate window) depends on the bounds:
       \ we try to "normally" print a number just within those bounds, and see
       \ which is larger. When int_flag is true (a slider for integers), we
       \ just look at the bounds.
       */
	if( int_flag ){
		sprintf( mintext, "%d", (int) minval );
		sprintf( maxtext, "%d", (int) maxval );
	}
	else{
		d2str( 0.987987* minval, "%g", mintext);
		d2str( (1-0.987987)* maxval, "%g", maxtext);
	}
	min_w = XTextWidth(norm_font, mintext, strlen(mintext) ) + 2*SR_HPAD+ 2* SR_SBRDR;
	max_w = XTextWidth(norm_font, maxtext, strlen(maxtext) ) + 2*SR_HPAD+ 2* SR_SBRDR;
	if( max_w > min_w ){
		info->slide_w= max_w;
	}
	else{
		info->slide_w= min_w;
	}
	info->slide_h = norm_font->ascent + norm_font->descent + SR_VPAD + SR_LPAD;

	if( vert_flag ){
		frame->width = info->slide_w + 2* SR_HPAD+ 2* SR_SBRDR;
		info->lr_w= frame->width- 2* SR_SBRDR;
		info->lr_h= 8;
		if( size> 0 ){
			frame->height= size;
		}
		else{
			frame->height= XFontWidth( norm_font)* -1 * size;
			size= frame->height;
		}
		frame->height+= 2* info->lr_h+ 2* SR_BRDR;
		info->line_w= size- info->slide_h;
	}
	else{
/* 		frame->height = norm_font->ascent + norm_font->descent + 2* SR_VPAD+ 2* SR_SBRDR;	*/
		frame->height = max_ascent + max_descent + 2* SR_VPAD+ 2* SR_SBRDR;
		info->lr_w= 8;
		info->lr_h= frame->height- 2* SR_SBRDR;
		if( size> 0 ){
			frame->width= size;
		}
		else{
			frame->width= XFontWidth( norm_font)* -1 * size;
			size= frame->width;
		}
		frame->width+= 2* info->lr_w+ 2* SR_BRDR;
		info->line_w= size- info->slide_w;
	}
	info->vert_flag= vert_flag;

	frame->x_loc = frame->y_loc = 0;
	frame->description= NULL;
	XSelectInput( t_disp, win,
		ExposureMask|StructureNotifyMask|ButtonPressMask|ButtonReleaseMask|KeyPressMask
	);
	if( SetParentBackground )
		XSetWindowBackground( t_disp, win, xtb_light_pix );

	frame->border= SR_BRDR;
	frame->win = xtb_XCreateSimpleWindow(t_disp, win, 0, 0,
					frame->width, frame->height, SR_BRDR,
					xtb_norm_pix, xtb_light_pix
	);
	XSetWindowColormap( t_disp, frame->win, cmap );
	XSelectInput(t_disp, frame->win, VisibilityChangeMask|StructureNotifyMask|ExposureMask|KeyPressMask|
		EnterWindowMask|LeaveWindowMask|
		ButtonPressMask|PointerMotionMask|PointerMotionHintMask|
		ButtonReleaseMask
	);

	if( !sr_hcurs ){
		sr_hcurs = XCreateFontCursor(t_disp, XC_sb_h_double_arrow );
	}
	if( !sr_vcurs ){
		sr_vcurs = XCreateFontCursor(t_disp, XC_sb_v_double_arrow );
	}
	if( !sr_hcursl ){
		sr_hcursl = XCreateFontCursor(t_disp, XC_sb_left_arrow );
	}
	if( !sr_hcursr ){
		sr_hcursr = XCreateFontCursor(t_disp, XC_sb_right_arrow );
	}
	if( !sr_vcursu ){
		sr_vcursu = XCreateFontCursor(t_disp, XC_sb_up_arrow );
	}
	if( !sr_vcursd ){
		sr_vcursd = XCreateFontCursor(t_disp, XC_sb_down_arrow );
	}
	fg_color.pixel = xtb_norm_pix;
	XQueryColor(t_disp, cmap, &fg_color);
	bg_color.pixel = xtb_back_pix;
	XQueryColor(t_disp, cmap, &bg_color);
	XRecolorCursor(t_disp, sr_hcurs, &fg_color, &bg_color);
	XRecolorCursor(t_disp, sr_vcurs, &fg_color, &bg_color);
	XRecolorCursor(t_disp, sr_hcursl, &fg_color, &bg_color);
	XRecolorCursor(t_disp, sr_hcursr, &fg_color, &bg_color);
	XRecolorCursor(t_disp, sr_vcursu, &fg_color, &bg_color);
	XRecolorCursor(t_disp, sr_vcursd, &fg_color, &bg_color);

	XDefineCursor(t_disp, frame->win, (vert_flag)? sr_vcurs : sr_hcurs);

	frame->parent= win;
	frame->frames= 1;
	frame->framelist[0]= frame;
	info->func = func;
	CLIP( init_val, minval, maxval);
	info->value = init_val;
	info->val= val;
	info->norm_pix = xtb_norm_pix;
	info->light_pix= xtb_light_pix;
	info->middle_pix= xtb_middle_pix;
	info->back_pix = xtb_back_pix;
	info->font = norm_font;
	info->alt_font= greek_font;
	info->frame= frame;
	info->frame->enabled= True;
	frame->info= (xtb_registry_info*) info;

	info->int_flag= int_flag;

	if( vert_flag ){
		info->minposition= info->slide_h/ 2+ info->lr_h+ SR_SBRDR;
		info->maxposition= size- info->slide_h/2+ info->lr_h+ SR_SBRDR;
		info->line_x = frame->width/ 2;
		info->slide_x= info->line_x- info->slide_w/ 2- SR_VPAD;
	}
	else{
		info->minposition= info->slide_w/ 2+ info->lr_w+ SR_SBRDR;
		info->maxposition= size- info->slide_w/2+ info->lr_w+ SR_SBRDR;
		info->line_y = frame->height/ 2;
		info->slide_y= info->line_y- info->slide_h/ 2- SR_VPAD;
	}
	xtb_sr_set_scale_( info, minval, maxval);
	xtb_sr_set_( info, init_val );
	if( vert_flag ){
		info->slide_ = xtb_XCreateSimpleWindow(t_disp, frame->win, 0, 0,
						info->slide_w, 1, SR_BRDR,
						xtb_middle_pix, xtb_light_pix
		);
	}
	else{
		info->slide_ = xtb_XCreateSimpleWindow(t_disp, frame->win, 0, 0,
						1, info->slide_h, SR_BRDR,
						xtb_middle_pix, xtb_light_pix
		);
	}
	XSetWindowColormap( t_disp, info->slide_, cmap );
	XSetWindowBorder( t_disp, info->slide_, info->middle_pix );
	XSelectInput(t_disp, info->slide_, VisibilityChangeMask|StructureNotifyMask|ExposureMask|KeyPressMask|
		EnterWindowMask|LeaveWindowMask|
		ButtonPressMask|PointerMotionMask|PointerMotionHintMask|
		ButtonReleaseMask
	);
	XDefineCursor(t_disp, info->slide_, (vert_flag)? sr_vcurs : sr_hcurs);

	info->slide = xtb_XCreateSimpleWindow(t_disp, frame->win, 0, 0,
					info->slide_w+ 1, info->slide_h, SR_BRDR,
					xtb_norm_pix, xtb_light_pix
	);
	XSetWindowColormap( t_disp, info->slide, cmap );
	XSelectInput(t_disp, info->slide, VisibilityChangeMask|StructureNotifyMask|ExposureMask|KeyPressMask|
		EnterWindowMask|LeaveWindowMask|
		ButtonPressMask|PointerMotionMask|PointerMotionHintMask|
		ButtonReleaseMask
	);
	XDefineCursor(t_disp, info->slide, (vert_flag)? sr_vcurs : sr_hcurs);

	info->less = xtb_XCreateSimpleWindow(t_disp, frame->win, 0, 0,
					info->lr_w, info->lr_h, SR_SBRDR,
					xtb_norm_pix, xtb_middle_pix
	);
	XSetWindowColormap( t_disp, info->less, cmap );
	XSelectInput(t_disp, info->less, VisibilityChangeMask|StructureNotifyMask|ExposureMask|KeyPressMask|
		EnterWindowMask|LeaveWindowMask|
		ButtonPressMask|PointerMotionMask|PointerMotionHintMask|
		ButtonReleaseMask
	);
	info->more = xtb_XCreateSimpleWindow(t_disp, frame->win, 0, 0,
					info->lr_w, info->lr_h, SR_SBRDR,
					xtb_norm_pix, xtb_middle_pix
	);
	XSetWindowColormap( t_disp, info->more, cmap );
	XSelectInput(t_disp, info->more, VisibilityChangeMask|StructureNotifyMask|ExposureMask|KeyPressMask|
		EnterWindowMask|LeaveWindowMask|
		ButtonPressMask|PointerMotionMask|PointerMotionHintMask|
		ButtonReleaseMask
	);
	XDefineCursor(t_disp, info->less, (vert_flag)? sr_vcursd : sr_hcursl);
	XDefineCursor(t_disp, info->more, (vert_flag)? sr_vcursu : sr_hcursr);

	xtb_register(frame, frame->win, sr_h, (xtb_registry_info*) info);
	xtb_register_(frame, info->slide_, sr_h, (xtb_registry_info*) info);
	xtb_register_(frame, info->slide, sr_h, (xtb_registry_info*) info);
	xtb_register_(frame, info->less, sr_h, (xtb_registry_info*) info);
	xtb_register_(frame, info->more, sr_h, (xtb_registry_info*) info);

	sprintf( name, "xtb_sr frame #%d", xtb_sr_serial );
	XStoreName( t_disp, frame->win, name );
	sprintf( name, "xtb_sr slide_ #%d", xtb_sr_serial );
	XStoreName( t_disp, info->slide_, name );
	sprintf( name, "xtb_sr slide #%d", xtb_sr_serial );
	XStoreName( t_disp, info->slide, name );
	sprintf( name, "xtb_sr less #%d", xtb_sr_serial );
	XStoreName( t_disp, info->less, name );
	sprintf( name, "xtb_sr more #%d", xtb_sr_serial );
	XStoreName( t_disp, info->more, name );
	xtb_sr_serial+= 1;

	XMapWindow(t_disp, info->slide_);
	XMapWindow(t_disp, info->slide);
	XMapWindow( t_disp, info->less);
	XMapWindow( t_disp, info->more);

	XMapWindow(t_disp, frame->win);

	XNextEvent( t_disp, &evt);
	xtb_dispatch( t_disp, frame->win, 0, NULL, &evt);
	frame->width += (2 * SR_BRDR);
	frame->height += (2 * SR_BRDR);
/*  frame->mapped= 1;   */
	frame->redraw= xtb_sr_redraw;
	frame->destroy= xtb_sr_del;
	frame->chfnt= xtb_sr_chfnt;
	return( info );
}

void xtb_sr_new(win, minval, maxval, init_val, size, vert_flag, func, val, frame)
Window win;   		/* Parent window      */
double init_val, minval, maxval;
int size, vert_flag;
FNPTR( func, xtb_hret, (Window, int, double, xtb_data) ); /* Callback */
xtb_data val;			/* User data          */
xtb_frame *frame;		/* Returned size      */
{
	xtb_sr_new_( win, minval, maxval, init_val, size, func, val, frame, 0, vert_flag );
}

void xtb_sri_new(win, minval, maxval, init_val, size, vert_flag, func, val, frame)
Window win;   		/* Parent window      */
int init_val, minval, maxval;
int size, vert_flag;
FNPTR( func, xtb_hret, (Window, int, double, xtb_data) ); /* Callback */
xtb_data val;			/* User data          */
xtb_frame *frame;		/* Returned size      */
{
	xtb_sr_new_( win, (double) minval, (double) maxval, (double) init_val, size, func, val, frame, 1, vert_flag );
}

int xtb_sr_chfnt( Window win, XFontStruct *font, XFontStruct *alt_font)
{   struct sr_info *ri= (struct sr_info*) xtb_lookup(win);
	if( ri ){
		ri->font= font;
		ri->alt_font= alt_font;
		sr_draw( win, ri, True );
		return(0);
	}
	else{
		return(1);
	}
}

void xtb_sr_redraw( win)
Window win;
{   struct sr_info *ri= (struct sr_info*) xtb_lookup(win);
	sr_draw( win, ri, True);
	return;
}

void xtb_sr_del(Window win, xtb_data *info)
/*
 * Deletes an input text widget.  User defined data is returned in `info'.
 */
{
	sr_info *si, *sri;

	if( xtb_unregister(win, (xtb_registry_info* *) &si)) {
		if( si ){
			if( info ){
				*info = si->val;
			}
			xtb_unregister( si->more, (xtb_registry_info* *) &sri);
			XDestroyWindow( t_disp, si->more);
			xtb_unregister( si->less, (xtb_registry_info* *) &sri);
			XDestroyWindow( t_disp, si->less);
			xtb_unregister( si->slide, (xtb_registry_info* *) &sri);
			XDestroyWindow( t_disp, si->slide);
			xtb_unregister( si->slide_, (xtb_registry_info* *) &sri);
			XDestroyWindow( t_disp, si->slide_);

			if( si->frame ){
				xfree( si->frame->framelist );
			}
			xfree( si);
		}
		else if( info ){
			*info= NULL;
		}

		XDestroyWindow(t_disp, win);
	}
}

#endif

/*
 * Formatting support
 */

// #define FATALERROR(msg) fprintf(StdErr,"%s\n", msg); abort();

#define FATALERROR(msg) fprintf(StdErr,"Potentially fatal error '%s' in %s:%d (%s)\n", msg, __FILE__, __LINE__, __FUNCTION__);

int xtb_fmt_tabset= 0;

xtb_fmt *xtb_w(w)
xtb_frame *w;
/*
 * Returns formatting structure for a widget.
 */
{ xtb_fmt *ret;

	ret = (xtb_fmt *) malloc((unsigned) sizeof(xtb_fmt));
	ret->wid.type = W_TYPE;
	ret->wid.w = w;
	if( w && xtb_fmt_tabset ){
		w->tabset= xtb_fmt_tabset;
	}
	return ret;
}

xtb_fmt *xtb_ws( xtb_frame *w, int set )
/*
 * Returns formatting structure for a widget.
 */
{ xtb_fmt *ret;

	ret = (xtb_fmt *) malloc((unsigned) sizeof(xtb_fmt));
	ret->wid.type = W_TYPE;
	ret->wid.w = w;
	if( w ){
		w->tabset= set;
	}
	return ret;
}

/* 960806 RJB
 \ HPUX cc seems not to like mixing arguments, and varargs. Giving
 \ first some argument(s), and then a variable list works in gcc and
 \ other ANSI compilers :( Therefore, I commented out those arguments,
 \ and changed the __STDC__ conditional code into HPUX8__STDC__ ...
 \ Not the most elegant solution (prototypes...), but it works.
 */

#ifdef CTAGS
xtb_fmt *xtb_hort(xtb_just just, int padding, int interspace,	VA_DCL )
#endif
xtb_fmt *xtb_hort(xtb_just just, int padding, int interspace, VA_DCL)
/*
 * Builds a horizontal structure
 */
{
	va_list ap;
	xtb_fmt *ret, *val;
	unsigned long width= 0;
#ifdef HPUX8__STDC__

	va_start(ap);
#else
	va_start(ap, interspace);
#endif
	ret = (xtb_fmt *) malloc((unsigned) sizeof(xtb_fmt));
	ret->align.type = A_TYPE;
	ret->align.dir = HORIZONTAL;
	ret->align.just = just;
	ret->align.padding = padding;
	ret->align.interspace = interspace;
	  /* Build array of incoming xtb_fmt structures */
	ret->align.ni = 0;
	while ((val = va_arg(ap, xtb_fmt *)) != (xtb_fmt *) 0) {
		if (ret->align.ni < MAX_BRANCH) {
			ret->align.items[ret->align.ni] = val;
			ret->align.ni++;
			if( val->wid.type== W_TYPE && val->wid.w && val->wid.w->mapped ){
				width+= val->wid.w->width;
			}
		} else {
			FATALERROR("too many branches\n");
		}
	}
/* 	if( !width ){	*/
/* 		ret->align.padding= ret->align.interspace= 0;	*/
/* 	}	*/
	return ret;
}


xtb_fmt *xtb_hort_cum(xtb_just just, int padding, int interspace,	xtb_fmt *val )
/*
 * Builds a horizontal structure, cumulatively callable version
 */
{
	static xtb_fmt *Ret= NULL;
	xtb_fmt *ret;
	unsigned long width= 0;

	if( val && !Ret ){
		Ret = (xtb_fmt *) calloc(1, (unsigned) sizeof(xtb_fmt));
	}
	if( (ret= Ret) && val ){
		ret->align.type = A_TYPE;
		ret->align.dir = HORIZONTAL;
		ret->align.just = just;
		ret->align.padding = padding;
		ret->align.interspace = interspace;
		  /* Build array of incoming xtb_fmt structures */
		{
			if (ret->align.ni < MAX_BRANCH) {
				ret->align.items[ret->align.ni] = val;
				ret->align.ni++;
				if( val->wid.type== W_TYPE && val->wid.w && val->wid.w->mapped ){
					width+= val->wid.w->width;
				}
			} else {
				fprintf( StdErr, "too many branches in xtb_hort_cum (max. %d)\n", MAX_BRANCH );
				return( NULL );
			}
		}
	}
	if( !val && Ret ){
	  /* reset for next invocation... */
		Ret= NULL;
	}
	return ret;
}

#ifdef CTAGS
xtb_fmt *xtb_vert(xtb_just just, int padding, int interspace,   VA_DCL   )
#endif
xtb_fmt *xtb_vert(xtb_just just, int padding, int interspace, VA_DCL )
/*
 * Builds a vertical structure
 */
{
	va_list ap;
	xtb_fmt *ret, *val;
	unsigned long height= 0;
#ifdef HPUX8__STDC__

	va_start(ap);
#else
	va_start(ap, interspace);
#endif
	ret = (xtb_fmt *) malloc((unsigned) sizeof(xtb_fmt));
	ret->align.type = A_TYPE;
	ret->align.dir = VERTICAL;
	ret->align.just = just;
	ret->align.padding = padding;
	ret->align.interspace = interspace;
	/* Build array of incoming xtb_fmt structures */
	ret->align.ni = 0;
	while ((val = va_arg(ap, xtb_fmt *)) != (xtb_fmt *) 0) {
		if (ret->align.ni < MAX_BRANCH) {
			ret->align.items[ret->align.ni] = val;
			ret->align.ni++;
			if( val->wid.type== W_TYPE && val->wid.w && val->wid.w->mapped ){
				height+= val->wid.w->height;
			}
		} else {
			FATALERROR("too many branches\n");
		}
	}
/* 	if( !height ){	*/
/* 		ret->align.padding= ret->align.interspace= 0;	*/
/* 	}	*/
	return ret;
}

xtb_fmt *xtb_vert_cum(xtb_just just, int padding, int interspace,   xtb_fmt *val   )
/*
 * Builds a vertical structure, cumulatively callable version.
 */
{
	static xtb_fmt *Ret= NULL;
	xtb_fmt *ret;
	unsigned long height= 0;

	if( val && !Ret ){
		Ret = (xtb_fmt *) calloc(1, (unsigned) sizeof(xtb_fmt));
	}
	ret= Ret;
	if( ret && val ){
		ret->align.type = A_TYPE;
		ret->align.dir = VERTICAL;
		ret->align.just = just;
		ret->align.padding = padding;
		ret->align.interspace = interspace;
		/* Build array of incoming xtb_fmt structures */
		if (ret->align.ni < MAX_BRANCH) {
			ret->align.items[ret->align.ni] = val;
			ret->align.ni++;
			if( val->wid.type== W_TYPE && val->wid.w && val->wid.w->mapped ){
				height+= val->wid.w->height;
			}
		} else {
			fprintf( StdErr, "too many branches in xtb_vert_cum (max. %d)\n", MAX_BRANCH );
			return( NULL );
		}
	}
	if( !val && Ret ){
		Ret= NULL;
	}
	return ret;
}

static void xtb_fmt_setpos(def, x, y)
xtb_fmt *def;
int x, y;
/*
 * Sets all position fields of widgets in `def' to x,y.
 */
{
	int i;

	switch( def->type ){
		case W_TYPE:
			if( def->wid.w->mapped ){
				def->wid.w->x_loc = x;
				def->wid.w->y_loc = y;
			}
			break;
		case A_TYPE:
			for (i = 0;   i < def->align.ni;   i++) {
				xtb_fmt_setpos(def->align.items[i], x, y);
			}
			break;
		default:
			FATALERROR("bad type");
	}
}


static void xtb_fmt_addpos(def, x, y)
xtb_fmt *def;
int x, y;
/*
 * Adds the offset specified to all position fields of widgets in `def'.
 */
{
	int i;

	switch (def->type) {
		case W_TYPE:
			if( def->wid.w->mapped ){
				def->wid.w->x_loc += x;
				def->wid.w->y_loc += y;
			}
			break;
		case A_TYPE:
			for (i = 0;   i < def->align.ni;   i++) {
				xtb_fmt_addpos(def->align.items[i], x, y);
			}
			break;
		default:
			FATALERROR("bad type");
	}
}

static int _last_width, _last_height;
static int last_h_update= 0, last_w_update= 0;
static int last_width, last_height;

static void xtb_fmt_hort(nd, defs, widths, heights, just, pad, inter, rw, rh)
int nd;   			/* Number of children     */
xtb_fmt *defs[];		/* Definitions themselves */
int widths[];			/* Widths of children     */
int heights[];			/* Heights of children    */
xtb_just just;			/* Justification          */
int pad, inter;   		/* Padding and interspace */
int *rw, *rh;			/* Returned size          */
/*
 * Formats items horizontally subject to the widths and heights
 * of the items passed.
 */
{
	int i;
	int max_height = 0;
	int tot_width = 0;
	int xspot;
	double _inter= inter, _xspot, _xend= 0;

	 /* Find parameters */
	for (i = 0;   i < nd;   i++) {
		if (heights[i] > max_height) max_height = heights[i];
		tot_width += widths[i];
	}
	 /* This is the resulting width without justification: */
	*rw = tot_width + (nd-1)*inter + (2 * pad);
	_xspot= xspot = pad;
	if( (just== XTB_JUST || just== XTB_JUST_J) && nd> 1 && last_width> *rw ){
		  /* Calculate a new inter-widget spacing, not just depending on what user
		   \ specified, but also on a previously determined formatting width.
		   \ One would expect _inter=(last_width-tot_width)/(nd-1.0) to work better
		   \ (no dependence on padding or specified inter), but it does not...
		   */
		_inter= (last_width- (*rw- 2.0*pad))/(nd-1.0)+ 2* pad- 1;
		  /* In this case, we don't want an initial offset.	*/
		_xspot= xspot= 0;
	}
	if( just== XTB_JUST_J ){
		last_w_update= 1;
	}
	 /* Place items -- assumes center justification */
	for (i = 0;   i < nd;   i++) {
		switch (just) {
			case XTB_TOP_J:
					last_w_update= 1;
			case XTB_TOP:
					xtb_fmt_addpos(defs[i], xspot, pad);
					break;
			case XTB_BOTTOM_J:
					last_w_update= 1;
			case XTB_BOTTOM:
					xtb_fmt_addpos(defs[i], xspot, max_height - heights[i] + pad);
					break;
			case XTB_CENTER_J:
					last_w_update= 1;
			case XTB_CENTER:
			default:
				/* Everyone else center */
					xtb_fmt_addpos(defs[i], xspot, (max_height - heights[i])/2 + pad);
					break;
		}
		_xend= _xspot+ widths[i];
		_xspot += (widths[i] + _inter);
		xspot= (int) (_xspot + 0.5);
	}
	 /* Figure out resulting size */
	if( (just== XTB_JUST || just== XTB_JUST_J) && last_width> *rw ){
	 /* Resulting width with justification is this (approx. last_width):   */
		*rw= (int) (_xend+ 0.5);
	}
	*rh = max_height + (2 * pad);
}


static void xtb_fmt_vert(nd, defs, widths, heights, just, pad, inter, rw, rh)
int nd;   			/* Number of children     */
xtb_fmt *defs[];		/* Definitions themselves */
int widths[];			/* Widths of children     */
int heights[];			/* Heights of children    */
xtb_just just;			/* Justification          */
int pad, inter;   		/* Padding and interspace */
int *rw, *rh;			/* Returned size          */
/*
 * Formats items vertically subject to the widths and heights
 * of the items passed.
 */
{
	int i;
	int max_width = 0;
	int tot_height = 0;
	int yspot;
	double _inter= inter, _yspot, _yend= 0;

	 /* Find parameters */
	for (i = 0;   i < nd;   i++) {
		if (widths[i] > max_width) max_width = widths[i];
		tot_height += heights[i];
	}
	*rh = tot_height + (nd-1)*inter + (2 * pad);
	_yspot= yspot = pad;
	if( (just== XTB_JUST || just== XTB_JUST_J) && nd> 1 && last_height> *rh ){
		  /* No idea why the last terms are not necessary (commented out) in the
		   \ vertical case...
		   */
		_inter= (last_height- (*rh- 2.0* pad))/(nd- 1.0)	/* - 2* pad- 1*/;
		_yspot= yspot= 0;
	}
	if( just== XTB_JUST_J ){
		last_h_update= 1;
	}
	 /* Place items -- assumes center justification */
	for (i = 0;   i < nd;   i++) {
		switch (just) {
			case XTB_LEFT_J:
				last_h_update= 1;
			case XTB_LEFT:
				xtb_fmt_addpos(defs[i], pad, yspot);
				break;
			case XTB_RIGHT_J:
				last_h_update= 1;
			case XTB_RIGHT:
				xtb_fmt_addpos(defs[i], max_width - widths[i] + pad, yspot);
				break;
			case XTB_CENTER_J:
				last_h_update= 1;
			case XTB_CENTER:
			default:
				/* Everyone else center */
				xtb_fmt_addpos(defs[i], (max_width - widths[i])/2 + pad, yspot);
				break;
		}
		_yend= _yspot+ heights[i];
		_yspot += (heights[i] + _inter);
		yspot= (int) (_yspot+ 0.5);
	}
	 /* Figure out resulting size */
	*rw = max_width + (2 * pad);
	if( (just== XTB_JUST || just== XTB_JUST_J) && last_height> *rh ){
		*rh= (int) (_yend+ 0.5);
	}
}

static void xtb_fmt_top(def, w, h, nm)
xtb_fmt *def;
unsigned *w, *h, *nm;
/*
 * Recursive portion of formatter
 */
{
	unsigned widths[MAX_BRANCH];
	unsigned heights[MAX_BRANCH];
	int i;
	static int level= 0;


	level+= 1;
	switch (def->type) {
		case A_TYPE:{
		  int padding= def->align.padding, interspace= def->align.interspace;
			  /* Formatting directive */
			  /* place children and determine sizes */
			_last_width= (level> 1)? w[-1] : 0;
			_last_height= (level> 1)? h[-1] : 0;
			if( last_w_update ){
				last_width= _last_width;
				last_w_update= 0;
			}
			if( last_h_update ){
				last_height= _last_height;
				last_h_update= 0;
			}

			def->align.width= def->align.height= def->align.Nmapped= 0;
			for (i = 0;   i < def->align.ni;   i++) {
			  unsigned N= 0;
				xtb_fmt_top(def->align.items[i], &(widths[i]), &(heights[i]), &N);
				def->align.width+= widths[i];
				def->align.height+= heights[i];
				def->align.Nmapped+= N;
			}
			if( !(*nm= def->align.Nmapped) ){
			  /* Ignore padding and interspace specifications when none of the specified
			   \ frames are mapped. This allows flexible rescaling as a function of the
			   \ visible widgets.
			   */
				padding= interspace= 0;
			}
			 /* restore these values for the current level:    */
			_last_width= (level> 1)? w[-1] : 0;
			_last_height= (level> 1)? h[-1] : 0;
			if( last_w_update ){
				last_width= _last_width;
				last_w_update= 0;
			}
			if( last_h_update ){
				last_height= _last_height;
				last_h_update= 0;
			}
			 /* now format based on direction */
			switch (def->align.dir) {
				case HORIZONTAL:
					xtb_fmt_hort(def->align.ni, def->align.items, widths, heights,
						def->align.just, padding, interspace, w, h);
					break;
				case VERTICAL:
					xtb_fmt_vert(def->align.ni, def->align.items, widths, heights,
						def->align.just, padding, interspace, w, h);
					break;
				default:
					FATALERROR("bad direction");
			}
			break;
		}
		case W_TYPE:
			/* Simple widget - return size */
			if( def->wid.w->mapped ){
				_last_width= *w = def->wid.w->width+ def->wid.w->border- 1;
				_last_height= *h = def->wid.w->height+ def->wid.w->border- 1;
				*nm= 1;
				if( def->wid.w->info->type== xtb_BK ){
					XLowerWindow( t_disp, def->wid.w->win );
				}
			}
			else{
				_last_width= *w = 0;
				_last_height= *h = 0;
				*nm= 0;
			}
			if( last_w_update ){
				last_width= _last_width;
				last_w_update= 0;
			}
			if( last_h_update ){
				last_height= _last_height;
				last_h_update= 0;
			}
			break;
		default:
			FATALERROR("bad type");
	}
	level-= 1;
}

xtb_fmt *xtb_fmt_do(def, w, h)
xtb_fmt *def;
unsigned *w, *h;
/*
 * Actually does formatting
 */
{ unsigned N= 0;
	/* First zero out all positions */
	xtb_fmt_setpos(def, 0, 0);

	_last_width= 0, _last_height= 0;
	last_h_update= 0, last_w_update= 0;
	last_width= 0, last_height= 0;


	/* Now call recursive portion */
	xtb_fmt_top(def, w, h, &N);
	return def;
}

void xtb_fmt_free(def)
xtb_fmt *def;
/*
 * Frees resources associated with formatting routines
 */
{
	int i;

	if (def->type == A_TYPE) {
		for (i = 0;   i < def->align.ni;   i++) {
			xtb_fmt_free(def->align.items[i]);
			def->align.items[i]= NULL;
			def->type= NO_TYPE;
		}
	}
	else if( (def->type== NO_TYPE || !def) && debugFlag ){
		fprintf( StdErr, "xtb::xtb_fmt_free(0x%lx): invalid argument\n", def );
		fflush( StdErr );
	}
	if( def ){
		xfree( def);
	}
}

void xtb_mv_frames(nf, frames)
int nf;   			/* Number of frames */
xtb_frame frames[];   	/* Array of frames  */
/*
 * Moves frames to the location indicated in the frame
 * structure for each item.
 */
{
	int i;

	for (i = 0;   i < nf;   i++) {
		if( frames[i].win || (frames[i].info && (frames[i].info->type== xtb_BK || frames[i].info->type== xtb_User)) ){
			XMoveWindow(t_disp, frames[i].win, frames[i].x_loc, frames[i].y_loc);
		}
		else{
			fprintf( StdErr, "xtb_mv_frames(%d,0x%lx): frame #%d (type %s) has no window\n",
				nf, frames, i,
				(frames[i].info)? d2str( frames[i].info->type, "%g",0) : "??"
			);
			if( frames[i].description ){
				fprintf( StdErr, "\t%s\n", frames[i].description );
			}
			fflush( StdErr);
		}
	}
}

void xtb_mv_frame(xtb_frame *frame)
{
	if( frame->win || (frame->info && (frame->info->type== xtb_BK || frame->info->type== xtb_User)) ){
		XMoveWindow(t_disp, frame->win, frame->x_loc, frame->y_loc);
	}
	else{
		fprintf( StdErr, "xtb_mv_frame(0x%lx): frame (type %s) has no window\n",
			frame, (frame->info)? d2str( frame->info->type, "%g",0) : "??"
		);
		if( frame->description ){
			fprintf( StdErr, "\t%s\n", frame->description );
		}
		fflush( StdErr);
	}
}

void xtb_select_frames_tabset(int nf, xtb_frame frames[], int tabset, int (*select)(int id, int tabset, int pattern) )
/*
 * Moves frames to the location indicated in the frame
 * structure for each item.
 */
{ int i;

	XSetInputFocus( t_disp, PointerRoot, RevertToParent, CurrentTime);
	for( i= 0; i < nf; i++ ){
		if( frames[i].win || (frames[i].info && (frames[i].info->type== xtb_BK || frames[i].info->type== xtb_User)) ){
		  int ok= False;
			if( select ){
				if( (*select)( i, frames[i].tabset, tabset) ){
					ok= True;
				}
			}
			else if( frames[i].tabset== tabset ){
				ok= True;
			}
			if( ok ){
				XMapWindow( t_disp, frames[i].win );
				frames[i].mapped= 1;
				if( frames[i].info->type== xtb_BK ){
					XLowerWindow( t_disp, frames[i].win );
				}
			}
			else{
				XUnmapWindow( t_disp, frames[i].win );
				frames[i].mapped= 0;
			}
		}
		else{
			fprintf( StdErr, "xtb_select_frames_tabset(%d,0x%lx): frame #%d (type %s) has no window\n",
				nf, frames, i,
				(frames[i].info)? d2str( frames[i].info->type, "%g",0) : "??"
			);
			if( frames[i].description ){
				fprintf( StdErr, "\t%s\n", frames[i].description );
			}
			fflush( StdErr);
		}
	}
}

int xtb_dialog_accepted, xtb_dialog_ti_accepted= 0;

/*ARGSUSED*/
static xtb_hret err_func(win, bval, info)
Window win;   		/* Button window     */
int bval;			/* Button value      */
xtb_data info;			/* Local button info */
/*
 * Handler function for button in error box.  Simply stops dialog.
 */
{
	(void) xtb_bt_set(win, 1, (xtb_data) 0);
	(void) xtb_bt_set(win, 0, (xtb_data) 0);
	xtb_dialog_accepted= xtb_dialog_ti_accepted= (int) info;
	return XTB_STOP;
}



int xtb_getline(tptr, lptr)
char **tptr;
char **lptr;
/*
 * Returns next line from tptr.  The text of tptr is changed!
 */
{
	*lptr = *tptr;
	while( *tptr && **tptr && (**tptr != '\n')) {
		(*tptr)++;
	}
	if (**tptr == '\n') {
		**tptr = '\0';
		(*tptr)++;
		return 1;
	} else {
		return( **lptr)? 1 : 0;
	}
}

#ifdef XGRAPH
	typedef struct XGFontStruct{
		XFontStruct *font;
		char name[128];
	} XGFontStruct;

	extern XGFontStruct dialogFont, titleFont;
#endif

/* Check a string's first character to see if the string should be
 \ left-aligned. A 1st ^A (0x01) is skipped, since that already has
 \ a special meaning to the to_info class (drawn in rev.).
 */
int xtb_is_leftalign( char *s )
{ char c= (*s== 0x01)? s[1] : s[0];
	return( isspace((unsigned char)c) || c== '#' || c== '*' );
}

#define	TAB			'\t'
#define	BACKSPACE	0010
#define DELETE		0177
#define	CONTROL_P	0x1b	/* actually ESC	*/
#define CONTROL_U	0025
#define CONTROL_W	0027
#define CONTROL_X	0030

xtb_hret xtb_dialog_ti_h( Window win, int ch, char *text, xtb_data val)
{
/*   char Text[MAXCHBUF];	*/
  int accept;
#ifndef XGRAPH
  char word_sep[]= ";:,./ \t-_][(){}";
#else
  extern char word_sep[];
#endif

    if( (ch == BACKSPACE) || (ch == DELETE) ){
		if( !( (ch== DELETE)? xtb_ti_dch_right(win) : xtb_ti_dch(win)) )
			Boing( 5);
		return( xtb_dialog_ti_h( win, 0, text, val) );
    }
	else if( (ch == CONTROL_U) || (ch == CONTROL_X) ){
		xtb_ti_set(win, "", (xtb_data) 0);
		return( xtb_dialog_ti_h( win, 0, text, val) );
    }
	else if( ch== CONTROL_W){
	  char *str;
		if( *text)
			str= &text[ strlen(text)-1 ];
		else{
			Boing( 5);
			return( xtb_dialog_ti_h( win, 0, text, val) );
		}
		if( index( word_sep, *str) ){
			if( !xtb_ti_dch(win) ){
				Boing( 5);
			}
			return( xtb_dialog_ti_h( win, 0, text, val) );
		}
		while( *str && !index( word_sep, *str) ){
			if( !xtb_ti_dch(win) ){
				Boing( 5);
				return( xtb_dialog_ti_h( win, 0, text, val) );
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
/* 	xtb_ti_get( win, Text, (xtb_data) NULL );	*/
	if( (accept= (ch== XK_Down || ch== XK_Up || ch== 0x12 )) ){
		xtb_dialog_ti_accepted= 1;
	}
    return XTB_HANDLED;
}

int xtb_box_ign_first_brelease= 0;

static void map_init_errwin_pos( xtb_frame *frame, err_info *info, XSizeHints *hints )
{ Window root_win, win_win;
  XEvent evt;
  int i, nx, ny, root_x, root_y, win_x, win_y, mask;

	xtb_frame_position( frame, &frame->x_loc, &frame->y_loc );
	XQueryPointer( t_disp, frame->win, &root_win, &win_win,
		&root_x, &root_y, &win_x, &win_y, &mask
	);
	mask= xtb_Mod2toMod1(mask);
	hints->x = nx= root_x - frame->width/2;
	hints->y = ny= root_y - (info->cont_ypos);
	if( hints->x< 0)
		hints->x= nx= 0;
	else if( hints->x > DisplayWidth( t_disp, t_scrn)- frame->width)
		hints->x = nx= DisplayWidth( t_disp, t_scrn)- frame->width;
	if( hints->y< 0)
		ny= hints->y= 0;
	else if( hints->y > DisplayHeight( t_disp, t_scrn)- frame->height)
		hints->y = ny= DisplayHeight( t_disp, t_scrn)- frame->height;
	hints->flags = PPosition;
	XMoveWindow(t_disp, frame->win, (frame->x_loc= nx), (frame->y_loc= ny) );
	XSetNormalHints(t_disp, frame->win, hints);
	XRaiseWindow(t_disp, frame->win);
	XMapWindow(t_disp, frame->win);
	for( i= 0; XCheckTypedWindowEvent( t_disp, frame->win, ConfigureNotify, &evt ); i++ );
	for( i= 0; XCheckWindowEvent( t_disp, frame->win, ExposureMask|StructureNotifyMask, &evt ); i++ );
}

#define ERRBOX_MORE_FRAMES	2+5

int xtb_ReparentDialogs= False;

#include <time.h>

static void make_error_box( Window parent, char *text, char *title, xtb_frame *frame, err_info **New_Info,
	Boolean OK_button, Boolean Can_button, char *default_input_text, int tilen, int maxlen,
	xtb_hret (*hlp_fnc)( Window win, int state, xtb_data val), char *hlp_label,
	xtb_hret (*hlp_fnc2)( Window win, int state, xtb_data val), char *hlp_label2,
	xtb_hret (*hlp_fnc3)( Window win, int state, xtb_data val), char *hlp_label3,
	  /* NB: make_error_box dynamically allocates the number of xtb_frames it needs, expanding when necessary
	   \ in the loop that adds the message lines to the dialog. HOWEVER, the various buttons, the title and the
	   \ (optional) errno message are added afterwards, without expansion-when-necessary. There are currently
	   \ 5 buttons that can be present, plus 2 lines of text (title + errno message). The expansion code in the
	   \ message-adding loop should take this marge into account: this is done via the ERRBOX_MORE_FRAMES macro
	   \ defined above. Thus, when extra buttons are added, this macro must be updated.
	   */
	XSizeHints *hints, int dbgFlag
)
/*
 * Makes an error box with a title.
 */
{
	Window pwin;
/* 	XSizeHints hints;	*/
	struct err_info *new_info;
    XWMHints *wmhints, WMH;
	xtb_data data;
	xtb_frame tf, lf;
	char *LinePtr, *lineptr, *line;
	int y, i;
	unsigned long wamask;
	XWindowAttributes winfo;
	XSetWindowAttributes wattr;
	int to_id= 0;
	Pixel xbp= xtb_back_pix, xnp= xtb_norm_pix, xlp= xtb_light_pix, xmp= xtb_middle_pix;
	Boolean ours= False;
#ifdef XGRAPH
	extern char *Prog_Name;
#else
	extern char *ProgramName;
#endif

	xtb_back_pix= xtb_white_pix;
	xtb_norm_pix= xtb_black_pix;
	xtb_light_pix= xtb_Lgray_pix;
	xtb_middle_pix= xtb_Mgray_pix;
	xtb_lighter_pix= xtb_LLgray_pix;

	if( !XFindContext(t_disp, parent, h_context, (caddr_t*) &data) ) {
		if( ((struct h_info *) data)->window== parent ){
			ours= True;
		}
	}

	if( xtb_ReparentDialogs &&
		( OK_button || Can_button || default_input_text || hlp_fnc || hlp_fnc2 || hlp_fnc3)
	){
		  /* "interactive" dialog that should be reparented by the Window Manager */
		ours= False;
	}

#ifndef XGRAPH
/* 	ux11_std_vismap(t_disp, &vis, &cmap, &t_screen, &depth, 0 );	*/
#endif
	if( !ours ){
		pwin= RootWindow( t_disp, t_screen );
		wamask = ux11_fill_wattr(&wattr, CWBackPixel, xtb_back_pix,
					CWBorderPixel, xtb_norm_pix,
					CWBackingStore, WhenMapped,
					CWSaveUnder, True,
					CWColormap, cmap, UX11_END);
	}
	else{
/* 		pwin= parent;	*/
		  /* 20020905: setting the parent to the actual specified parent has in some WM the effect that the dialog is
		   \ clipped by the parent. This is not what we want :(
		   \ But setting the RootWindow has in (those same) WM the effect that keyboard events are somehow completely
		   \ ignored. Unfortunately, it is not trivial to change the CWOverrideRedirect flag after a window has been
		   \ created, in order for the window manager to reparent it (and that only for windows smaller than the screen).
		   \ This could fortunately be remedied by just giving the dialog explicitly the focus... :)
		   */
		pwin= RootWindow( t_disp, t_screen );
		wamask = ux11_fill_wattr(&wattr, CWBackPixel, xtb_back_pix,
					CWBorderPixel, xtb_norm_pix,
					CWOverrideRedirect, True,
					CWBackingStore, WhenMapped,
					CWSaveUnder, True,
					CWColormap, cmap, UX11_END);
	}
	frame->win = XCreateWindow(t_disp, pwin,
				   0, 0, 1, 1, 2,
				   depth, InputOutput, vis,
				   wamask, &wattr);
	XSetWindowColormap( t_disp, frame->win, cmap );
	if( !err_cursor ){
		err_cursor= XCreateFontCursor(t_disp, XC_draft_small );
	}
	fg_color.pixel = xtb_norm_pix;
	XQueryColor(t_disp, cmap, &fg_color);
	bg_color.pixel = xtb_back_pix;
	XQueryColor(t_disp, cmap, &bg_color);
	XRecolorCursor(t_disp, err_cursor, &fg_color, &bg_color);
	XDefineCursor(t_disp, frame->win, err_cursor);

	frame->x_loc = frame->y_loc = frame->width = frame->height = 0;
#ifdef XGRAPH
	XStoreName(t_disp, frame->win, Prog_Name );
#else
	XStoreName(t_disp, frame->win, ProgramName );
#endif

	new_info = (struct err_info *) calloc( 1, (unsigned) sizeof(struct err_info));
	new_info->type= xtb_ERR;
	*New_Info= new_info;

	if( !title ){
		title= "";
	}

	new_info->alloc_lines = E_LINES;
	new_info->num_lines = 0;
	new_info->subframe = (xtb_frame *) calloc((unsigned) sizeof(xtb_frame), E_LINES+ 1);
	new_info->width = (int *) calloc((unsigned) sizeof(int), E_LINES+ 1 );
	new_info->ypos = (int *) calloc((unsigned) sizeof(int), E_LINES+ 1 );
	new_info->xcenter = (int *) calloc((unsigned) sizeof(int), E_LINES+ 1 );

	xtb_to_new(frame->win, title, XTB_TOP_LEFT, &titleFont.font, NULL, &tf);
	{ char ttext[256];
	  struct tm *tm;
	  time_t timer= time(NULL);

		tm= localtime( &timer );
		sprintf( ttext, "Title of dialog posted %s", asctime(tm) );
		xtb_describe( &tf, ttext );
	}
	new_info->title = tf.win;
	  /* tf title frame is stored in the subframe list as the last entry. */

	if (tf.width > frame->width)
		frame->width = tf.width;

	if( !text ){
		text= "";
	}

	LinePtr = lineptr= strdup(text);
	y = E_VPAD+ tf.height+ E_INTER;
	lf.height= 0;
	while( xtb_getline(&lineptr, &line) ){
	  char desc[64];
		if( strlen( line ) ){
			if( new_info->num_lines+ ERRBOX_MORE_FRAMES >= new_info->alloc_lines ){
				new_info->alloc_lines *= 2;
				new_info->subframe = (xtb_frame *) realloc((char *) new_info->subframe,
								(unsigned) sizeof(xtb_frame)* new_info->alloc_lines+ 1);
				new_info->width = (int *) realloc((char *) new_info->width,
								(unsigned) sizeof(int)* new_info->alloc_lines+ 1);
				new_info->ypos = (int *) realloc((char *) new_info->ypos,
								(unsigned) sizeof(int)* new_info->alloc_lines+ 1);
				new_info->xcenter = (int *) realloc((char *) new_info->xcenter,
								(unsigned) sizeof(int)* new_info->alloc_lines+ 1);
			}
			xtb_to_new(frame->win, line, XTB_TOP_LEFT, &dialogFont.font, NULL, &lf );
			sprintf( desc, "Line %d (id=%d)\n", new_info->num_lines, to_id );
			xtb_describe( &lf, desc );
			((struct to_info*)lf.info)->id= to_id++;
			((struct to_info*)lf.info)->text_return= &(new_info->selected_text);
			((struct to_info*)lf.info)->text_id_return= &(new_info->selected_id);
			new_info->subframe[new_info->num_lines]= lf;
			new_info->width[new_info->num_lines]= lf.width;
			new_info->ypos[new_info->num_lines]= y;
			if( lf.width > frame->width){
				frame->width = lf.width;
			}
			 /* lines starting with whitespace are drawn left-justified    */
			new_info->xcenter[new_info->num_lines]= ( xtb_is_leftalign(line) )? 0 : 1;
			if( new_info->num_lines && xtb_is_leftalign( line ) ){
				new_info->xcenter[ new_info->num_lines-1 ]= 0;
			}
			if( debugLevel==-2 ){
				fprintf( StdErr, "\tdialog line %d (%dx%d@%d) (%s): %s\n",
					new_info->num_lines,
					lf.width, lf.height, y,
					line, (new_info->xcenter[new_info->num_lines])? "centered" : "left-just"
				);
				fflush( StdErr );
			}
			new_info->num_lines += 1;
		}
		else if( new_info->num_lines ){
		 /* empty line: that means that the previous line must be
           \ shown left-justified
           */
			new_info->xcenter[ new_info->num_lines-1 ]= 0;
			if( debugLevel==-2 ){
				fprintf( StdErr, "\tdialog line %d: made left-just\n", new_info->num_lines-1);
				fflush( StdErr );
			}
		}
		y+= lf.height+ E_INTER;
	}
	if( xtb_errno ){
		xtb_to_new(frame->win, serror(), XTB_TOP_LEFT, &dialogFont.font, NULL, &lf);
		new_info->subframe[new_info->num_lines] = lf;
		new_info->width[new_info->num_lines]= lf.width;
		new_info->ypos[new_info->num_lines]= y;
		new_info->xcenter[new_info->num_lines]= 1;
		if( lf.width > frame->width){
			frame->width = lf.width;
		}
		xtb_describe( &lf, "The last generated error message\n" );
		if( debugLevel==-2 ){
			fprintf( StdErr, "\terrno (%d) line %d (%s): %s\n",
				xtb_errno, new_info->num_lines, line, (new_info->xcenter[new_info->num_lines])? "centered" : "left-just"
			);
			fflush( StdErr );
		}
		new_info->num_lines += 1;
		y+= lf.height+ E_INTER;
	}
	if( default_input_text ){
		if( tilen== 0 ){
			tilen= 50;
		}
		else if( tilen< 0 ){
			tilen= (*default_input_text)? -tilen* strlen(default_input_text) : maxlen/ 10;
			if( tilen< 50 ){
				tilen= 50;
			}
		}
		if( tilen> maxlen ){
			tilen= maxlen;
		}
		xtb_ti_new( frame->win, default_input_text, tilen, maxlen,
			xtb_dialog_ti_h, (xtb_data) 0, &new_info->subframe[new_info->num_lines]
		);
		new_info->ti_maxlen= maxlen;
		new_info->ti_win= new_info->subframe[new_info->num_lines].win;
		new_info->width[new_info->num_lines]= new_info->subframe[new_info->num_lines].width;
		y+= new_info->subframe[new_info->num_lines].height/2;
		new_info->ypos[new_info->num_lines]= y;
		new_info->xcenter[new_info->num_lines]= 1;
		if( new_info->subframe[new_info->num_lines].width > frame->width){
			frame->width = new_info->subframe[new_info->num_lines].width;
		}
		xtb_describe( &new_info->subframe[new_info->num_lines],
			" Enter the requested information here\n"
			" Press ^V to escape the following special chars:\n"
			"   Enter: close with accepting changes to default input\n"
			"   Escape: close without accepting changes to default input\n"
			"   ^?: open help/info window when corresponding button is present\n"
			"   ^V: escape..\n"
			"\n"
		);
		if( debugLevel==-2 ){
			fprintf( StdErr, "\tinput field (%s) line %d: %s\n",
				default_input_text, new_info->num_lines, (new_info->xcenter[new_info->num_lines])? "centered" : "left-just"
			);
			fflush( StdErr );
		}
		y+= new_info->subframe[new_info->num_lines].height+ 2* E_INTER;
		new_info->ti_fr= &new_info->subframe[new_info->num_lines];
		new_info->num_lines += 1;
	}


	new_info->subframes= new_info->num_lines;

	{ int last_y= y, right_w= 0;
	  int OK_w= 0, bwidth= 0;
		if( OK_button ){
		  xtb_frame *cf= &new_info->subframe[new_info->subframes];
			xtb_bt_new(frame->win, "OK", err_func, (xtb_data) 1, cf);
			xtb_describe( cf, "Click to accept\n" );
			new_info->contbtn = cf->win;
			XDefineCursor(t_disp, cf->win, bt_cursor);
			bwidth+= (OK_w= E_HPAD* 2+ cf->width);
			new_info->cont_ypos= y+ cf->height/ 2;
			last_y= y;
			y += (cf->height + E_INTER);
			new_info->subframes+= 1;
			new_info->contfr= cf;
		}
		else{
			new_info->contbtn= 0;
			new_info->contfr= NULL;
			new_info->cont_ypos= y;
			if( SetParentBackground ){
				XSetWindowBackground( t_disp, frame->win, xtb_light_pix );
			}
		}
		if( Can_button ){
		  xtb_frame *cf= &new_info->subframe[new_info->subframes];
			xtb_bt_new(frame->win, "Cancel", err_func, (xtb_data) 0, cf);
			xtb_describe( cf, "Click to cancel\n" );
			new_info->cancbtn = cf->win;
			XDefineCursor(t_disp, cf->win, bt_cursor);
			bwidth+= cf->width+ E_HPAD* 2;
			if( OK_button ){
				y= last_y;
			}
			last_y= y;
			y += (cf->height + E_INTER);
			new_info->subframes+= 1;
			new_info->cancfr= cf;
		}
		else{
			new_info->cancbtn= 0;
			new_info->cancfr= NULL;
			new_info->canc_ypos= y;
			if( SetParentBackground ){
				XSetWindowBackground( t_disp, frame->win, xtb_light_pix );
			}
		}

		if( hlp_fnc3 ){
		  xtb_frame *hf= &new_info->subframe[new_info->subframes];
			if( OK_button || Can_button ){
				y = last_y;
			}
			xtb_bt_new(frame->win, (hlp_label3)? hlp_label3 : "Surprise", hlp_fnc3, (xtb_data) new_info, hf);
			xtb_describe( hf, "Click for more help/additional information\n" );
			new_info->hlp_win3 = hf->win;
			XDefineCursor(t_disp, hf->win, bt_cursor);
			bwidth+= hf->width+ E_HPAD* 2;
			new_info->hlp_fnc3= hlp_fnc3;
			last_y= y;
			y += (hf->height + E_INTER);
			new_info->subframes+= 1;
			new_info->hlp_fr3= hf;
		}
		else{
			new_info->hlp_win3= 0;
			new_info->hlp_fr3= NULL;
		}
		if( hlp_fnc2 ){
		  xtb_frame *hf= &new_info->subframe[new_info->subframes];
			if( OK_button || Can_button || hlp_fnc3 ){
				y = last_y;
			}
			xtb_bt_new(frame->win, (hlp_label2)? hlp_label2 : "More help", hlp_fnc2, (xtb_data) new_info, hf);
			xtb_describe( hf, "Click for more help/additional information\nOr ^1 or ^! on keyboard\n" );
			new_info->hlp_win2 = hf->win;
			XDefineCursor(t_disp, hf->win, bt_cursor);
			bwidth+= hf->width+ E_HPAD* 2;
			new_info->hlp_fnc2= hlp_fnc2;
			last_y= y;
			y += (hf->height + E_INTER);
			new_info->subframes+= 1;
			new_info->hlp_fr2= hf;
		}
		else{
			new_info->hlp_win2= 0;
			new_info->hlp_fr2= NULL;
		}
		if( hlp_fnc ){
		  xtb_frame *hf= &new_info->subframe[new_info->subframes];
			if( OK_button || Can_button || hlp_fnc2 || hlp_fnc3 ){
				y = last_y;
			}
			xtb_bt_new(frame->win, (hlp_label)? hlp_label : "Help", hlp_fnc, (xtb_data) new_info, hf);
			xtb_describe( hf, "Click for help/additional information\nOr ^/ or ^? on keyboard\n" );
			new_info->hlp_win = hf->win;
			XDefineCursor(t_disp, hf->win, bt_cursor);
			bwidth+= hf->width+ E_HPAD* 2;
			new_info->hlp_fnc= hlp_fnc;
			last_y= y;
			y += (hf->height + E_INTER);
			new_info->subframes+= 1;
			new_info->hlp_fr= hf;
		}
		else{
			new_info->hlp_win= 0;
			new_info->hlp_fr= NULL;
		}
		if( bwidth > frame->width ){
			frame->width = bwidth;
		}
		if( new_info->contfr ){
	/* 		XMoveWindow(t_disp, new_info->contbtn, (int) (frame->width/2 - new_info->contfr->width/2), last_y);	*/
			XMoveWindow(t_disp, new_info->contbtn, (int) E_HPAD*2, last_y);
		}
		if( new_info->cancfr ){
			XMoveWindow(t_disp, new_info->cancbtn, (int) OK_w+ E_HPAD*2, last_y);
		}
		if( new_info->hlp_fr3 ){
			XMoveWindow(t_disp, new_info->hlp_win3, (int) (frame->width - right_w- new_info->hlp_fr3->width- E_HPAD*2), last_y);
			right_w+= new_info->hlp_fr3->width;
		}
		if( new_info->hlp_fr2 ){
			XMoveWindow(t_disp, new_info->hlp_win2, (int) (frame->width - right_w- new_info->hlp_fr2->width- E_HPAD*2), last_y);
			right_w+= new_info->hlp_fr2->width;
		}
		if( new_info->hlp_fr ){
			XMoveWindow(t_disp, new_info->hlp_win, (int) (frame->width - right_w- new_info->hlp_fr->width- E_HPAD*2), last_y);
			right_w+= new_info->hlp_fr->width;
		}
	}

	 /* Placement */
	frame->width += (2 * E_HPAD);
	 /* Title */
	XMoveWindow(t_disp, new_info->title, (int) (frame->width/2 - tf.width/2), E_VPAD );
	 /* All lines */
	for (i = 0;   i < new_info->num_lines;   i++) {
		if( new_info->xcenter[i] ){
			XMoveWindow(t_disp, new_info->subframe[i].win, (int)(frame->width/2 - new_info->width[i]/2), new_info->ypos[i] );
		}
		else{
			XMoveWindow(t_disp, new_info->subframe[i].win, E_HPAD, new_info->ypos[i] );
		}
	}


	if( new_info->title ){
		new_info->subframe[new_info->subframes]= tf;
		new_info->subframes+= 1;
	}

	  /* Restore the frame info in the subframe[]->info fields, that got surely
	   \ disturbed because of subframe[] reallocations. In addition, the X11 registry
	   \ stores the association between window and frame: this information should be
	   \ updated also.
	   */
	for( i= 0; i< new_info->subframes; i++ ){
	  int mask= XTB_UPDATE_REG_FRAME;
		new_info->subframe[i].info->frame= &(new_info->subframe[i]);
		xtb_update_registry( new_info->subframe[i].win, &(new_info->subframe[i]), NULL, 0, mask );
	}

	 /* Make dialog the right size */
	y += (E_VPAD - E_INTER);

	frame->height= y;

	CLIP( frame->height, 0, 65535 );
	CLIP( frame->width, 0, 65535 );

	hints->flags = PSize;
	hints->x= hints->y= 0;
	hints->width = frame->width;
	hints->height = (unsigned int) frame->height;

/* 	if( !(wmhints= XAllocWMHints()) ){	*/
		wmhints= &WMH;
/* 	}	*/
	wmhints->flags = InputHint | StateHint;
	if( ours ){
		wmhints->flags|= XUrgencyHint;
	}
	wmhints->input = True;
	wmhints->initial_state = NormalState;
	{
	  static XTextProperty wName, iName;
	  XClassHint *classhints;
	  char *ttitle= (title[0]==0x01)? &title[1] : title;
		if( XStringListToTextProperty( &ttitle, 1, &wName)== 0 ){
		}
		if( XStringListToTextProperty( &ttitle, 1, &iName)== 0 ){
		}
		if( (classhints= XAllocClassHint()) ){
#ifdef XGRAPH
			classhints->res_name= Prog_Name;
#else
			classhints->res_name= ProgramName;
#endif
			classhints->res_class= NULL;
		}
		XSetWMProperties( t_disp, frame->win, &wName, &iName, NULL, 0, hints, wmhints, classhints );
		if( classhints ){
			XFree( classhints );
		}
	}

	if( ours ){
		if( parent ){
			XSetTransientForHint(t_disp, parent, frame->win);
		}
		else{
/* 			XSetTransientForHint(t_disp, RootWindow(t_disp, t_screen), frame->win);	*/
		}
	}
	else{
		XSetWMProtocols( t_disp, frame->win, &xtb_wm_delete_window, 1 );
	}

	XSetNormalHints(t_disp, frame->win, hints);
	XSetWMNormalHints(t_disp, frame->win, hints);
	XResizeWindow(t_disp, frame->win, frame->width, frame->height );

	map_init_errwin_pos( frame, new_info, hints );
	XFlush( t_disp );

	  /* 200004010: check whether we got the full requested window..	*/
	{ int check= 2, count= 0;
		while( check && count< 5 ){
			XGetWindowAttributes(t_disp, frame->win, &winfo);
			if( frame->width!= winfo.width ){
				if( dbgFlag || abs(frame->width- winfo.width)> 1000 ){
					fprintf( StdErr, "\tmake_error_box(): requested frame width %d, got %d; correcting\n",
						frame->width, winfo.width
					);
				}
/* 				frame->width= winfo.width;	*/
			}
			else{
				check-= 1;
			}
			if( frame->height!= winfo.height ){
				if( dbgFlag || abs(frame->height- winfo.height)> 1000 ){
					fprintf( StdErr, "\tmake_error_box(): requested frame height %d, got %d; correcting\n",
						frame->height, winfo.height
					);
				}
				y= frame->height /* = winfo.height */;
			}
			else{
				check-= 1;
			}
			if( check ){
				XResizeWindow(t_disp, frame->win, frame->width, frame->height );
				check= 2;
				count+= 1;
			}
			  /* Even though we update the NormalHints width and height only, the
			   \ position must still remain "set"...: flags= position&size.
			   */
			hints->flags = PPosition|PSize;
			hints->width = frame->width;
			hints->height = (unsigned int) y;
			XSetNormalHints(t_disp, frame->win, hints);
			XSetWMNormalHints(t_disp, frame->win, hints);
		}
	}

	  /* It seems that it may be necessary to move the window once more.	*/
	XMoveWindow( t_disp, frame->win, hints->x, hints->y );
	XFlush(t_disp);
	frame->width += 4;
	frame->height = y + 4;
	xtb_register(frame, frame->win, (xtb_hret (*)()) 0, (xtb_registry_info*) new_info);
	XSelectInput(t_disp, frame->win, VisibilityChangeMask|StructureNotifyMask|ExposureMask|
		ButtonPressMask|ButtonReleaseMask|KeyPressMask|KeyReleaseMask|PointerMotionMask|PointerMotionHintMask|
		EnterWindowMask|LeaveWindowMask
	);
	xfree( LinePtr );

	xtb_back_pix= xbp;
	xtb_norm_pix= xnp;
	xtb_light_pix= xlp;
	xtb_middle_pix= xmp;

	if( ours && !( OK_button || Can_button || default_input_text || hlp_fnc || hlp_fnc2 || hlp_fnc3)
	){
		XSetInputFocus( t_disp, frame->win, RevertToParent, CurrentTime);
	}
}

static void del_err_box(err)
Window err;
/*
 * Deletes all components of an error
 */
{
	struct err_info *info;
	int i;

	if (xtb_unregister(err, (xtb_registry_info* *) &info)) {
		if( info ){
			for (i = 0;   i < info->subframes;   i++) {
				if( info->subframe[i].destroy ){
#if DEBUG == 2
					if( info->subframe[i].description ){
						fprintf( StdErr, "Freeing description \"%s\" of subframe %d\n", info->subframe[i].description, i );
					}
#endif
					xfree( info->subframe[i].description );
					(info->subframe[i].destroy)(info->subframe[i].win, NULL);
					info->subframe[i].win= 0;
					info->subframe[i].mapped= 0;
					xfree( info->subframe[i].framelist );
				}
			}
			xfree( info->subframe);
			xfree( info->width);
			xfree( info->ypos);
			xfree( info);
		}
		XDestroyWindow(t_disp, err);
		xtb_XSync( t_disp, False );
	}
}

int clip_int( int var, int low, int high )
{
	if( var< low ){
		var= low;
	}
	else if( var> high ){
		var= high;
	}
	return(var);
}

int xtb_frame_position( xtb_frame *frame, int *x, int *y )
{ XWindowAttributes info;
  Window dummy;
	XGetWindowAttributes( t_disp, frame->win, &info );
	XTranslateCoordinates( t_disp, frame->win, info.root, 0, 0,
		&info.x, &info.y, &dummy
	);
	frame->x_loc= *x= info.x;
	frame->y_loc= *y= info.y;
	return(1);
}


/* Post an (error) message. The dialog has 2 buttons (as of 20010520), that determine what value the
 \  function returns. When the OK button is pressed (or the Return or Enter key), the function returns
 \ 1. Else, False is returned. When a dialog is already posted, -1 is returned.
 */
int xtb_error_box( Window parent, char *mesg, char *title)
{   xtb_frame err_frame;
    Window root_win, win_win;
    Boolean spb= SetParentBackground;

	XEvent evt;
	XSizeHints hints;
	int finished = 0, nx, ny, root_x, root_y, win_x, win_y, mask, db= debugFlag, keypressed= False, buttonpressed= False;
	static int active= 0;
	struct err_info *err_info;

	if( active ){
		return( -1 );
	}
	active= 1;

	SetParentBackground= True;

	xtb_errno= errno;
	xtb_dialog_accepted= False;

	debugFlag= 0;
	if( db ){
		fprintf( StdErr, "xtb_error_box(\"%s\",\"%s\")\n",
			mesg, title
		);
		fflush( StdErr );
	}

	memset( &err_frame, 0, sizeof(xtb_frame) );
	make_error_box( parent, mesg, title, &err_frame, &err_info, True, True, NULL, 0,0,
		NULL, NULL, NULL, NULL, NULL, NULL, &hints, db);
/* 	xtb_frame_position( &err_frame, &nx, &ny );	*/
	nx= err_frame.x_loc;
	ny= err_frame.y_loc;
	XQueryPointer( t_disp, err_frame.win, &root_win, &win_win,
		&root_x, &root_y, &win_x, &win_y, &mask
	);
	mask= xtb_Mod2toMod1(mask);
	do{
		if( db && debugLevel== -2 && err_info->selected_text ){
			fprintf( StdErr, "Selected text: \"%s\", id=%d\n", err_info->selected_text, err_info->selected_id );
		}
		XNextEvent(t_disp, &evt);
		if( db && debugLevel== -2 ){
		  char *throbber[]= { "\b|", "\b/", "\b-", "\b\\", "\b|", "\b/", "\b-", "\b\\", "\b*" };
		  static short i= 0, j= 0;
			if( (i % 10) == 0 ){
				fputs( throbber[j], StdErr );
				fflush(StdErr);
				i= 0;
				j = (j+1) % 9;
			}
			else{
				i+= 1;
			}
		}
		switch( evt.type){
			case Expose:
				break;
			case ClientMessage:{
				if( err_frame.win== evt.xclient.window && evt.xclient.data.l[0]== xtb_wm_delete_window &&
					strcmp( XGetAtomName(t_disp, evt.xclient.message_type), "WM_PROTOCOLS")== 0
				){
					finished= 1;
				}
				break;
			}
			case MotionNotify:{
			 int x, y;
				XQueryPointer( t_disp, err_frame.win, &root_win, &win_win,
					&x, &y, &win_x, &win_y, &mask
				);
				mask= xtb_Mod2toMod1(mask);
				xtb_frame_position( &err_frame, &err_frame.x_loc, &err_frame.y_loc );
				nx+= (x- root_x)/ 3;
				ny+= (y- root_y)/ 3;
				nx= clip_int( nx, 0, DisplayWidth( t_disp, t_scrn)- err_frame.width );
				ny= clip_int( ny, 0, DisplayHeight( t_disp, t_scrn)- err_frame.height );
				XMoveWindow(t_disp, err_frame.win, (int) nx, (int) ny);
				XFlush( t_disp );
				root_x= x;
				root_y= y;
				break;
			}
			case LeaveNotify:
				break;
			case EnterNotify:
				XQueryPointer( t_disp, err_frame.win, &root_win, &win_win,
					&root_x, &root_y, &win_x, &win_y, &mask
				);
/* 		fprintf( StdErr, "EnterNotify: rx,ry (%d,%d)\n", root_x, root_y );	*/
				mask= xtb_Mod2toMod1(mask);
				break;
			case ButtonPress:
				buttonpressed= True;
				break;
			case ButtonRelease:
				if( xtb_dispatch(t_disp, err_frame.win, err_info->subframes, err_info->subframe, &evt) != XTB_STOP && buttonpressed ){
					if( xtb_box_ign_first_brelease> 0 ){
						xtb_box_ign_first_brelease-= 1;
						buttonpressed= False;
					}
					else{
						finished= 1;
					}
				}
				break;
			case KeyPress:
				keypressed= True;
				break;
			case KeyRelease:
				if( keypressed ){
				  char keys[1]= "";
				  KeySym keysym= 0;
				  int nbytes = XLookupString(&evt.xkey, keys, 1, (KeySym *) 0, (XComposeStatus *) 0);
					keysym= XLookupKeysym( (XKeyPressedEvent*) &evt.xkey, 0);
					xtb_modifier_state= xtb_Mod2toMod1( 0xFF & evt.xkey.state );
					if( nbytes== 0 ){
					  /* This is to acknowledge the pressing of non-printable keys. */
						nbytes= 1;
					}
					if( nbytes ){
						switch( keysym ){
							case XK_Return:
							case XK_KP_Enter:
								xtb_dialog_accepted= True;
								finished= 1;
								break;
							case XK_Escape:
								xtb_dialog_accepted= False;
								finished= 1;
								break;
							case XK_Up:
							case XK_Down:
								if( CheckMask(xtb_modifier_state, Mod1Mask) && err_frame.height> DisplayHeight(t_disp,t_scrn) ){
									xtb_frame_position( &err_frame, &err_frame.x_loc, &err_frame.y_loc );
									XMoveWindow( t_disp, err_frame.win,
										(nx= root_x= err_frame.x_loc),
										(ny= root_y= (keysym== XK_Up)?
											0 :
											(DisplayHeight(t_disp, t_scrn)- err_frame.height)
										)
									);
								}
								break;
							case XK_Right:
							case XK_Left:
								if( CheckMask(xtb_modifier_state, Mod1Mask) && err_frame.width> DisplayWidth(t_disp,t_scrn) ){
									xtb_frame_position( &err_frame, &err_frame.x_loc, &err_frame.y_loc );
									XMoveWindow( t_disp, err_frame.win,
										(nx= root_x= (keysym== XK_Left)?
											0 :
											(DisplayWidth(t_disp, t_scrn)- err_frame.width)
										),
										(ny= root_y= err_frame.y_loc)
									);
								}
								break;
							case 'h':
							case 'H':
								xtb_frame_position( &err_frame, &err_frame.x_loc, &err_frame.y_loc );
								XMoveWindow( t_disp, err_frame.win,
									(nx= root_x= (DisplayWidth(t_disp, t_scrn)- err_frame.width)/2),
									(ny= root_y= err_frame.y_loc)
								);
								break;
							case 'v':
							case 'V':
								xtb_frame_position( &err_frame, &err_frame.x_loc, &err_frame.y_loc );
								XMoveWindow( t_disp, err_frame.win,
									(nx= root_x= err_frame.x_loc),
									(ny= root_y= (DisplayHeight(t_disp, t_scrn)- err_frame.height)/2)
								);
								break;
						}
					}
					else{
						finished= 1;
					}
					  /* Fall through to default if a key wasn't yet pressed!	*/
					break;
				}
			default:
				XRaiseWindow( t_disp, err_frame.win );
				  /* 20090326: flush the ConfigureNotify event(s) that XRaiseWindow() induces for this window,
				   \ and that would get us into an event avalanch
				   */
				XNextEvent(t_disp, &evt);
				while( evt.type== ConfigureNotify && err_frame.win == evt.xany.window ){
					XNextEvent(t_disp, &evt);
				}
				XPutBackEvent(t_disp, &evt);
				break;
		}
	}
	while( !finished && xtb_dispatch(t_disp, err_frame.win, err_info->subframes, err_info->subframe, &evt) != XTB_STOP );
	del_err_box(err_frame.win);
	SetParentBackground= spb;
	active= 0;
	debugFlag= db;
	return( xtb_dialog_accepted );
}

int xtb_popup_menu( Window parent, char *mesg, char *title, char **text_return, xtb_frame **Menu_Frame )
{   xtb_frame *menu_frame;
    Window root_win, win_win;
    Boolean spb= SetParentBackground;
	xtb_hret fr_r;

	XEvent evt;
	XSizeHints hints;
	int finished = 0, button_pressed= 0, nx, ny, root_x, root_y, win_x, win_y, mask, db= debugFlag;
	static int active= 0;
	struct err_info *menu_info;
	int return_id;
	int nbytes= 0;
	char keys[2]= "";
	KeySym keysyms[2]= {0};

	if( active || !Menu_Frame ){
		return(-1);
	}
	active= 1;

	SetParentBackground= True;

	xtb_errno= 0;

	debugFlag= 0;
	if( db ){
		fprintf( StdErr, "xtb_popup_menu(\"%s\",\"%s\",..,0x%lx)\n",
			mesg, title, *Menu_Frame
		);
		fflush( StdErr );
	}

	if( *Menu_Frame ){
		menu_frame= *Menu_Frame;
		menu_info= (struct err_info*) xtb_lookup( menu_frame->win );
		map_init_errwin_pos( menu_frame, menu_info, &hints );
	}
	else{
		if( (menu_frame= (xtb_frame*) calloc( 1, sizeof(xtb_frame))) ){
			make_error_box( parent, mesg, title, menu_frame, &menu_info, False, False, NULL, 0,0,
				NULL, NULL, NULL, NULL, NULL, NULL, &hints, db);
			*Menu_Frame= menu_frame;
		}
		else{
			return(-1);
		}
	}
/* 	xtb_frame_position( menu_frame, &nx, &ny );	*/
	nx= menu_frame->x_loc;
	ny= menu_frame->y_loc;
/* 	XGetWindowAttributes( t_disp, menu_frame->win, &info );	*/
/* 	nx= info.x;	*/
/* 	ny= info.y;	*/

	XQueryPointer( t_disp, menu_frame->win, &root_win, &win_win,
		&root_x, &root_y, &win_x, &win_y, &mask
	);
	mask= xtb_Mod2toMod1(mask);
	do{
	 int x, y;
	 int delta= (menu_info->num_lines)? menu_frame->height/ menu_info->num_lines : menu_frame->height;

		if( delta< 5 ){
			if( menu_info->subframes ){
				delta= menu_info->subframe[0].height;
			}
		}
		if( delta< 5 ){
			delta= 5;
		}

		if( db && debugLevel== -2 && menu_info->selected_text ){
			fprintf( StdErr, "Selected text: \"%s\", id=%d\n", menu_info->selected_text, menu_info->selected_id );
		}
		XNextEvent(t_disp, &evt);
		switch( evt.type){
			case ClientMessage:{
				if( menu_frame->win== evt.xclient.window && evt.xclient.data.l[0]== xtb_wm_delete_window &&
					strcmp( XGetAtomName(t_disp, evt.xclient.message_type), "WM_PROTOCOLS")== 0
				){
					finished= 1;
				}
				break;
			}
			case EnterNotify:
				XQueryPointer( t_disp, menu_frame->win, &root_win, &win_win,
					&root_x, &root_y, &win_x, &win_y, &mask
				);
				mask= xtb_Mod2toMod1(mask);
				break;
			case ButtonPress:
				button_pressed+= 1;
				break;
			case ButtonRelease:
				if( button_pressed> 0 ){
					button_pressed-= 1;
					if( xtb_box_ign_first_brelease> 0 ){
						xtb_box_ign_first_brelease-= 1;
					}
					else{
						finished= 1;
					}
				}
				break;
			case KeyPress:
				nbytes = XLookupString(&evt.xkey, keys, 2,
							   (KeySym *) keysyms, (XComposeStatus *) 0);
				keysyms[0]= XLookupKeysym( (XKeyPressedEvent*) &evt.xkey, 0);
				xtb_modifier_state= xtb_Mod2toMod1( 0xFF & evt.xkey.state );
				if( nbytes== 0 ){
				  /* This is to acknowledge the pressing of non-printable keys. */
					nbytes= 1;
				}
				break;
			case KeyRelease:
				xtb_modifier_state= xtb_Mod2toMod1( 0xFF & evt.xkey.state );
				if( nbytes ){
					switch( keysyms[0] ){
						case XK_Escape:
							menu_info->selected_id= 0;
							menu_info->selected_text= NULL;
							finished= 1;
							break;
						case XK_Up:
						case XK_Down:
							if( CheckMask(xtb_modifier_state, Mod1Mask) && menu_frame->height> DisplayHeight(t_disp,t_scrn) ){
								xtb_frame_position( menu_frame, &menu_frame->x_loc, &menu_frame->y_loc );
								XMoveWindow( t_disp, menu_frame->win,
									(nx= root_x= menu_frame->x_loc),
									(ny= root_y= (keysyms[0]== XK_Up)?
										0 :
										(DisplayHeight(t_disp, t_scrn)- menu_frame->height)
									)
								);
							}
							break;
						case XK_Right:
						case XK_Left:
							if( CheckMask(xtb_modifier_state, Mod1Mask) && menu_frame->width> DisplayWidth(t_disp,t_scrn) ){
								xtb_frame_position( menu_frame, &menu_frame->x_loc, &menu_frame->y_loc );
								XMoveWindow( t_disp, menu_frame->win,
									(nx= root_x= (keysyms[0]== XK_Left)?
										0 :
										(DisplayWidth(t_disp, t_scrn)- menu_frame->width)
									),
									(ny= root_y= menu_frame->y_loc)
								);
							}
							break;
						case 'h':
						case 'H':
							xtb_frame_position( menu_frame, &menu_frame->x_loc, &menu_frame->y_loc );
							XMoveWindow( t_disp, menu_frame->win,
								(nx= root_x= (DisplayWidth(t_disp, t_scrn)- menu_frame->width)/2),
								(ny= root_y= menu_frame->y_loc)
							);
							break;
						case 'v':
						case 'V':
							xtb_frame_position( menu_frame, &menu_frame->x_loc, &menu_frame->y_loc );
							XMoveWindow( t_disp, menu_frame->win,
								(nx= root_x= menu_frame->x_loc),
								(ny= root_y= (DisplayHeight(t_disp, t_scrn)- menu_frame->height)/2)
							);
							break;
					}
				}
				break;
			case MotionNotify:{
			  /* Must fall through to the default: entry! because root_x and root_y
			   \ are updated here!!
			   */
				XQueryPointer( t_disp, menu_frame->win, &root_win, &win_win,
					&x, &y, &win_x, &win_y, &mask
				);
				mask= xtb_Mod2toMod1(mask);
				if( CheckMask( mask, Mod1Mask) ){
					xtb_frame_position( menu_frame, &menu_frame->x_loc, &menu_frame->y_loc );
					nx+= x- root_x;
					ny+= y- root_y;
					nx= clip_int( nx, - menu_frame->width+ delta, DisplayWidth( t_disp, t_scrn)- delta );
					ny= clip_int( ny, - menu_frame->height+ delta, DisplayHeight( t_disp, t_scrn)- delta );
					XMoveWindow(t_disp, menu_frame->win, (int) nx, (int) ny);
				}
				root_x= x;
				root_y= y;
			}
			default:
				if( evt.type!= MotionNotify && evt.xany.window== menu_frame->win ){
					XQueryPointer( t_disp, menu_frame->win, &root_win, &win_win,
						&x, &y, &win_x, &win_y, &mask
					);
					mask= xtb_Mod2toMod1(mask);
					if( y< delta && hints.y< 2* delta ){
						ny+= delta;
						nx= clip_int( (int) nx, - menu_frame->width+ delta, DisplayWidth( t_disp, t_scrn)- delta );
						ny= clip_int( (int) ny, - menu_frame->height+ delta, DisplayHeight( t_disp, t_scrn)- delta );
						XMoveWindow(t_disp, menu_frame->win, (int) nx, (int) ny);
					}
					else if( y> DisplayHeight(t_disp, t_screen)- delta &&
						hints.y+ menu_frame->height> DisplayHeight(t_disp, t_screen)- 2* delta
					){
						ny-= delta;
						nx= clip_int( (int) nx, - menu_frame->width+ delta, DisplayWidth( t_disp, t_scrn)- delta );
						ny= clip_int( (int) ny, - menu_frame->height+ delta, DisplayHeight( t_disp, t_scrn)- delta );
						XMoveWindow(t_disp, menu_frame->win, (int) nx, (int) ny);
					}
				}
				break;
		}
		fr_r= xtb_dispatch(t_disp, menu_frame->win, menu_info->subframes, menu_info->subframe, &evt);
	}
	while( fr_r != XTB_STOP && !finished );
	if( text_return && menu_info->selected_text ){
		*text_return= strdup( menu_info->selected_text );
	}
	return_id= menu_info->selected_id;
	SetParentBackground= spb;
	active= 0;
	debugFlag= db;
	XUnmapWindow( t_disp, menu_frame->win );
	xtb_XSync( t_disp, False );
	return( return_id );
}

char *xtb_popup_get_itemtext( xtb_frame *menu, int id )
{  err_info *minfo;
	if( menu && (minfo= (err_info*) menu->info) && id>= 0 && id< minfo->num_lines ){
		return( ((to_info*) minfo->subframe[id].info)->text );
	}
	else{
		return(NULL);
	}
}

void xtb_popup_delete( xtb_frame **frame )
{ xtb_frame *menu_frame;
	if( frame && *frame ){
		menu_frame= *frame;
		del_err_box(menu_frame->win);
		xfree( menu_frame );
		*frame= NULL;
	}
}

#ifdef XGRAPH
	extern char *parse_codes( char *);
	char (*xtb_parse_codes)( char *txt )= parse_codes;
#else
	char (*xtb_parse_codes)( char *txt )= NULL;
#endif

xtb_hret xtb_input_dialog_parse_codes(Window win, int bval, xtb_data info)
{ err_info *dialog= (err_info*) info;
	if( dialog ){
	  struct ti_info *info;
		xtb_bt_set( win, 1, 0 );
		info= (struct ti_info *) xtb_lookup(dialog->ti_win);
		if( xtb_parse_codes ){
			(*xtb_parse_codes)( info->text );
		}
		xtb_ti_set( dialog->ti_win, info->text, 0 );
		xtb_bt_set( win, 0, 0 );
	}
	return( XTB_HANDLED );
}

Window xtb_input_dialog_inputfield= 0;
/* 20021031: a global pointer to contain an expanded edited buffer that is used only when necessary. */
char *xtb_input_edited_buffer= NULL;

static int xid_active= 0;

char *xtb_input_dialog( Window parent, char *text, int tilen, int maxlen, char *mesg, char *title,
	int modal,
	char *hlp_label, xtb_hret (*hlp_btn)(Window,int,xtb_data),
	char *hlp_label2, xtb_hret (*hlp_btn2)(Window,int,xtb_data),
	char *hlp_label3, xtb_hret (*hlp_btn3)(Window,int,xtb_data)
)
{ Window root_win, win_win;
  Boolean spb= SetParentBackground;
/*   XWindowAttributes info;	*/
  XEvent evt;
  XSizeHints hints;
  int finished = 0, nx, ny, root_x, root_y, win_x, win_y, mask, db= debugFlag, Nevt= 0;
  xtb_frame dialog_frame;
  struct err_info *dialog_info= NULL;
  int nbytes= 0;
  char keys[1]= "";
  KeySym keysyms[1]= {0};
  xtb_hret fr_r= XTB_NOTDEF;
  Window xidi= xtb_input_dialog_inputfield;

	if( xid_active ){
		return(NULL);
	}
	xid_active= 1;

	SetParentBackground= True;

	xtb_errno= 0;

	debugFlag= 0;
	if( db ){
		fprintf( StdErr, "xtb_input_dialog(\"%s\",\"%s\",\"%s\")\n",
			text, mesg, title
		);
		fflush( StdErr );
	}

	xtb_dialog_ti_accepted= 0;
	memset( &dialog_frame, 0, sizeof(xtb_frame) );
	make_error_box( parent, mesg, title, &dialog_frame, &dialog_info, True, True, text, tilen, maxlen,
		hlp_btn, hlp_label, hlp_btn2, hlp_label2, hlp_btn3, hlp_label3, &hints, db);

/* 	xtb_frame_position( &dialog_frame, &nx, &ny );	*/
	nx= dialog_frame.x_loc;
	ny= dialog_frame.y_loc;
	XQueryPointer( t_disp, dialog_frame.win, &root_win, &win_win,
		&root_x, &root_y, &win_x, &win_y, &mask
	);
	mask= xtb_Mod2toMod1(mask);
	xtb_input_dialog_inputfield= dialog_info->ti_win;
	  /* 20021031: set the result buffer to NULL! */
	xtb_input_edited_buffer= NULL;
	do{
	 int x, y;
	 int delta= (dialog_info->num_lines)? dialog_frame.height/ dialog_info->num_lines : dialog_frame.height;
	 int handled= 0;
	 static Boolean escaped= 0;

		if( delta< 5 ){
			if( dialog_info->subframes ){
				delta= dialog_info->subframe[0].height;
			}
		}
		if( delta< 5 ){
			delta= 5;
		}
		XNextEvent(t_disp, &evt);
		Nevt+= 1;
		switch( evt.type){
			case ClientMessage:{
				if( dialog_frame.win== evt.xclient.window && evt.xclient.data.l[0]== xtb_wm_delete_window &&
					strcmp( XGetAtomName(t_disp, evt.xclient.message_type), "WM_PROTOCOLS")== 0
				){
					finished= 1;
				}
				break;
			}
			case EnterNotify:
				XQueryPointer( t_disp, dialog_frame.win, &root_win, &win_win,
					&root_x, &root_y, &win_x, &win_y, &mask
				);
				mask= xtb_Mod2toMod1(mask);
				break;
			case KeyPress:
				nbytes = XLookupString(&evt.xkey, keys, 1,
							   (KeySym *) 0, (XComposeStatus *) 0);
				keysyms[0]= XLookupKeysym( (XKeyPressedEvent*) &evt.xkey, 0);
				xtb_modifier_state= xtb_Mod2toMod1( 0xFF & evt.xkey.state );
				if( nbytes== 0 ){
				  /* This is to acknowledge the pressing of non-printable keys. */
					nbytes= 1;
					keys[0]= '\0';
				}
				if( nbytes ){
					switch( keysyms[0] ){
						case XK_KP_Enter:
						case XK_Return:
							if( !escaped ){
								xtb_dialog_ti_accepted= 1;
								finished= 1;
								handled= 1;
							}
							break;
						case XK_Escape:
							if( !escaped ){
								xtb_dialog_ti_accepted= 0;
								finished= 1;
								handled= 1;
							}
							break;
						case XK_Tab:
							XSetInputFocus( t_disp, dialog_info->ti_win, RevertToParent, CurrentTime);
							if( evt.xany.window== dialog_frame.win ){
								handled= 1;
							}
							break;
						case XK_exclam: if( CheckMask( xtb_modifier_state, ControlMask) ){
						  /* ^!	*/
							if( !escaped ){
								if( dialog_info->hlp_fnc2 ){
									(*dialog_info->hlp_fnc2)( dialog_info->hlp_win2, 1, dialog_info);
								}
								handled= 1;
							}
							if( keys[0]== '!' || keys[0]== '1' ){
								keys[0]= '\0';
							}
							break;
						}
						case XK_question: if( CheckMask( xtb_modifier_state, ControlMask) ){
						  /* ^?	*/
							if( !escaped ){
								if( dialog_info->hlp_fnc ){
									(*dialog_info->hlp_fnc)( dialog_info->hlp_win, 1, dialog_info);
								}
								handled= 1;
							}
							if( keys[0]== 0x1f ){
								keys[0]= '\0';
							}
							break;
						}
						case XK_Up:
						case XK_Down:
							if( CheckMask(xtb_modifier_state, Mod1Mask) && dialog_frame.height> DisplayHeight(t_disp,t_scrn) ){
								xtb_frame_position( &dialog_frame, &dialog_frame.x_loc, &dialog_frame.y_loc );
								XMoveWindow( t_disp, dialog_frame.win,
									(nx= root_x= dialog_frame.x_loc),
									(ny= root_y= (keysyms[0]== XK_Up)?
										0 :
										(DisplayHeight(t_disp, t_scrn)- dialog_frame.height)
									)
								);
							}
							break;
						case XK_Right:
						case XK_Left:
							if( CheckMask(xtb_modifier_state, Mod1Mask) && dialog_frame.width> DisplayWidth(t_disp,t_scrn) ){
								xtb_frame_position( &dialog_frame, &dialog_frame.x_loc, &dialog_frame.y_loc );
								XMoveWindow( t_disp, dialog_frame.win,
									(nx= root_x= (keysyms[0]== XK_Left)?
										0 :
										(DisplayWidth(t_disp, t_scrn)- dialog_frame.width)
									),
									(ny= root_y= dialog_frame.y_loc)
								);
							}
							break;
						case 'h':
						case 'H':
							if( evt.xany.window!= dialog_info->ti_fr->win ){
								xtb_frame_position( &dialog_frame, &dialog_frame.x_loc, &dialog_frame.y_loc );
								XMoveWindow( t_disp, dialog_frame.win,
									(nx= root_x= (DisplayWidth(t_disp, t_scrn)- dialog_frame.width)/2),
									(ny= root_y= dialog_frame.y_loc)
								);
								keys[0]= '\0';
								handled= 1;
							}
							break;
						case 'v':
						case 'V':
							if( CheckMask( xtb_modifier_state, ControlMask) ){
							  /* ^V	*/
								if( (escaped= !escaped) ){
									handled= 1;
								}
								if( keys[0]== 0x16 ){
									keys[0]= '\0';
								}
								break;
							}
							else if( evt.xany.window!= dialog_info->ti_fr->win ){
								xtb_frame_position( &dialog_frame, &dialog_frame.x_loc, &dialog_frame.y_loc );
								XMoveWindow( t_disp, dialog_frame.win,
									(nx= root_x= dialog_frame.x_loc),
									(ny= root_y= (DisplayHeight(t_disp, t_scrn)- dialog_frame.height)/2)
								);
								keys[0]= '\0';
								handled= 1;
							}
							break;
					}
					switch( keys[0] ){
						case 0x1b:
							if( !escaped ){
								xtb_dialog_ti_accepted= 0;
								finished= 1;
								handled= 1;
							}
							break;
						case 0x1f:
						  /* ^?	*/
							if( !escaped ){
								if( dialog_info->hlp_fnc ){
									(*dialog_info->hlp_fnc)( dialog_info->hlp_win, 1, dialog_info);
								}
								handled= 1;
							}
							break;
						case 0x16:
						  /* ^V	*/
							if( (escaped= !escaped) ){
								handled= 1;
							}
							break;
					}
					if( keys[0]!= 0x16 ){
						escaped= False;
					}
				}
				break;
			case MotionNotify:{
			  /* Must fall through to the default: entry!	*/
				XQueryPointer( t_disp, dialog_frame.win, &root_win, &win_win,
					&x, &y, &win_x, &win_y, &mask
				);
				mask= xtb_Mod2toMod1(mask);
				if( CheckMask( mask, Mod1Mask) ){
					xtb_frame_position( &dialog_frame, &dialog_frame.x_loc, &dialog_frame.y_loc );
					nx+= x- root_x;
					ny+= y- root_y;
					nx= clip_int( nx, - dialog_frame.width+ delta, DisplayWidth( t_disp, t_scrn)- delta );
					ny= clip_int( ny, - dialog_frame.height+ delta, DisplayHeight( t_disp, t_scrn)- delta );
					XMoveWindow(t_disp, dialog_frame.win, (int) nx, (int) ny);
				}
				root_x= x;
				root_y= y;
			}
			default:
				if( evt.type!= MotionNotify && evt.xany.window== dialog_frame.win ){
					XQueryPointer( t_disp, dialog_frame.win, &root_win, &win_win,
						&x, &y, &win_x, &win_y, &mask
					);
					mask= xtb_Mod2toMod1(mask);
					if( y< delta && hints.y< 2* delta ){
						ny+= delta;
						nx= clip_int( (int) nx, - dialog_frame.width+ delta, DisplayWidth( t_disp, t_scrn)- delta );
						ny= clip_int( (int) ny, - dialog_frame.height+ delta, DisplayHeight( t_disp, t_scrn)- delta );
						XMoveWindow(t_disp, dialog_frame.win, (int) nx, (int) ny);
					}
					else if( y> DisplayHeight(t_disp, t_screen)- delta &&
						hints.y+ dialog_frame.height> DisplayHeight(t_disp, t_screen)- 2* delta
					){
						ny-= delta;
						nx= clip_int( (int) nx, - dialog_frame.width+ delta, DisplayWidth( t_disp, t_scrn)- delta );
						ny= clip_int( (int) ny, - dialog_frame.height+ delta, DisplayHeight( t_disp, t_scrn)- delta );
						XMoveWindow(t_disp, dialog_frame.win, (int) nx, (int) ny);
					}
				}
				break;
		}
		if( !handled ){
		  int xdm= xtb_dispatch_modal;
			xtb_dispatch_modal= modal;
			fr_r= xtb_dispatch(t_disp, dialog_frame.win, dialog_info->subframes, dialog_info->subframe, &evt);
			xtb_dispatch_modal= xdm;
			if( dialog_info->ti_fr->mapped && evt.xany.window== dialog_info->ti_fr->win ){
			  XWindowAttributes att;
			  static char done= 0;
				XGetWindowAttributes( t_disp, dialog_info->ti_fr->win, &att );
				if( att.map_state== IsViewable && !done ){
					XSetInputFocus( t_disp, dialog_info->ti_fr->win, RevertToPointerRoot, CurrentTime);
					done= 1;
				}
			}
		}
	}
	while( !finished && fr_r!= XTB_STOP && !xtb_dialog_ti_accepted);
	SetParentBackground= spb;
	xid_active= 0;
	debugFlag= db;

	if( xtb_dialog_ti_accepted ){
	  int newlen= xtb_ti_length(dialog_info->ti_win,0)+ 1;
		if( newlen<= maxlen ){
			xtb_ti_get( dialog_info->ti_win, text, NULL);
		}
		else if( (xtb_input_edited_buffer= (char*) calloc(newlen, sizeof(char))) ){
			xtb_ti_get( dialog_info->ti_win, xtb_input_edited_buffer, NULL);
			strncpy( text, xtb_input_edited_buffer, maxlen );
			text[maxlen-1]= '\0';
			  /* return the pointer to our new buffer! */
			text= xtb_input_edited_buffer;
		}
		else{
			fprintf( StdErr, "xtb_input_dialog(,\"%s\",\"%s\"): can't get memory to store expanded result; discarding edit! (%s)\n",
				mesg, title, serror()
			);
			fflush( StdErr );
			text= NULL;
		}
	}
	else{
		text= NULL;
	}

	del_err_box(dialog_frame.win);
	xtb_input_dialog_inputfield= xidi;

	return( text );
}

/* 20040830 */
char *xtb_input_dialog_r( Window parent, char *text, int tilen, int maxlen, char *mesg, char *title,
	char *hlp_label, xtb_hret (*hlp_btn)(Window,int,xtb_data),
	char *hlp_label2, xtb_hret (*hlp_btn2)(Window,int,xtb_data),
	char *hlp_label3, xtb_hret (*hlp_btn3)(Window,int,xtb_data)
)
{ char *xieb= xtb_input_edited_buffer;
  char xida= xid_active, *Text;
  int xdta= xtb_dialog_ti_accepted;
	xid_active= 0;
	Text= xtb_input_dialog( parent, text, tilen, maxlen, mesg, title, True,
		hlp_label, hlp_btn, hlp_label2, hlp_btn2, hlp_label3, hlp_btn3
	);
	xid_active= xida;
	xtb_input_edited_buffer= xieb;
	xtb_dialog_ti_accepted= xdta;
	return(Text);
}

char *xtb_describe( xtb_frame *frame, char *description)
{
	if( frame && description ){
		if( debugFlag && debugLevel== -2 ){
			fprintf( StdErr, "xtb_describe(0x%lx,\"%s\")\n", frame, description);
			fflush( StdErr );
		}
		if( frame->info->type== xtb_TI2 ){
			xtb_describe( xtb_ti2_ti_frame(frame->win), description );
		}
		return( (frame->description= strdup( description )) );
	}
	return( NULL );
}

char *xtb_describe_s( xtb_frame *frame, char *description, int tabset)
{
	if( frame && description ){
		if( debugFlag && debugLevel== -2 ){
			fprintf( StdErr, "xtb_describe(0x%lx,\"%s\")\n", frame, description);
			fflush( StdErr );
		}
		frame->tabset= tabset;
		if( frame->info->type== xtb_TI2 ){
			xtb_describe_s( xtb_ti2_ti_frame(frame->win), description, tabset );
		}
		return( (frame->description= strdup( description )) );
	}
	return( NULL );
}

xtb_frame *xtb_find_next_named_button( xtb_frame *framelist, int N,
	xtb_frame **current, int *currentN, xtb_frame **subcurrent, int *subcurrentN,
	char *pattern, int firstchar
){ Boolean found= 0;
   int tried= 0;
	if( !framelist || !current ){
		return(NULL);
	}
	if( !*current ){
		*current= framelist;
		*currentN= 0;
	}
	else if( !subcurrent || !(*subcurrent) ){
		(*current)++;
		(*currentN)+= 1;
	}
	while( (*current) && !found && tried< N ){
		if( *currentN>= N ){
			(*current)= framelist;
			*currentN= 0;
		}
		else{
		  b_info *info= (b_info*) (*current)->info;
			if( info && info->frame && info->frame->win== (*current)->win ){
				if( info->type== xtb_BT ){
					if( info->frame->mapped ){
						if( !pattern ){
							found= (info->text[0]== firstchar);
						}
						else{
							found= (!strstr( info->text, pattern));
						}
					}
				}
				else if( info->type== xtb_BR && subcurrent && subcurrentN ){
				  br_info *r_info= (br_info*) (*current)->info;
				  __ALLOCA( frlist, xtb_frame, r_info->btn_cnt, frlist_len);
				  int i, cN= *subcurrentN;
					for( i= 0; i< r_info->btn_cnt; i++ ){
						frlist[i]= *((*current)->framelist[i]);
					}
					if( xtb_find_next_named_button( frlist, r_info->btn_cnt,
							subcurrent, subcurrentN, NULL, NULL, pattern, firstchar ) &&
							*subcurrentN> cN
					){
						GCA();
						return( (*subcurrent) );
					}
					else{
						*subcurrent= NULL;
						*subcurrentN= -1;
					}
					GCA();
				}
			}
			if( !found ){
				(*current)++;
				(*currentN)+= 1;
			}
			tried+= 1;
		}
	}
	if( !found ){
		(*current)= NULL;
		*currentN= 0;
	}
	return( (*current) );
}

  /* xtb_CharArray() takes a list of pairs <elements>,<text> and creates (allocates) an
   \ array of strings out of it. Of this array (the function's return value), the 1st
   \ element (array[0]) points to a text buffer containing all strings concatenated;
   \ each following element points to the start in that buffer of the next text string
   \ passed in the arguments. The first argument should be a pointer to an integer that will
   \ return the number of entries in the returned array. If the 2nd argument is True, each
   \ text string will be terminated, such that each array[i] element represents exactly 1
   \ of the specified strings. When this argument is False, the strings are not terminated;
   \ thence, array[0] points to a full concatenation of all the specified strings, array[1]
   \ to the last N-1, and so forth.
   \ When elements is positive, <text> is supposed to be of type char**; an array of strings.
   \ When elements is negative, <text> is supposed to be a single string, a char*.
   \ Strings receive a terminating newline if they don't have one yet.
   */
char **xtb_CharArray( int *N, int terminate, VA_DCL )
{ va_list ap;
  int i, n= 0, len, tlen= 0, elems, ok= 1;
  char *string, **array, *buf= NULL, **result;
	va_start(ap, terminate);
/* 	N= (int*) va_arg(ap, int*);	*/
/* 	terminate= (int) va_arg(ap, int*);	*/
	while( (elems= (int) va_arg(ap, int)) && ok ){
		if( elems< 0 ){
			if( (string= (char*) va_arg(ap, char*)) ){
				len= strlen(string);
				tlen+= len+ 1;
				if( string[len-1]!= '\n' ){
					tlen+= 1;
				}
				n+= 1;
			}
			else{
				ok= False;
			}
		}
		else{
			if( (array= (char**) va_arg(ap, char**)) ){
				for( i= 0; i< elems; i++ ){
					if( array[i] ){
						len= strlen(array[i]);
						tlen+= len+ 1;
						if( array[i][len-1]!= '\n' ){
							tlen+= 1;
						}
						n+= 1;
					}
				}
			}
			else{
				ok= False;
			}
		}
	}
	va_end(ap);
	if( n && tlen ){
		if( (result= (char**) calloc( n, sizeof(char*) ) ) ){
			if( (buf= (char*) calloc( tlen, sizeof(char) ) ) ){
			  int idx= 0;
				va_start(ap, terminate);
/* 				N= (int*) va_arg(ap, int*);	*/
				*N= n;
/* 				terminate= (int) va_arg(ap, int*);	*/
				ok= True;
				tlen= 0;
				while( (elems= (int) va_arg(ap, int)) && ok ){
					if( elems< 0 ){
						if( (string= (char*) va_arg(ap, char*)) ){
						  int l= strlen(string);
							result[idx]= &buf[tlen];
							if( terminate ){
								strcpy( result[idx], string );
								tlen+= l+ 1;
								if( string[l-1]!= '\n' ){
									strcat( result[idx], "\n" );
									tlen+= 1;
								}
							}
							else{
								memcpy( result[idx], string, l* sizeof(char) );
								tlen+= l;
								if( string[l-1]!= '\n' ){
									result[idx][l]= '\n';
									tlen+= 1;
								}
							}
							idx++;
						}
						else{
							ok= False;
						}
					}
					else{
						if( (array= (char**) va_arg(ap, char**)) ){
							for( i= 0; i< elems; i++ ){
								if( array[i] ){
								  int l= strlen(array[i]);
									result[idx]= &buf[tlen];
									if( terminate ){
										strcpy( result[idx], array[i] );
										tlen+= l+ 1;
										if( array[i][l-1]!= '\n' ){
											strcat( result[idx], "\n" );
											tlen+= 1;
										}
									}
									else{
										memcpy( result[idx], array[i], l* sizeof(char) );
										tlen+= l;
										if( array[i][l-1]!= '\n' ){
											result[idx][l]= '\n';
											tlen+= 1;
										}
									}
									idx++;
								}
							}
						}
						else{
							ok= False;
						}
					}
				}
				va_end(ap);
			}
			else{
				fprintf( StdErr, "xtb_CharArray(): can't allocate a buffer of %d bytes (%s)\n", tlen, serror() );
				xfree( result );
			}
		}
		else{
			fprintf( StdErr, "xtb_CharArray(): can't allocate an array of %d strings (%s)\n", n, serror() );
		}
	}
	va_end(ap);
	return(result);
}

int xtb_ParseGeometry( char *geoSpec, XSizeHints *sizehints, Window win, int update )
{ int geo_mask;
  long dum;

	if( win ){
	  /* 20060926: we (may) get sizehints uninitialised. Get the current values,
	   \ but only if win!=0 (i.e. when we need them)
	   */
		XGetWMNormalHints( t_disp, win, sizehints, &dum );
	}
	geo_mask = XParseGeometry(geoSpec, &sizehints->x, &sizehints->y,
				  &sizehints->width, &sizehints->height);
	if( (geo_mask & XValue) || (geo_mask & YValue) ) {
		if( (geo_mask & XNegative) ){
			sizehints->x= DisplayWidth(t_disp,t_screen)- sizehints->width+ sizehints->x;
		}
		if( (geo_mask & YNegative) ){
			sizehints->y= DisplayHeight(t_disp,t_screen)- sizehints->height+ sizehints->y;
		}
		sizehints->flags = (sizehints->flags & ~PPosition) | USPosition;
	}
	if( (geo_mask & WidthValue) || (geo_mask & HeightValue) ){
		sizehints->flags = (sizehints->flags & ~PSize) | USSize;
	}
	if( update && win ){
		XSetNormalHints(t_disp, win, sizehints);
		  /* 20060926: update the WMSizeHints... */
		XSetWMNormalHints(t_disp, win, sizehints);
		if( (geo_mask & XValue) || (geo_mask & YValue) ){
			XMoveWindow( t_disp, win, sizehints->x, sizehints->y );
		}
		if( (geo_mask & WidthValue) || (geo_mask & HeightValue) ){
			XResizeWindow( t_disp, win, sizehints->width, sizehints->height );
		}
	}
	return( geo_mask );
}
