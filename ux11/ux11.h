/*
 * X11 utility functions
 *
 * <stdio.h> must be included before this
 */

#ifndef X11_UTIL_HEADER
#define X11_UTIL_HEADER

#ifdef __STDC__
#	define VOID_P	void *
#else
#	define VOID_P	char *
#endif

#ifdef __STDC__
#	include <stdlib.h>
#	include <stddef.h>
#endif

#ifdef __APPLE_CC__
#	undef pixel
#endif

#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/Intrinsic.h>
#include <X11/extensions/Xdbe.h>
#ifdef XGRAPH
#	include "../va_dcl.h"
#endif

#ifdef __STDC__
#define DECLARE(func, rtn, args)	extern rtn func args
#else
#define DECLARE(func, rtn, args)	extern rtn func ()
#endif

#define UX11_END	0L

#define xg_abort()	{fputs("abort\n",stderr);exit(-10);}
#ifdef XGRAPH
#	ifndef DEBUG
#		define abort()	xg_abort()
#	endif
#endif

DECLARE( event_name, char*, (int type));
DECLARE( req_str, char*, (int req_code));

DECLARE(ux11_open_display, Display *, (int argc, char *argv[], int *display_specified ));

DECLARE( ux11_multihead_DisplayWidth, int, (Display *d, int screen, int cntr_x, int cntr_y, int *base_x, int *base_y, int *head));
DECLARE( ux11_multihead_DisplayHeight, int, (Display *d, int screen, int cntr_x, int cntr_y, int *base_x, int *base_y, int *head));

DECLARE(ux11_fill_wattr, unsigned long, ( XSetWindowAttributes *wattr, VA_DCL ));

DECLARE(ux11_fill_gcvals, unsigned long, (XGCValues *, VA_DCL));

DECLARE(ux11_fill_hints, unsigned long, ( XWMHints *, VA_DCL));

DECLARE(ux11_fill_xa, int, (Arg *arg_list, int size, VA_DCL));

DECLARE(ux11_find_visual, int, (Display *disp, int (*good_func)(),
				XVisualInfo *rtn_vis));

DECLARE(ux11_color_vis, int, (XVisualInfo *vis));

#define UX11_DEFAULT	1
#define UX11_ALTERNATE	2
#define UX11_ALTERNATE_RW	3

extern int *__ux11_min_depth(), *__ux11_vis_class(), *__ux11_useDBE();
#define ux11_min_depth	(*(__ux11_min_depth()))
#define ux11_vis_class	(*(__ux11_vis_class()))
#define ux11_useDBE		(*(__ux11_useDBE()))

DECLARE(ux11_std_vismap, int, (Display *disp, Visual **rtn_vis,
			       Colormap *rtn_cmap, int *rtn_scrn,
			       int *rtn_depth, int search_visuals ));

DECLARE(ux11_get_value, char *, (int argc, char *argv[], char *value, char *def));

DECLARE(ux11_font_microm, long, (Display *disp, int scrn_num, XFontStruct *font));

DECLARE(ux11_find_font, long, (Display *disp, int scrn_num, char *pat,
			      long (*good_func)(), VOID_P data,
			      XFontStruct **rtn_font, char **rtn_name));

DECLARE(ux11_size_font, int, (Display *disp, int scrn_num, long size,
			      XFontStruct **rtn_font, char **rtn_name, int bold));

DECLARE(ux11_error, char *, (XErrorEvent *evt));

#endif
