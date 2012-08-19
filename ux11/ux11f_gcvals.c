/*
 * X11 Utility Functions
 */

#include <stdio.h>
#include "ux11.h"
#include "ux11_internal.h"

/* 960806 RJB
 \ HPUX cc seems not to like mixing arguments, and varargs. Giving
 \ first some argument(s), and then a variable list works in gcc and
 \ other ANSI compilers :( Therefore, I commented out those arguments,
 \ and changed the __STDC__ conditional code into HPUX8__STDC__ ...
 \ Not the most elegant solution (prototypes...), but it works.
 */
#ifdef CTAGS
unsigned long ux11_fill_gcvalls( VA_DCL)
#endif
VARARGS(ux11_fill_gcvals, unsigned long, ( XGCValues *gcvals, VA_DCL))
/*
 * ux11_fill_gcvals(gcvals, name, value, VA_DCL ... , UX11_END);
 * XGCValues *gcvals;
 * Sets the values of an XGCValues structure using variable
 * argument lists.  The returned value is the the value_mask
 * indicating what field is set.  The fields available
 * are those described for the value mask (e.g. GCFunction, etc).
 */
{
    va_list ap;
    unsigned long value_mask = 0;
    unsigned long field;
#ifdef HPUX8__STDC__


	va_start(ap);
#else

    va_start(ap, gcvals);
#endif

    while ((field = va_arg(ap, unsigned long)) != UX11_END) {
	if (field == GCFunction) {
	    gcvals->function = va_arg(ap, int);
	    value_mask |= GCFunction;
	} else if (field == GCPlaneMask) {
	    gcvals->plane_mask = va_arg(ap, unsigned long);
	    value_mask |= GCPlaneMask;
	} else if (field == GCForeground) {
	    gcvals->foreground = va_arg(ap, unsigned long);
	    value_mask |= GCForeground;
	} else if (field == GCBackground) {
	    gcvals->background = va_arg(ap, unsigned long);
	    value_mask |= GCBackground;
	} else if (field == GCLineWidth) {
	    gcvals->line_width = va_arg(ap, int);
	    value_mask |= GCLineWidth;
	} else if (field == GCLineStyle) {
	    gcvals->line_style = va_arg(ap, int);
	    value_mask |= GCLineStyle;
	} else if (field == GCCapStyle) {
	    gcvals->cap_style = va_arg(ap, int);
	    value_mask |= GCCapStyle;
	} else if (field == GCJoinStyle) {
	    gcvals->join_style = va_arg(ap, int);
	    value_mask |= GCJoinStyle;
	} else if (field == GCFillStyle) {
	    gcvals->fill_style = va_arg(ap, int);
	    value_mask |= GCFillStyle;
	} else if (field == GCFillRule) {
	    gcvals->fill_rule = va_arg(ap, int);
	    value_mask |= GCFillRule;
	} else if (field == GCTile) {
	    gcvals->tile = va_arg(ap, Pixmap);
	    value_mask |= GCTile;
	} else if (field == GCStipple) {
	    gcvals->stipple = va_arg(ap, Pixmap);
	    value_mask |= GCStipple;
	} else if (field == GCTileStipXOrigin) {
	    gcvals->ts_x_origin = va_arg(ap, int);
	    value_mask |= GCTileStipXOrigin;
	} else if (field == GCTileStipYOrigin) {
	    gcvals->ts_y_origin = va_arg(ap, int);
	    value_mask |= GCTileStipYOrigin;
	} else if (field == GCFont) {
	    gcvals->font = va_arg(ap, Font);
	    value_mask |= GCFont;
	} else if (field == GCSubwindowMode) {
	    gcvals->subwindow_mode = va_arg(ap, int);
	    value_mask |= GCSubwindowMode;
	} else if (field == GCGraphicsExposures) {
	    gcvals->graphics_exposures = va_arg(ap, Bool);
	    value_mask |= GCGraphicsExposures;
	} else if (field == GCClipXOrigin) {
	    gcvals->clip_x_origin = va_arg(ap, int);
	    value_mask |= GCClipXOrigin;
	} else if (field == GCClipYOrigin) {
	    gcvals->clip_y_origin = va_arg(ap, int);
	    value_mask |= GCClipYOrigin;
	} else if (field == GCClipMask) {
	    gcvals->clip_mask = va_arg(ap, Pixmap);
	    value_mask |= GCClipMask;
	} else if (field == GCDashOffset) {
	    gcvals->dash_offset = va_arg(ap, int);
	    value_mask |= GCDashOffset;
	} else if (field == GCDashList) {
#ifdef __GNUC__
	    gcvals->dashes = va_arg(ap, int);
#else
	    gcvals->dashes = va_arg(ap, char);
#endif
	    value_mask |= GCDashList;
	} else if (field == GCArcMode) {
	    gcvals->arc_mode = va_arg(ap, int);
	    value_mask |= GCArcMode;
	} else {
	    /* Error */
	    fprintf(stderr, "unknown field to ux11_fill_gcvals: %x\n", field);
	    abort();
	}
    }
    va_end(ap);
    return value_mask;
}
