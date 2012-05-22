/*
 * X11 Utility Functions
 */

#include <stdio.h>
#include "ux11.h"
#include "ux11_internal.h"

#ifdef CTAGS
unsigned long ux11_fill_wattr( VA_DCL )
#endif
VARARGS(ux11_fill_wattr, unsigned long, (XSetWindowAttributes *wattr,VA_DCL ))
/*
 * ux11_fill_wattr(wattr, name, value, VA_DCL ... , UX11_END);
 * XSetWindowAttributes *wattr;
 * Sets items in a fresh window attribute structure using
 * variable argument lists.  The settable fields are given
 * by the value mask (e.g. CWBackPixmap, etc).  The return 
 * value is the valuemask of those fields set in the structure.
 */
{
    va_list ap;
    unsigned long value_mask = 0;
    unsigned long field;
#ifdef HPUX8__STDC__

	va_start( ap);
#else

    va_start(ap, wattr);
#endif

    while ((field = va_arg(ap, unsigned long)) != UX11_END) {
	if (field == CWBackPixmap) {
	    wattr->background_pixmap = va_arg(ap, Pixmap);
	    value_mask |= CWBackPixmap;
	} else if (field == CWBackPixel) {
	    wattr->background_pixel = va_arg(ap, unsigned long);
	    value_mask |= CWBackPixel;
	} else if (field == CWBorderPixmap) {
	    wattr->border_pixel = va_arg(ap, Pixmap);
	    value_mask |= CWBorderPixmap;
	} else if (field == CWBorderPixel) {
	    wattr->border_pixel = va_arg(ap, unsigned long);
	    value_mask |= CWBorderPixel;
	} else if (field == CWBitGravity) {
	    wattr->bit_gravity = va_arg(ap, int);
	    value_mask |= CWBitGravity;
	} else if (field == CWWinGravity) {
	    wattr->win_gravity = va_arg(ap, int);
	    value_mask |= CWWinGravity;
	} else if (field == CWBackingStore) {
	    wattr->backing_store = va_arg(ap, int);
	    value_mask |= CWBackingStore;
	} else if (field == CWBackingPlanes) {
	    wattr->backing_planes = va_arg(ap, unsigned long);
	    value_mask |= CWBackingPlanes;
	} else if (field == CWBackingPixel) {
	    wattr->backing_pixel = va_arg(ap, unsigned long);
	    value_mask |= CWBackingPixel;
	} else if (field == CWOverrideRedirect) {
	    wattr->override_redirect = va_arg(ap, Bool);
	    value_mask |= CWOverrideRedirect;
	} else if (field == CWSaveUnder) {
	    wattr->save_under = va_arg(ap, Bool);
	    value_mask |= CWSaveUnder;
	} else if (field == CWEventMask) {
	    wattr->event_mask = va_arg(ap, long);
	    value_mask |= CWEventMask;
	} else if (field == CWDontPropagate) {
	    wattr->do_not_propagate_mask = va_arg(ap, long);
	    value_mask |= CWDontPropagate;
	} else if (field == CWColormap) {
	    wattr->colormap = va_arg(ap, Colormap);
	    value_mask |= CWColormap;
	} else if (field == CWCursor) {
	    wattr->cursor = va_arg(ap, Cursor);
	    value_mask |= CWCursor;
	} else {
	    /* Error - not real graceful here */
	    fprintf(stderr, "unknown field to ux11_fill_wattr: %lx\n", field);
	    abort();
	}
    }
    va_end(ap);
    return value_mask;
}
