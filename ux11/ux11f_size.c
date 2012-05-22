/*
 * X11 Utility Functions
 */

#include <stdio.h>
#include "ux11.h"
#include "ux11_internal.h"

#ifdef CTAGS
unsigned long ux11_fill_size( VA_DCL )
#endif
VARARGS(ux11_fill_size, unsigned long, ( XSizeHints *hints, XWMHints *gcvals, VA_DCL ))
/*
 * ux11_fill_size(hints, name, value, VA_DCL ... , UX11_END);
 * XSizeHints *hints;
 * Sets the values of an XSizeHints structure using variable
 * argument lists.  The returned value is the the value_mask
 * indicating what field is set.  The fields available
 * are those described for the value mask (e.g. USPosition, etc).
 * Needs to be reworked substantially.
 */
{
    va_list ap;
    unsigned long value_mask = 0;
    unsigned long field;
    XPoint *pnt;
#ifdef HPUX8__STDC__


	va_start(ap);
#else

    va_start(ap, gcvals);
/*     gcvals = va_arg(ap, XWMHints *);	*/
#endif

    while ((field = va_arg(ap, unsigned long)) != UX11_END) {
	if (field == USPosition) {
	    pnt = va_arg(ap, XPoint *);
	    hints->x = pnt->x;
	    hints->y = pnt->y;
	    value_mask |= USPosition;
	} else if (field == USSize) {
	    pnt = va_arg(ap, XPoint *);
	    hints->width = pnt->x;
	    hints->height = pnt->y;
	    value_mask |=  USSize;
	} else if (field == PPosition) {
	    pnt = va_arg(ap, XPoint *);
	    hints->x = pnt->x;
	    hints->y = pnt->y;
	    value_mask |=  PPosition;
	} else if (field == PSize) {
	    pnt = va_arg(ap, XPoint *);
	    hints->width = pnt->x;
	    hints->height = pnt->y;
	    value_mask |=  PSize;
	} else if (field == PMinSize) {
	    pnt = va_arg(ap, XPoint *);
	    hints->min_width = pnt->x;
	    hints->min_height = pnt->y;
	    value_mask |=  PMinSize;
	} else if (field == PMaxSize) {
	    pnt = va_arg(ap, XPoint *);
	    hints->max_width = pnt->x;
	    hints->max_height = pnt->y;
	    value_mask |=  PMaxSize;
	} else if (field == PResizeInc) {
	    pnt = va_arg(ap, XPoint *);
	    hints->width_inc = pnt->x;
	    hints->height_inc = pnt->y;
	    value_mask |=  PResizeInc;
	} else if (field == PAspect) {
	    value_mask |=  PAspect;
	} else {
	    /* Default action */
	}

	if (field == InputHint) {
	    hints->input = va_arg(ap, int);
	    value_mask |= InputHint;
	} else if (field == StateHint) {
	    hints->initial_state = va_arg(ap, int);
	    value_mask |= StateHint;
	} else if (field == IconPixmapHint) {
	    hints->icon_pixmap = va_arg(ap, Pixmap);
	    value_mask |= IconPixmapHint;
	} else if (field == IconWindowHint) {
	    hints->icon_window = va_arg(ap, Window);
	    value_mask |= IconWindowHint;
	} else if (field == IconPositionHint) {
	    pnt = va_arg(ap, XPoint *);
	    hints->icon_x = pnt->x;
	    hints->icon_y = pnt->y;
	    value_mask |= IconPositionHint;
	} else if (field == IconMaskHint) {
	    hints->icon_mask = va_arg(ap, Pixmap);
	    value_mask |= IconMaskHint;
	} else if (field == WindowGroupHint) {
	    hints->window_group = va_arg(ap, XId);
	    value_mask |= WindowGroupHint;
	} else {
	    /* Default action */
	    fprintf(stderr, "unknown field to ux11_fill_hints: %x\n", field);
	    abort();
	}
    }
    va_end(ap);
    return value_mask;
}
