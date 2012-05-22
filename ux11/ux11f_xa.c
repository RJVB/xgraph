/*
 * X11 Utility Functions
 */

#include <stdio.h>
#include "ux11.h"
#include "ux11_internal.h"

#ifdef CTAGS
int ux11_fill_xa( VA_DCL )
#endif
VARARGS(ux11_fill_xa, int, ( Arg *arg_list, int size, VA_DCL ))
/*
 * int ux11_fill_xa(arg_list, size, name, value, VA_DCL ... , UX11_END)
 * Arg *arg_list;
 * int size;
 * String name;
 * XtArgVal value;
 * Sets the components of a X toolkit argument list.  The argument list
 * is passed in as `arg_list'.  Its size should be passed in as `size'.
 * The routine returns the number of arguments set if successful, zero
 * if not successful (e.g. not enough slots).
 */
{
    va_list ap;
    String field;
    int len;
#ifdef HPUX8__STDC__

	va_start(ap);
#else
    va_start(ap, size);
#endif

    len = 0;
    while ((field = va_arg(ap, String)) != (String) UX11_END) {
	if (len >= size) {
	    /* Not enough slots */
	}
	arg_list[len].name = field;
	arg_list[len].value = va_arg(ap, XtArgVal);
	len++;
    }
    return len;
}
