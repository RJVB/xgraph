/*
 * Internal definitions for ux11 library
 * Requires inclusion of ux11.h.
 */

#ifndef UX11_INTERNAL
#define UX11_INTERNAL

#ifdef OLD_DEFINES
#ifdef __STDC__
#	ifdef __GNUC__
#		include <varargs.h>
#		define _STDARG_H
#	else
/* #		include "ux11_stdarg.h"	*/
/* #		include "varargs.h"	*/
#	endif
#	define VOID_P	void *
#	define VARARGS(func, rtn, args)	rtn func args
#else
#	include <varargs.h>
#	define VARARGS(func, rtn, args)	rtn func(va_alist) va_dcl
#	define VOID_P	char *
#endif
#endif

#include "../va_dcl.h"

/*
 * Some standards not defined in header files
 */

#ifndef abort
DECLARE(abort, void, ());
#endif

#ifndef __STDC__
DECLARE(fprintf, int, (FILE *fp, char *format, VA_DCL));
DECLARE(malloc, char *, (unsigned long));
DECLARE(realloc, char *, (char *ptr, unsigned size));
#endif

#endif	/* UX11_INTERNAL	*/
