#ifndef VA_DCL

#ifdef __cplusplus
	extern "C" {
#endif


/* 20040109:
	Aligned everything to using stdarg.h
 */

#if defined (__GNUC__) || defined (__STDC__)

#ifdef _AUX_SOURCE
#	define VA_DCL	int __builtin_va_alist, ...
#endif
#if defined(_APOLLO_SOURCE) || defined(sgi)
#	define VA_DCL	int va_alist, ...
#endif

#endif

#ifndef VA_DCL
#	if defined(_HPUX_SOURCE) && !defined(__GNUC__)
#		define VA_DCL long va_alist, ...
#	else
#		define VA_DCL	/* */ ...
#	endif
#endif

#ifdef __STDC__
#	define FNPTR(fname, rtn, args)	rtn (*fname)args
#	if 1 
	  /* gcc is quite the standard...	*/
#	ifdef __cplusplus
		}
#	endif
/* #		include <varargs.h>	*/
#		include <stdarg.h>
#	ifdef __cplusplus
		extern "C" {
#	endif
#		ifndef _STDARG_H
		  /* This is for Irix's MIPSPro compiler, which defines __STDARG_H__ ... */
#			define _STDARG_H
#		endif
#		undef VA_DCL
#		define VA_DCL	...
#		define VARARGS(func, rtn, args)	rtn func args
#	else
#	ifdef __cplusplus
		}
#	endif
#		include <varargs.h>
#	ifdef __cplusplus
		extern "C" {
#	endif
#		define VOID_P	void *
#		ifndef _HPUX_SOURCE
#			define VARARGS(func, rtn, args)	rtn func args
#		else
		  /* hpux cc: strange varargs handling??	*/
#			define VARARGS(func, rtn, args)	/* VARARGS */ rtn func(va_alist) va_dcl
#		endif
#	endif
#else
#	define FNPTR(fname, rtn, args)	rtn (*fname)()
#	ifdef __cplusplus
		}
#	endif
#	include <varargs.h>
#	ifdef __cplusplus
		extern "C" {
#	endif
#	define VARARGS(func, rtn, args)	/*VARARGS*/ rtn func(va_alist) va_dcl
#endif

#ifndef _STDARG_H
/* 960808. OK. I'm fed up with the differences. So much for ansi portability concerning
 \ varags. 'll do everything the K&R way!
 */
#	undef VA_DCL
#	define VA_DCL	/* VARARGS */
#	undef VARARGS
#	define VARARGS(func, rtn, args)	/* VARARGS */ rtn func(va_alist) va_dcl
#endif

#ifdef __cplusplus
	}
#endif

#endif
