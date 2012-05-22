#ifndef _XGERRNO_H_

#include <errno.h>

#ifdef NO_STRERROR

#	if !( defined(linux) || ((defined(__MACH__) || defined(_APPLE_CC__)) && !(defined(_ANSI_SOURCE) || defined(_POSIX_C_SOURCE) || defined(_POSIX_SOURCE))) || defined(__CYGWIN__) )
#		if defined(__MACH__) || defined(__APPLE_CC__)
		extern __const int sys_nerr;
		extern __const char *__const sys_errlist[];
#		elif defined(_AIX)
		extern char *sys_errlist[];
		extern int sys_nerr;
#		else
		extern const char *sys_errlist[];
		extern const int sys_nerr;
#		endif
#	endif

#	ifndef _CX_H
#		define serror() ((errno>0&&errno<sys_nerr)?sys_errlist[errno]:"invalid errno")
#	endif

#else

#	define serror()	strerror(errno)

#endif

#define SYS_ERRLIST	_sys_errlist
#define SYS_NERR	_sys_nerr

#define _XGERRNO_H_
#endif
