/*
 * xgraph - program to graphically display numerical data
 *
 * David Harrison
 * University of California, Berkeley
 * 1989
 *
 * Copyright (c) 1988, 1989, Regents of the University of California.
 * All rights reserved.
 *
 * Use and copying of this software and preparation of derivative works
 * based upon this software are permitted.  However, any distribution of
 * this software or derivative works must include the above copyright
 * notice.
 *
 * This software is made available AS IS, and neither the Electronics
 * Research Laboratory or the University of California make any
 * warranty about the software, its performance or its conformity to
 * any specification.
 */

#ifndef _RIGHTS_
#define _RIGHTS_

#include "config.h"

#define COPYRIGHT "\015\013\tXGraph: $Copyright: (c) 1989-93, Regents of the University of California\015\013\t\tRenE J.V. Bertin 1990-now\tAll rights reserved. $\015\013"
/* static char copyright[] = COPYRIGHT;	*/

#define C__IDENTIFY(s,i)	static char *cident= s

#if !defined(DEBUG) || defined(POPUP_IDENT)

#	ifndef POPUP_IDENT

#		define C___IDENTIFY(s,i)\
static const char *cident_stub(){ C__IDENTIFY(s,i);\
	static char called=0;\
	if( !called){\
		called=cident;\
		return(cident_stub());\
	}\
	else{\
		called= 0;\
		return(cident);\
	}}

#	else

#ifndef _MAIN_C
	extern int XG_error_box( LocalWin **wi, char *title, char *mesg, ... );
#endif

#		define C___IDENTIFY(s,i)\
static const char *cident_stub(){ _IDENTIFY("",i); static char copyright= s;\
	static char called=0;\
	if( !called){\
		called=1;\
		return(cident_stub());\
	}\
	else{\
	 LocalWin *lwi=NULL;\
		called= 0;\
		XG_error_box( &lwi, "About this file and XGraph in general:", copyright, "\n ", ident, 0 );\
		return(ident);\
	}}
#	endif

#else
#	define C___IDENTIFY(s,i) C__IDENTIFY(s,i);
#endif

#ifdef __GNUC__
// #	define C_IDENTIFY(s)	__attribute__((used)) C__IDENTIFY(s," (gcc" STRING(__GNUC__) ")")
#	ifdef __clang__
#		define C_IDENTIFY(s)	__attribute__((used)) C__IDENTIFY(s," (clang" STRING(__clang__) ")")
#	else
#		define C_IDENTIFY(s)	__attribute__((used)) C__IDENTIFY(s," (gcc" STRING(__GNUC__) ")")
#	endif
#else
#	define C_IDENTIFY(s)	C__IDENTIFY(s," (cc)")
#endif

#ifdef _MAIN_C
C_IDENTIFY(COPYRIGHT);
#endif


extern char greek_inf[4], greek_min_inf[5];
extern int use_greek_inf;

#define GREEK_MIN_INF	greek_min_inf
#define GREEK_MIN_INF_LEN	sizeof(GREEK_MIN_INF)
#define GREEK_INF	greek_inf
#define GREEK_INF_LEN	sizeof(GREEK_INF)

extern char *cgetenv(char *name);
#undef GetEnv
#undef SetEnv
extern char *GetEnv( char *name);
extern char *SetEnv( char *name, char *value);
#define getenv(n) GetEnv(n)
#define setenv(n,v) SetEnv(n,v)
extern char *EnvDir;						/* ="tmp:"; variables here	*/

#endif /* _RIGHTS_ */
