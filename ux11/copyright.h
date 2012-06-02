#ifdef XGRAPH
#	include "../copyright.h"
#else

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

#define COPYRIGHT "\015\013\t\tXGraph: Copyright (c) 1989-93,\015\013\t\tRegents of the University of California\015\013\t\tRJB 1990-1994\015\013\t\tAll rights reserved.\015\013\t"
static char copyright[] = COPYRIGHT;

#if defined(i386) || defined(__i386__)
#	define __ARCHITECTURE__	"i386"
#elif defined(__x86_64__) || defined(x86_64) || defined(_LP64)
#	define __ARCHITECTURE__	"x86_64"
#elif defined(__ppc__)
#	define __ARCHITECTURE__	"ppc"
#else
#	define __ARCHITECTURE__	""
#endif

#ifndef SWITCHES
#	ifdef DEBUG
#		define _IDENTIFY(s,i)	static const char *ident= "@(#) '" __FILE__ "'-[" __DATE__ "," __TIME__ "]-(" s ") DEBUG version" i __ARCHITECTURE__ " $"
#	else
#		define _IDENTIFY(s,i)	static const char *ident= "@(#) '" __FILE__ "'-[" __DATE__ "," __TIME__ "]-(" s ")" i __ARCHITECTURE__ " $"
#	endif
#else
  /* SWITCHES contains the compiler name and the switches given to the compiler.	*/
#	define _IDENTIFY(s,i)	static const char *ident= "@(#) '" __FILE__ "'-[" __DATE__ "," __TIME__ "]-(" s ")["__ARCHITECTURE__" "SWITCHES"]" " $"
#endif

#define __IDENTIFY(s,i)\
static const char *ident_stub(){ _IDENTIFY(s,i);\
	static char called=0;\
	if( !called){\
		called=1;\
		return(ident_stub());\
	}\
	else{\
		called= 0;\
		return(ident);\
	}}

#ifdef __GNUC__
// #	define IDENTIFY(s)	__attribute__((used)) __IDENTIFY(s," (gcc)")
#	define IDENTIFY(s)	__attribute__((used)) __IDENTIFY(s," (gcc" STRING(__GNUC__) ")")
#else
#	define IDENTIFY(s)	__IDENTIFY(s," (cc)")
#endif

IDENTIFY(COPYRIGHT);


#endif /* _RIGHTS_ */

#endif
