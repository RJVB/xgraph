/*
 *  R : A Computer Language for Statistical Data Analysis
 *  Copyright (C) 2002  The R Development Core Team.
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

/* (C) 1999, R.J.V. Bertin
 \ Code that puts in an identifier string, combined with compiling information.
 */
#if !defined(IDENTIFY)

#ifndef SWITCHES
#	ifdef DEBUG
#		define _IDENTIFY(s,i)	static char *ident= "@(#) $Id:'" __FILE__ "'-[" __DATE__ "," __TIME__ "]-(\015\013\t\t" s "\015\013\t) DEBUG version" i "$"
#	else
#		define _IDENTIFY(s,i)	static char *ident= "@(#) $Id:'" __FILE__ "'-[" __DATE__ "," __TIME__ "]-(\015\013\t\t" s "\015\013\t)" i "$"
#	endif
#else
  /* SWITCHES contains the compiler name and the switches given to the compiler.	*/
#	define _IDENTIFY(s,i)	static char *ident= "@(#) $Id:'" __FILE__ "'-[" __DATE__ "," __TIME__ "]-(\015\013\t\t" s "\015\013\t)["SWITCHES"]" "$"
#endif

#define __IDENTIFY(s,i)\
static char *ident_stub(){ _IDENTIFY(s,i);\
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
#	define IDENTIFY(s)	__IDENTIFY(s," (gcc)")
#else
#	define IDENTIFY(s)	__IDENTIFY(s," (cc)")
#endif

#endif

IDENTIFY("snprintf-R162.c, from R 1.6.2 (www.r-project.org) distilled by RJVB");

#include <stdlib.h>
#include <stdarg.h>

int vsnprintf(char *s, size_t n, const char *format, va_list ap);

int snprintf (char *s, size_t n, const char *format, ...)
{
    int total;
    va_list ap;

    va_start(ap, format);
    total = vsnprintf(s, n, format, ap);
    va_end(ap);
    return total;
}
