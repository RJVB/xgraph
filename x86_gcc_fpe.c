/* This is taken from the gcc info files, version 2.95.2, on floating point issues.
 \ But it is only useful for Fortran, it appears.
 */

#if defined(i386) && defined(__GNUC__)

#include <stdio.h>

#ifdef STANDALONE

#if !defined(IDENTIFY)

#ifndef SWITCHES
#	ifdef DEBUG
#		define _IDENTIFY(s,i)	static const char *ident= "@(#) $Id:'" __FILE__ "'-[" __DATE__ "," __TIME__ "]-(\015\013\t\t" s "\015\013\t) DEBUG version" i "$"
#	else
#		define _IDENTIFY(s,i)	static const char *ident= "@(#) $Id:'" __FILE__ "'-[" __DATE__ "," __TIME__ "]-(\015\013\t\t" s "\015\013\t)" i "$"
#	endif
#else
  /* SWITCHES contains the compiler name and the switches given to the compiler.	*/
#	define _IDENTIFY(s,i)	static const char *ident= "@(#) $Id:'" __FILE__ "'-[" __DATE__ "," __TIME__ "]-(\015\013\t\t" s "\015\013\t)["SWITCHES"]" "$"
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
#	define IDENTIFY(s)	__IDENTIFY(s," (gcc)")
#else
#	define IDENTIFY(s)	__IDENTIFY(s," (cc)")
#endif

#endif
IDENTIFY( "gcc x86 FPU control" );

#endif

#include <fpu_control.h>

#ifndef CheckMask
#	define	CheckMask(var,mask)	(((var)&(mask))==(mask))
#endif

#define STR(name)	# name
#ifndef STRING
#	define STRING(name)	STR(name)
#endif

#define CatMask(buf,m,mask)	if( CheckMask(m,mask) ){ if(*buf){strcat(buf,"|");} strcat(buf, # mask); }

static char *fpu_mask_string( fpu_control_t m)
{ static char buf[180];
	buf[0]= '\0';
	CatMask( buf, m, _FPU_RC_ZERO);
	CatMask( buf, m, _FPU_RC_UP);
	CatMask( buf, m, _FPU_RC_DOWN);
	CatMask( buf, m, _FPU_RC_NEAREST);
		CatMask( buf, m, _FPU_EXTENDED)
		else CatMask( buf, m, _FPU_DOUBLE)
		else CatMask( buf, m, _FPU_SINGLE);
	CatMask( buf, m, _FPU_MASK_PM);
	CatMask( buf, m, _FPU_MASK_UM);
	CatMask( buf, m, _FPU_MASK_OM);
	CatMask( buf, m, _FPU_MASK_ZM);
	CatMask( buf, m, _FPU_MASK_DM);
	CatMask( buf, m, _FPU_MASK_IM);
}

static void __attribute__ ((constructor))
trapfpe ()
{
	    /* This enables FPE exception trapping for "common" exceptions:	*/
	fpu_control_t cw= _FPU_DEFAULT & ~(_FPU_MASK_IM | _FPU_MASK_ZM | _FPU_MASK_OM), ow, nw;
    /* Whereas this puts the FPU in double precision (as opposed to extended precision) mode:	*/
  cw= (_FPU_DEFAULT & ~(_FPU_EXTENDED | (1<<12))) | _FPU_DOUBLE;

	if( !getenv( "DEFAULT_FPU") ){
	  char *c;
		_FPU_GETCW(ow);
		_FPU_SETCW( cw );
		_FPU_GETCW(nw);
		if( (c=getenv( "SHOW_FPU")) ){
			fprintf( stderr, "FPU state mask: originally 0x%lx, requested 0x%lx, now 0x%lx\n", ow, cw, nw );
			if( strncasecmp(c, "verbose", 7)== 0 ){
				fprintf( stderr, "\tOriginal mask: %s\n", fpu_mask_string(ow) );
				fprintf( stderr, "\tRequested mask: %s\n", fpu_mask_string(cw) );
				fprintf( stderr, "\tActual mask: %s\n", fpu_mask_string(nw) );
				fprintf( stderr, "\t\t(see fpu_control.h for explanation)\n");
			}
		}
	}
	else if( getenv( "SHOW_FPU") ){
		_FPU_GETCW(ow);
		fprintf( stderr, "FPU state mask: 0x%lx\n", ow );
	}
}

#endif
