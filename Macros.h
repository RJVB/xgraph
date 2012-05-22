/* ****************
 * Macros.h: some useful macros
 * (C)(R) R. J. Bertin
 * :ts=4
 * ****************
 */

#ifndef _MACRO_H
#define _MACRO_H

#ifndef _CPU_H
#	include "cpu.h"
#endif

#include <stddef.h>
#include <stdlib.h>
#include <limits.h>

/* Original SUN macro.h file included for convenience: */

#ifdef NULL
#	undef NULL
#endif
#define NULL    ((void*)0)

#define reg     register

#define MAXUSHORT	USHRT_MAX
#ifndef MAXSHORT
#	define MAXSHORT	SHRT_MAX
#endif
#ifndef MAXINT
#	define MAXINT		INT_MAX		/* works with 16 and 32 bit ints	*/
#	define MAXUINT	UINT_MAX
#	define MAXLONG	LONG_MAX
#endif
#define MAXULONG	ULONG_MAX

#define SETBIT(s,n)     (s[(n)/16] |=  (1 << ((n) % 16)))
#define XORBIT(s,n)     (s[(n)/16] ^=  (1 << ((n) % 16)))
#define CLRBIT(s,n)     (s[(n)/16] &= ~(1 << ((n) % 16)))
#define TSTBIT(s,n)     (s[(n)/16] &   (1 << ((n) % 16)))

#ifndef LOOPDN
#define LOOPDN(v, n)    for ((v) = (n); --(v) != -1; )
#endif

#define DIM(a)          (sizeof(a) / sizeof(a[0]))

#define MOD(v,d)  ( (v) >= 0  ? (v) % (d) : (d) - ( (-(v))%(d) ) )
#define ABS(v)    ((v)<0?-(v):(v))
#define FABS(v)   (((v)<0.0)?-(v):(v))
#ifndef SIGN
#	define SIGN(x)		(((x)<0)?-1:1)
#endif
#define ODD(x)		((x) & 1)

#define BITWISE_AND(a,b)	((a)&(b))
#define BITWISE_OR(a,b)	((a)|(b))
#define BITWISE_XOR(a,b)	((a)^(b))
#define BITWISE_NOT(a)	(~(a))

#define LOGIC_AND(a,b)	((a)&&(b))
#define LOGIC_OR(a,b)	((a)||(b))
#define LOGIC_XOR(a,b)	(((int)(a)!=0)^((int)(b)!=0))
#define LOGIG_NOT(a)	(!(a))

#ifdef DEBUG
#define ASSERTS(cond) \
   {if (!(cond)) \
       printf("Assertion failure: line %d in %s\n", __LINE, __FILE);   }
#else
#define ASSERTS(cond)
#endif


/********************* Macros to access linear arrays as MultiDimensional ****/
/*                     Usage: map?d( x1,..,xn, Size(x1),..,Size(xn) )        */
/*                     !!! map?d() return a long argument !!!                */

#define map2d(x,y,X,Y)          ((long)y*X+(long)x)
#define map3d(x,y,z,X,Y,Z)      (((long)z*Y+(long)y)*X+(long)x)
#define map4d(a,b,c,d,A,B,C,D)  ((((long)d*C+(long)c)*B+(long)b)*A+(long)a)

/*                      Macros to allocate (multidim) arrays                */
/*          Usage: if( calloc??_error(pointer,type,size1,..,sizen)) ERROR   */

#ifdef _MAC_THINKC_
#define lcalloc clalloc
	char *lcalloc();
#endif

#ifndef lfree
/* cx.h has not yet been included	*/
#define lcalloc calloc
#endif


#define calloc_error(p,t,n)  ((p=(t*)lcalloc((unsigned long)(n),(unsigned long)sizeof(t)))==(t*)0)
#define calloc2d_error(p,t,x,y)  ((p=(t*)lcalloc((unsigned long)(x*y),(unsigned long)sizeof(t)))==(t*)0)
#define calloc3d_error(p,t,x,y,z)  ((p=(t*)lcalloc((unsigned long)(x*y*z),(unsigned long)sizeof(t)))==(t*)0)
#define calloc4d_error(p,t,a,b,c,d)  ((p=(t*)lcalloc((unsigned long)(a*b*c*d),(unsigned long)sizeof(t)))==(t*)0)

#define lcalloc_error calloc_error
#define lcalloc2d_error calloc2d_error
#define lcalloc3d_error calloc3d_error
#define lcalloc4d_error calloc4d_error

/*                  Miscellaneous Macros                                    */

#ifndef MAX
#define MAX(a,b)                (((a)>(b))?(a):(b))
#endif
#ifndef MIN
#define MIN(a,b)                (((a)<(b))?(a):(b))
#endif
#define MAXp(a,b)				(((a)&&(a)>(b))?(a):(b))	/* maximal non-zero	*/
#define MINp(a,b)				(((a)&&(a)<(b))?(a):(b))	/* minimal non-zero	*/
#ifndef SWAP
#	define SWAP(a,b,type)	{type c= (a); (a)= (b); (b)= c;}
#endif

#define ProgName				(argv[0])


#ifndef CLIP
#	define CLIP(var,low,high)	if((var)<(low)){\
		(var)=(low);\
	}else if((var)>(high)){\
		(var)=(high);}
#endif
#define CLIP_EXPR(var,expr,low,high)	{ double l, h; if(((var)=(expr))<(l=(low))){\
	(var)=l;\
}else if((var)>(h=(high))){\
	(var)=h;}}

#define CLIP_IEXPR(var,expr,low,high)	{ double e= (expr), l= (low), h= (high);\
	if( e< l ){ \
		e= l; \
	} \
	else if( e> h ){ \
		e= h; \
	} \
	(var)= (int) e; \
}

#ifndef CLIP_EXPR_CAST
/* A safe casting/clipping macro.	*/
#	define CLIP_EXPR_CAST(ttype,var,stype,expr,low,high)	{stype clip_expr_cast_lvalue=(expr); if(clip_expr_cast_lvalue<(low)){\
		(var)=(ttype)(low);\
	}else if(clip_expr_cast_lvalue>(high)){\
		(var)=(ttype)(high);\
	}else{\
		(var)=(ttype)clip_expr_cast_lvalue;}}
#endif

#ifdef True
#	undef True
#endif
#ifdef False
#	undef False
#endif
#ifdef Boolean
#	undef Boolean
#endif

#ifndef _XtIntrinsic_h
	typedef enum Boolean { False, True} Boolean;
#else
#	define True 1
#	define False 0
#endif

#ifdef MCH_AMIGA
#	define sleep(x)	Delay(60*x)
#endif

#include "xgerrno.h"

/* #define SinCos(a,s,c)	{(s)=sin((a));(c)=cos((a));}	*/
#if defined(linux)
		/* In principle, a sincos() routine is (much) faster than 2 separate calls to sin() and cos().
		 \ So use it when the hardware and software provide it!
		 \ It is available on LinuxX86 (in the FPU, so also under other Unixes),
		 \ on SGIs, it seems not to be there, but it can be had via Performer (pfSinCos())
		 \ On other machines, I don't know.
		 */
#	define SinCos(a,s,c)	sincos( ((a)+Gonio_Base_Offset)/Units_per_Radian,(s),(c))
#	define NATIVE_SINCOS	1

#elif defined(sgi) && defined(__PR_H__)

#	define SinCos(a,s,c)	{float fs, fc; pfSinCos( (float)((a)+Gonio_Base_Offset)/Units_per_Radian,&fs,&fc); *(s)=fs,*(c)=fc;}
#	define sincos(a,s,c)	{float fs, fc; pfSinCos( (float)(a),&fs,&fc); *(s)=fs,*(c)=fc;}
#	define NATIVE_SINCOS	2

#elif defined(__APPLE__)
#	if !defined(NATIVE_SINCOS)
#		include "AppleVecLib.h"
		static inline void SinCos(double a, double *s, double *c)
		{ const int nn = 1;
		  extern double Units_per_Radian, Gonio_Base_Offset;
		  const double aa = (a + Gonio_Base_Offset) / Units_per_Radian;
			vvsincos( s, c, &aa, &nn );
		}

		static inline void sincos(double a, double *s, double *c)
		{ const int nn = 1;
			vvsincos( s, c, &a, &nn );
		}
#		define NATIVE_SINCOS	1
#	endif
#else

#	define SinCos(a,s,c)	*(s)=Sin((a)),*(c)=Cos((a))
#	define sincos(a,s,c)	*(s)=sin((a)),*(c)=cos((a))
#	define NATIVE_SINCOS	0
#endif

#ifndef degrees
#	define degrees(a)			((a)*57.2957795130823229)
#endif
#ifndef radians
#	define radians(a)			((a)/57.2957795130823229)
#endif


/* #define degrees(a)		((a)*57.295779512)	*/
/* #define radians(a)		((a)/57.295779512)	*/


/* #define streq(a,b)	!strcmp(a,b)	*/
/* #define strneq(a,b,n)	!strncmp(a,b,n)	*/
#define strcaseeq(a,b)	!strcasecmp(a,b)
#define strncaseeq(a,b,n)	!strncasecmp(a,b,n)

/* some convenient typedef's:	*/

/* functionpointer (method) typedef's	*/

typedef char (*char_method)();

#ifndef _TRAPS_H
	typedef int (*int_method)();
#endif

typedef unsigned int (*uint_method)();
typedef short (*short_method)();
typedef unsigned short (*ushort_method)();
typedef long (*long_method)();
typedef unsigned long (*ulong_method)();
typedef void (*void_method)();
typedef void* (*pointer_method)();
typedef pointer_method (*method_method)();
typedef double (*double_method)();

/* typedef unsigned char pixel;	*/


#include "NaN.h"
/* NAN_VAL is defined in NaN.h . The code enclosed in the following block
 \ should thus not be compiled, which is true because it remains only for
 \ eventual future debugging.
 */
#ifndef NAN_VAL
/* since math.h contains a MAXDOUBLE definition, undef any
 * made in limits.h
 */
#ifdef MAXDOUBLE
#	undef MAXDOUBLE
#endif
#include "math.h"
#include "mathdef.h"

/* Handling of Not_a_Number's (only in IEEE floating-point standard) */

#ifdef _APOLLO_SOURCE
#	include "nan_domain.h"
#else
#	ifdef linux
#		include <bits/nan.h>
#	else
#		include <nan.h>
#	endif
#endif

#if 0
#if defined(i386)
/* definitions for byte-reversed-ordered machines:	*/
typedef union IEEEsfp {
	struct {
		short low, high;
	} l;
	struct {
		unsigned f:23;
		unsigned e:8;
		unsigned s:1;
	} s;
	float f;
} IEEEsfp;

typedef union IEEEfp {
	struct {
		long low, high;
	} l;
	struct {
		unsigned f1:32,f2:20;
		unsigned e:11;
		unsigned s:1;
	} s;
	double d;
} IEEEfp;
#else
typedef union IEEEsfp {
	struct {
		short high, low;
	} l;
	struct {
		unsigned s:1;
		unsigned e:8;
		unsigned f:23;
	} s;
	float f;
} IEEEsfp;

typedef union IEEEfp {
	struct {
		long high, low;
	} l;
	struct {
		unsigned s:1;
		unsigned e:11;
		unsigned f1:20, f2:32;
	} s;
	double d;
} IEEEfp;
#endif

#define NAN_VAL	0x7ff
#define POS_INF	0x7ff00000
#define NEG_INF	0xfff00000

#undef NaN
/* NaN or Inf: (s.e==NAN_VAL and ((s.f1!=0 and s.f2!=0) or (s.f1==0 and s.f2==0)))
 * i.e. s.e==NAN_VAL and (!(s.f1 xor s.f2))
 */
#define NaNorInf(X)	(((IEEEfp *)&(X))->s.e==NAN_VAL)

#define I3Ed(X)	((IEEEfp*)&(X))

#ifdef i386
/* mul[Inf,0] on this arch. yields s.f1=0, s.f2=0x80000 and s.s=1	*/
#	define NaN(X)	((I3Ed(X)->s.e==NAN_VAL)? \
	((I3Ed(X)->s.f1!=0)? I3Ed(X)->s.f1 : \
		((I3Ed(X)->s.f2!= 0)? I3Ed(X)->s.f2 : 0)) : 0)
#else
/* mul[Inf,0] on this arch. yields s.f1=0x7ffff, s.f2=0 and s.s=0	*/
#	define NaN(X) ((((IEEEfp *)&(X))->s.e==NAN_VAL &&\
		((IEEEfp *)&(X))->s.f1!= 0 )? ((IEEEfp *)&(X))->s.f1 : 0 )
#endif

#define INF(X)	(((IEEEfp *)&(X))->s.e==NAN_VAL &&\
	((IEEEfp *)&(X))->s.f1== 0 &&\
	((IEEEfp *)&(X))->s.f2== 0)

#define set_NaN(X)	{IEEEfp *local_IEEEfp=(IEEEfp*)(&(X));\
	local_IEEEfp->s.s= 0; local_IEEEfp->s.e=NAN_VAL ;\
	local_IEEEfp->s.f1= 0xfffff ;\
	local_IEEEfp->s.f2= 0xffffffff;}

#define Inf(X)	((INF(X))?((((IEEEfp *)&(X))->l.high==POS_INF)?1:-1):0)

#define set_Inf(X,S)	{IEEEfp *local_IEEEfp=(IEEEfp*)(&(X));\
	local_IEEEfp->l.high=(S>0)?POS_INF:NEG_INF ;\
	local_IEEEfp->l.low= 0x00000000 ;}

#define set_INF(X)	{IEEEfp *local_IEEEfp=(IEEEfp*)(&(X));\
	local_IEEEfp->s.e=NAN_VAL ;\
	local_IEEEfp->s.f1= 0x00000 ;\
	local_IEEEfp->s.f2= 0x00000000 ;}


#endif /* NAN_VAL	*/
#else
#	include "NaN.h"
#endif

/*  Warning: the following macros are also defined in xtb/xtb.h!	*/
#ifndef CheckMask
#	define	CheckMask(var,mask)	(((var)&(mask))==(mask))
#endif
#ifndef CheckORMask
#	define	CheckORMask(var,mask)	(((var)&(mask))!=0)
#endif
#ifndef CheckExclMask
#	define	CheckExclMask(var,mask)	(((var)^(mask))==0)
#endif

#ifndef STR
#	define STR(name)	# name
#endif

#include "xgALLOCA.h"

/* Some macros taken from the NR:	*/
#ifndef SQR
static double sqrval;
#	define SQR(v,type)	(type)(((sqrval=(v)))? sqrval*sqrval : 0)
#endif

#ifndef IMIN
static int mi1,mi2;
#	define IMIN(m,n)	(mi1=(m),mi2=(n),(mi1<mi2)? mi1 : mi2)
#endif

DEFUN( *cgetenv, (char *name), char);
#undef GetEnv
#undef SetEnv
DEFUN( *GetEnv, ( char *name), char);
DEFUN( *SetEnv, ( char *name, char *value), char);
#define getenv(n) GetEnv(n)
#define setenv(n,v) SetEnv(n,v)
extern char *EnvDir;						/* ="tmp:"; variables here	*/

#include "pragmas.h"

#endif
