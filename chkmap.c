/* Test of a quirky routine... (or compiler??)	*/

#include <stdio.h>
#include <math.h>
#include <X11/X.h>
#include <X11/Xlib.h>

#ifdef linux
#include <bits/nan.h>
#else
#include <nan.h>
#endif

#if !defined(IDENTIFY)

#ifndef SWITCHES
#	ifdef DEBUG
#		define _IDENTIFY(s,i)	static const char *ident= "$Id:@(#) '" __FILE__ "'-[" __DATE__ "," __TIME__ "]-(\015\013\t\t" s "\015\013\t) DEBUG version" i " $"
#	else
#		define _IDENTIFY(s,i)	static const char *ident= "$Id:@(#) '" __FILE__ "'-[" __DATE__ "," __TIME__ "]-(\015\013\t\t" s "\015\013\t)" i " $"
#	endif
#else
  /* SWITCHES contains the compiler name and the switches given to the compiler.	*/
#	define _IDENTIFY(s,i)	static const char *ident= "$Id: @(#) '" __FILE__ "'-[" __DATE__ "," __TIME__ "]-(\015\013\t\t" s "\015\013\t)["SWITCHES"]" " $"
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

IDENTIFY( "Does (p)gcc correctly evaluate simple math?" );

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

#define StdErr	stderr

#define NaNorInf(X)	(((IEEEfp *)&(X))->s.e==NAN_VAL)

#define INF(X)	(((IEEEfp *)&(X))->s.e==NAN_VAL &&\
	((IEEEfp *)&(X))->s.f1== 0 &&\
	((IEEEfp *)&(X))->s.f2== 0)

#define Inf(X)	((INF(X))?((((IEEEfp *)&(X))->l.high==POS_INF)?1:-1):0)

#define Gonio(fun,x)	(fun(((x)+Gonio_Base_Offset)/Units_per_Radian))
#define InvGonio(fun,x)	((fun(x)*Units_per_Radian-Gonio_Base_Offset))
#define Sin(x) Gonio(sin,x)
#define Cos(x) Gonio(cos,x)
#define Tan(x) Gonio(tan,x)

double Gonio_Base_Value= 2*M_PI, Gonio_Base_Value_2= M_PI, Gonio_Base_Value_4= M_PI_2;
double Units_per_Radian= 1.0, Gonio_Base_Offset=0;

int debugFlag= 0;

double Gonio_Base( double base, double offset)
{ double YY;
	if( base< 0 ){
		base*= -1;
	}

	YY= Gonio_Base_Value;

	if( base ){
		Units_per_Radian= (Gonio_Base_Value= base )/ (2*M_PI);
		Gonio_Base_Value_2= base / 2.0;
		Gonio_Base_Value_4= base / 4.0;
	}
	Gonio_Base_Offset= offset;
	if( debugFlag && base!= YY ){
		fprintf( StdErr, "Gonio_Base(%g,%g): base=%g => %g units per radian\n",
			base, offset, Gonio_Base_Value, Units_per_Radian
		);
		fflush( StdErr );
	}
	return( Gonio_Base_Value);
}

XSegment *make_arrow_point1( double xp, double yp, double ang, double alen, double aspect )
{ static XSegment arr[2];
  static prev_alen= 1;
	if( !NaNorInf(ang) ){
	  double x1, y1, x3, y3, c, s, r_aspect;
		if( NaNorInf(alen) ){
		  double p;
			if( Inf(alen)== -1 && prev_alen> 0 ){
				p= -prev_alen;
			}
			else{
				p= prev_alen;
			}
			if( debugFlag ){
				fprintf( StdErr, "make_arrow_point1(): requested alen=%g; using cached previous value %g\n",
					alen, p
				);
			}
			alen= p;
		}
		else{
			prev_alen= alen;
		}
		x3= x1= xp- alen;
		y1= yp+ alen/3.0;
		y3= yp- alen/3.0;
		if( aspect> 1 ){
			r_aspect= 1.0/ aspect;
			aspect= 1;
		}
		else{
			r_aspect= 1;
		}
		c= Cos( (-ang) )* r_aspect;
		s= Sin( (-ang) )* aspect;
		arr[0].x1= (short)( xp+ (x1-xp)* c- (y1-yp)* s+ 0.5);
		arr[0].y1= (short)( yp+ (x1-xp)* s+ (y1-yp)* c+ 0.5);
		arr[0].x2= (short)( xp+ 0.5);
		arr[0].y2= (short)( yp+ 0.5);
		arr[1].x1= arr[0].x2;
		arr[1].y1= arr[0].y2;
/* #define test_linux_pgcc	*/
#define linux_pgcc_bug_yell
#if defined(linux) && !defined(test_linux_pgcc)
{
		  /* RJB 20000212: gcc version pgcc-2.91.66 19990314 (egcs-1.1.2 release), Linux Mandrake 6.1:
		   \ Weird. There are floating point calculation errors that occur in -O compiled code,
		   \ resulting in nan values. This seems to arise in some cases when evaluating an
		   \ expression of the type (a-b)*c or (a-b)/c. Thus, in the code below, I had
		   \ to change (x3-xp)*c by x3*c-xp*c to make it work. Note that similar expressions
		   \ (x3-xp)*s, (x1-xp)*c, etc. did *not* require this, at least not that I have yet
		   \ noticed...
		   \ Something similar happens in Elapsed_Since.
		   \ Some examples:
				make_arrow_point1(x=314,y=292,ang=203.649,alen=11,aspect=1:0.987132) => (nan,nan)
				make_arrow_point1(x=327,y=298,ang=62.0977,alen=11,aspect=1:0.987296) => (nan,nan)
				make_arrow_point1(x=350,y=241,ang=330.218,alen=11,aspect=1:0.987296) => (nan,nan)
				make_arrow_point1(x=468,y=330,ang=275.032,alen=11,aspect=0.987296:1) => (nan,nan)
				make_arrow_point1(x=526,y=330,ang=275.031,alen=11,aspect=0.987296:1) => (nan,nan)
			\
		   */
		   /* RJB 20000421: Installed gcc 2.95.2 . Got the following output:
				@(#) $Id:'/home/bertin/work/Archive/xgraph/chkmap.c'-[Apr 21 2000,22:06:54]-(
					Does pgcc correctly evaluate simple math?
				  )[/usr/local/bin/gcc -fwritable-strings -mcpu=pentiumpro -fno-fast-math -malign-double -ffloat-store -fno-strict-aliasing  -O6 -fstrength-reduce  -fexpensive-optimizations -frerun-cse-after-loop -fomit-frame-pointer -fschedule-insns -fschedule-insns2 -finline-functions     -o /home/bertin/work/Archive/xgraph/chkmap chkmap.c]$
				make_arrow_point1(x=314,y=292,ang=203.649,alen=11,aspect=1:1) => (nan,nan)-(321.421,300.909)
		    \ However, when compiling with -fno-omit-frame-pointer, behaviour seems to be correct?!
		    \ (Actually, I remember that pine, imapd or both have this option defined for linux compilation)
		    */
#ifdef linux_pgcc_bug_yell
  double x1, y1, x2, y2;
		x1= xp+ (x1-xp)* c- (y1-yp)* s;
		y1= yp+ (x1-xp)* s+ (y1-yp)* c;
		x2= xp+ x3*c-xp* c- (y3-yp)* s;
		y2= yp+ (x3-xp)* s+ (y3-yp)* c;
		if( NaNorInf(x1) || NaNorInf(y1) || NaNorInf(x2) || NaNorInf(y2) ){
			fprintf( StdErr, "make_arrow_point1(x=%g,y=%g,ang=%g,alen=%g,aspect=%g:%g) => (%g,%g)-(%g,%g)\n",
				xp, yp, ang,
				alen, aspect, r_aspect,
				x1, y1, x2, y2
			);
		}
#endif
		arr[1].x2= (short)( xp+ x3*c-xp*c- (y3-yp)*s+ 0.5);
		arr[1].y2= (short)( yp+ x3*s-xp*s+ (y3-yp)*c+ 0.5);
/* 		fprintf( StdErr, "x2=xp+ (x3-xp)*c- (y3-yp)*s= %g+ (%g-%g)*%g- (%g-%g)*%g=%g\n",	*/
/* 			xp, x3, xp, c, y3, yp, s, x2	*/
/* 		);	*/
/* 		fprintf( StdErr, "Arrow-head tip@(%g,%g),%gdeg,len=%g,aspect=%g: (%hd,%hd)-(%hd,%hd),(%hd,%hd)-(%hd,%hd)\n",	*/
/* 			xp, yp, ang, alen, aspect,	*/
/* 			arr[0].x1, arr[0].y1, arr[0].x2, arr[0].y2,	*/
/* 			arr[1].x1, arr[1].y1, arr[1].x2, arr[1].y2	*/
/* 		);	*/
}
#else
		arr[1].x2= (short)( xp+ (x3-xp)* c- (y3-yp)* s+ 0.5);
		arr[1].y2= (short)( yp+ (x3-xp)* s+ (y3-yp)* c+ 0.5);
#endif
		return( arr );
	}
	return( NULL );
}

main()
{
	debugFlag= 1;
	fprintf( stderr, "%s\n", ident_stub() );
	make_arrow_point1( 314.0, 292.0, 203.649, 11.0, 1.0 );
	make_arrow_point1( 327.0, 298.0, 62.0977, 11.0, 1.0 );
	make_arrow_point1( 350.0, 241.0, 330.218, 11.0, 1.0 );
	make_arrow_point1( 468.0, 330.0, 275.032, 11.0, 0.987296 );
	make_arrow_point1( 526.0, 330.0, 275.031, 11.0, 0.987296 );
	make_arrow_point1( 80, 326, 230.576, 11, 1);
	make_arrow_point1( 80, 326, 230.576, 11, 1);
	make_arrow_point1( 287, 341, 335.927, 11, 1);
	make_arrow_point1( 367, 294, 215.105, 11, 1);
	make_arrow_point1( 296, 263, 326.435, 11, 1);
	make_arrow_point1( 445, 195, 95.1724, 11, 1);
	make_arrow_point1( 388, 315, 18.1069, 11, 1);
	make_arrow_point1( 445, 195, 95.1724, 11, 1);
	make_arrow_point1( 388, 315, 18.1069, 11, 1);
	make_arrow_point1( 406, 74, 22.0371, 11, 0.987132);
	make_arrow_point1( 378, 74, 37.7699, 11, 1);
	make_arrow_point1( 419, 74, 13.8921, 11, 1);
	make_arrow_point1( 124, 147, 213.825, 11, 0.987132);
	make_arrow_point1( 446, 199, 271.654, 11, 0.987132);
	make_arrow_point1( 337, 341, 348.996, 11, 0.987132);
	make_arrow_point1( 80, 326, 230.576, 11, 1);
	exit(0);
}
