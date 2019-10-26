/*
vim:ts=5:sw=5:
:ts=5
 */
	/* Mxt.h: headerfile for extended math lib mxt.lib for Macintosh MPW C*/

#ifndef _MXT_H
#define _MXT_H

#ifdef __cplusplus
	extern "C" {
#endif

#ifndef _CPU_H
#	include <cpu.h>
#endif
#ifndef _MACRO_H
#	include <Macros.h>
#endif


/* since math.h contains a MAXDOUBLE definition, undef any
 * made in limits.h
 */
#ifdef MAXDOUBLE
#	undef MAXDOUBLE
#endif
#include "math.h"
#include "mathdef.h"

/* Handling of Not_a_Number's (only in IEEE floating-point standard) */

#if 1

#   include "NaN.h"

#else

#if defined(linux)
#	include <bits/nan.h>
#elif defined(_AIX) || defined(__MACH__) || defined(__APPLE_CC__)
#	include <local/nan.h>
#elif !defined(WIN32) && !defined(__CYGWIN__)
#	include <nan.h>
#endif

#ifdef i386
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

#define I3Ed(X)	((IEEEfp*)&(X))
#define I3Edf(X)	((IEEEsfp*)&(X))

/* NaN or Inf: (s.e==NAN_VAL and ((s.f1!=0 and s.f2!=0) or (s.f1==0 and s.f2==0)))
 * i.e. s.e==NAN_VAL and (!(s.f1 xor s.f2))
 */
#define NaNorInf(X)	(((IEEEfp *)&(X))->s.e==NAN_VAL)

#define fpNaNorInf(X)	((sizeof(X)==sizeof(float))? \
		(((IEEEsfp *)&(X))->s.e==0xff) :\
		(((IEEEfp *)&(X))->s.e==NAN_VAL) \
	)

/* 
#define NaN(X)	(((IEEEfp *)&(X))->s.e==NAN_VAL &&\
	((IEEEfp *)&(X))->s.f1!= 0 &&\
	((IEEEfp *)&(X))->s.f2!= 0)
 */

/* #define NaN(X)	((((IEEEfp *)&(X))->s.e==NAN_VAL &&\	*/
/* 	((IEEEfp *)&(X))->s.f1!= 0 )? ((IEEEfp *)&(X))->s.f1 : 0 )	*/

#define NaN(X)	( (I3Ed(X)->s.e==NAN_VAL)? \
	((I3Ed(X)->s.f1!= 0 )? I3Ed(X)->s.f1 : \
		((I3Ed(X)->s.f2!= 0)? I3Ed(X)->s.f2 : 0)) : 0)

#define fpNaN(X)	( (sizeof(X)==sizeof(float))? \
		((I3Edf(X)->s.e==0xff)?  I3Edf(X)->s.f : 0 ) : \
		((I3Ed(X)->s.e==NAN_VAL)? \
		((I3Ed(X)->s.f1!= 0 )? I3Ed(X)->s.f1 : \
			((I3Ed(X)->s.f2!= 0)? I3Ed(X)->s.f2 : 0)) : 0) \
	)

#define INF(X)	(((IEEEfp *)&(X))->s.e==NAN_VAL &&\
	((IEEEfp *)&(X))->s.f1== 0 &&\
	((IEEEfp *)&(X))->s.f2== 0)

#define fpINF(X)	( (sizeof(X)==sizeof(float))? \
		(((IEEEsfp *)&(X))->s.e==0xff &&\
			((IEEEsfp *)&(X))->s.f== 0 ) : \
		(((IEEEfp *)&(X))->s.e==NAN_VAL &&\
			((IEEEfp *)&(X))->s.f1== 0 &&\
			((IEEEfp *)&(X))->s.f2== 0) \
	)

#define set_NaN(X)	{IEEEfp *local_IEEEfp=(IEEEfp*)(&(X));\
	local_IEEEfp->s.s= 0; local_IEEEfp->s.e=NAN_VAL ;\
	local_IEEEfp->s.f1= 0xfffff ;\
	local_IEEEfp->s.f2= 0xffffffff;}

#define set_fpNaN(X)	if( sizeof(X) == sizeof(float) ){ \
		IEEEsfp *local_IEEEsfp=(IEEEsfp*)(&(X));\
		local_IEEEsfp->s.s= 0; local_IEEEsfp->s.e=0xff ;\
		local_IEEEsfp->s.f= 0xffffff ;\
	} \
	else{ \
		IEEEfp *local_IEEEfp=(IEEEfp*)(&(X));\
		local_IEEEfp->s.s= 0; local_IEEEfp->s.e=NAN_VAL ;\
		local_IEEEfp->s.f1= 0xfffff ;\
		local_IEEEfp->s.f2= 0xffffffff; \
	}

#define Inf(X)	((INF(X))?((((IEEEfp *)&(X))->l.high==POS_INF)?1:-1):0)

#define fpInf(X)	( (sizeof(X)==sizeof(float))? \
		(fpINF(X))?((((IEEEsfp *)&(X))->s.s)?-1:1):0 :\
		(INF(X))?((((IEEEfp *)&(X))->l.high==POS_INF)?1:-1):0 \
	)

#define set_Inf(X,S)	{IEEEfp *local_IEEEfp=(IEEEfp*)(&(X));\
	local_IEEEfp->l.high=(S>0)?POS_INF:NEG_INF ;\
	local_IEEEfp->l.low= 0x00000000 ;}

#define set_fpInf(X,S)	if( sizeof(X)==sizeof(float) ){ \
		IEEEsfp *local_IEEEsfp=(IEEEsfp*)(&(X));\
		local_IEEEsfp->s.e=0xff ;\
		local_IEEEsfp->s.s= (S>0)? 0 : 1 ;\
		local_IEEEsfp->s.f= 0 ;\
	} \
	else{ \
		IEEEfp *local_IEEEfp=(IEEEfp*)(&(X));\
		local_IEEEfp->l.high=(S>0)?POS_INF:NEG_INF ;\
		local_IEEEfp->l.low= 0x00000000 ;\
	}

#define set_INF(X)	{IEEEfp *local_IEEEfp=(IEEEfp*)(&(X));\
	local_IEEEfp->s.e=NAN_VAL ;\
	local_IEEEfp->s.f1= 0x00000 ;\
	local_IEEEfp->s.f2= 0x00000000 ;}

#define set_fpINF(X)	if( sizeof(X) == sizeof(float) ){ \
		IEEEsfp *local_IEEEsfp=(IEEEsfp*)(&(X));\
		local_IEEEsfp->s.e=0xff ;\
		local_IEEEsfp->s.s= 0 ;\
		local_IEEEsfp->s.f= 0x000000 ;\
	} \
	else{ \
		IEEEfp *local_IEEEfp=(IEEEfp*)(&(X));\
		local_IEEEfp->s.e=NAN_VAL ;\
		local_IEEEfp->s.f1= 0x00000 ;\
		local_IEEEfp->s.f2= 0x00000000 ;\
	}

#endif

DEFUN( _NaNorInf, (double *x), int);
DEFUN( _NaN, (double *x), int);
DEFUN( _INF, (double *x), int);
DEFUN( _set_NaN, (double *x), IEEEfp* );
DEFUN( _Inf, (double *x), int);
DEFUN( _set_Inf, (double *x, int sign), IEEEfp* );
DEFUN( _set_INF, (double *x), IEEEfp* );

DEFUN( *d2str,	(double x, char *format, char *buf),	char);

DEFUN( _lround, (double x), long);/* returns long with rounded value of x */
DEFUN( _round, (double x), int);
DEFUN( _dsign, (double x), int);
DEFUN( fraxion, (double x), double);       /* xxxxxxxxxxxxx.yyy => 0.yyy */
DEFUN( _sign, (int x), int);
DEFUN( set_sintab, (), int);		/* allocate & calculate sintab	*/
DEFUN( sinus, (double x, double period), double);
DEFUN( cosinus, (double x, double period), double);
DEFUN( free_sintab, (), void);          		/* must be called at end of program!!! */
DEFUN( blockwave, (double x, double period), int);	/* blockwave with ampl 1 */
DEFUN( impulse, (double x, double period, double tol), int);
                         /* impulse with ampl 1 if x/ period==  +/- tol    */
DEFUN( randomise, (), int);					/* randomise random-generators */
DEFUN( rand_choice, (double min, double max), double);
DEFUN( rand_in, (double min, double max), double);
DEFUN( uniform_rand, (double av, double stdv), double );
DEFUN( normal_rand, (double av, double stdv), double );
DEFUN( normal_in, (double min, double max), double );
DEFUN( abnormal_rand, (double av, double stdv), double );
DEFUN( Permute, (void *a, int n, int type), int);

/* Compare b against a, using precision. If precision>= 0 it
 \ represents a fraction, so that b equals a if
 \ a*(1-precision) < b < a*(1+precision)  [ a_min < b < a_max ]
 \ If precision< 0, b equals a if
 \ a - |precision|*DBL_EPSILON < b < a + |precision|*DBL_EPSILON  [ a_min < b < a_max ]
 \ with DBL_EPSILON the machine imprecision.
 \ dcmp returns 0.0 if b equals a; b - a_min (<0) if b is smaller than a
 \ and b - a_max (>0) if b larger than a.
 */
DEFUN( dcmp, (double b, double a, double precision), double);

/* calculates Prod(x-n*dx>0)	*/
DEFUN( dfac, (double x, double dx), double);

/* returns the triple root of x:	*/
DEFUN( curt, (double x), double);

DEFUN( atan3, (double dy,double dx), double); /* compute atan(dy/dx) in interval [0,2PI]	*/
DEFUN( FFPsincos, (double *cos, double arg), double);	/* compat. with FFP sincos()	*/

DEFUN( noise_add, (double p, unsigned char *pic, int x, int y), double);
							/* add p% noise to pic[ x, y];  returns actual % added	*/
DEFUN( SIN, (double x), double);
DEFUN( COS, (double x), double);		/* linear (sawtooth) approx. of sin, cos	*/
DEFUN( SINCOS, (double f, double *si, double *co), void);/* idem of sincos() ( sin & cos at once)	*/

DEFUN( phival,	(double x, double y),	double );
DEFUN( phival2,	(double x, double y),	double );
DEFUN( conv_angle,	(double phi),	double );
DEFUN( conv_angle2,	(double phi),	double );
DEFUN( mod_angle,	(double phi),	double );

#define Euclidian_SQDist(dx,dy,dz)	((dx)*(dx)+(dy)*(dy)+(dz)*(dz))
#define Euclidian_Dist(dx,dy,dz)	(sqrt(Euclidian_SQDist(dx,dy,dz)))

DEFUN( _Euclidian_SQDist, (double dx, double dy, double dz), double);
DEFUN( _Euclidian_Dist, (double dx, double dy, double dz), double);

DEFUN( SubVisAngle, (double dist, double radius), double);

extern double Units_per_Radian;
#define Gonio(fun,x)	(fun((x)/Units_per_Radian))
#define InvGonio(fun,x)	((fun(x)*Units_per_Radian))

DEFUN( Gonio_Base, (double base), double);
DEFUN( _Sin, (double x), double);
DEFUN( _Cos, (double x), double);
DEFUN( _Tan, (double x), double);
DEFUN( _ASin, (double x), double);
DEFUN( _ACos, (double x), double);
DEFUN( _ATan, (double x), double);
DEFUN( _ATan2, (double x, double y), double);	/* returns the angle to (x,y) in <-Gonio_Base/2,Gonio_Base/2]	*/
DEFUN( _Arg, (double x, double y), double);	/*    "     "    "    "   "   in [0,Gonio_Base>	*/

#define Sin(x) Gonio(sin,x)
#define Cos(x) Gonio(cos,x)
#define Tan(x) Gonio(tan,x)
#define ASin(x) InvGonio(asin,x)
#define ACos(x) InvGonio(acos,x)
#define ATan(x) InvGonio(atan,x)
#define ATan2(x,y) (atan2(x,y)*Units_per_Radian)
#define Arg(x,y) (atan3(x, y)*Units_per_Radian)

#	define sindeg(a)			sin(radians(a))
#	define cosdeg(a)			cos(radians(a))
#	define tandeg(a)			tan(radians(a))
#	define asindeg(a)			degrees(asin(a))
#	define acosdeg(a)			degrees(acos(a))
#	define atandeg(a)			degrees(atan(a))

extern int sintab_set;

extern double drand48();
#define drand()	drand48()

/* ****************** Statistics functions ********************* */

/* tmin,tmax hold the global minimum and maximum; 
 * min,max hold the numerical (i.e. are not NaN or Inf) min and max.
 */
#define MinMax(type, n, array, tmin, tmax, min, max, reset)\
	{type a= (array);long i=0;double A;\
		if(reset){\
			A=(double)*a;tmin=A;tmax=A;\
			while(NaNorInf(A)&&i<n){a++;i++;A=(double)*a;};\
			if(!NaNorInf(A)){\
				min=A;\
				max=A;\
			}\
			a=(array);\
		}\
		for( i= 0; i< n; i++, a++, A=(double)*a ){\
			A=(double) *a; if( !NaNorInf(A) ){\
				if( A> max)\
					max=A;\
				else if( A< min)\
					min= A;\
			}\
			if( A> tmax)\
				tmax=A;\
			else if( A< tmin)\
				tmin= A;\
		}\
	}

#define MinMax_SS_threshold(type, n, array, tmin, tmax, min, max, tSS, SS, threshold, comparedto, reset)\
	{type a= (array);long i=0;double A;\
		if(reset){\
			A=(double)*a;tmin=A;tmax=A;\
			while(NaNorInf(A)&&i<n){a++;i++;A=(double)*a;};\
			if(!NaNorInf(A)){\
				min=A;\
				max=A;\
			}\
			SS_Reset_(SS);\
			a=(array);\
		}\
		for( i= 0; i< n; i++, a++){\
			A=(double)*a;\
			if( !NaNorInf(A) ){\
				if( A> max)\
					max=A;\
				else if( A< min)\
					min= A;\
				if( A comparedto (threshold) )\
					SS_Add_Data_(SS, 1, A, 1.0);\
			}\
			if( A> tmax)\
				tmax=A;\
			else if( A< tmin)\
				tmin= A;\
			if( A comparedto (threshold) )\
				SS_Add_Data_(tSS, 1, A, 1.0);\
		}\
	}

#define SS_SKEW

typedef struct simplestats{
	unsigned long count, last_count, takes;
	double weight_sum, last_weight, sum, sum_sqr, sum_cub, last_item;
	double min, max, mean, stdv, skew;
} SimpleStats;

/* Simple Statistics on cyclic data (angles). Gonio_Base specifies the
 * range; data should be given within -0.5*Gonio_Base,+0.5*Gonio_Base
 * (i.o.w.) the singularity should be at 0 and in the middle of the range).
 */
typedef struct simpleanglestats{
	long pos_count, neg_count, last_count, takes;
	double pos_weight_sum, neg_weight_sum, last_weight, pos_sum, neg_sum, sum_sqr, last_item;
	double min, max, mean, stdv;
	double Gonio_Base;
} SimpleAngleStats;

/* An empty SimpleStats structure:	*/
extern SimpleStats EmptySimpleStats;
extern SimpleAngleStats EmptySimpleAngleStats;

DEFUN( SS_Add_Data, (SimpleStats *a, long count, double sum, double weight), SimpleStats *);
DEFUN( SS_Add_Squared_Data, (SimpleStats *a, long count, double sumsq, double weight), SimpleStats *);
DEFUN( SS_Add, (SimpleStats *a, SimpleStats *b), SimpleStats *);
DEFUN( SS_Sum, (SimpleStats *a, SimpleStats *b), SimpleStats *);
#define SS_Sum_(a,b)	SS_Sum(&(a),&(b))
DEFUN( SS_sprint, (char *buffer, char *format, char *sep, double min_err, SimpleStats *a), char *);
#define SS_sprint_(buffer,format,sep,mi,a)	SS_sprint(buffer,format,sep,mi,&(a))
DEFUN( SS_sprint_full, (char *buffer, char *format, char *sep, double min_err, SimpleStats *a), char *);
#define SS_sprint_full_(buffer,format,sep,mi,a)	SS_sprint_full(buffer,format,sep,mi,&(a))

DEFUN( SAS_Add_Data, (SimpleAngleStats *a, long count, double sum, double weight), SimpleAngleStats *);
DEFUN( SAS_Add_Squared_Data, (SimpleAngleStats *a, long count, double sum, double sumsq, double weight), SimpleAngleStats *);
DEFUN( SAS_Add, (SimpleAngleStats *a, SimpleAngleStats *b), SimpleAngleStats *);
DEFUN( SAS_Sum, (SimpleAngleStats *a, SimpleAngleStats *b), SimpleAngleStats *);
#define SAS_Sum_(a,b)	SAS_Sum(&(a),&(b))
DEFUN( SAS_sprint, (char *buffer, char *format, char *sep, double min_err, SimpleAngleStats *a), char *);
#define SAS_sprint_(buffer,format,sep,mi,a)	SAS_sprint(buffer,format,sep,mi,&(a))
DEFUN( SAS_sprint_full, (char *buffer, char *format, char *sep, double min_err, SimpleAngleStats *a), char *);
#define SAS_sprint_full_(buffer,format,sep,mi,a)	SAS_sprint_full(buffer,format,sep,mi,&(a))

#define SS_Reset(a)	{(*a)=EmptySimpleStats;}
#define SS_Reset_(a)	{(a)=EmptySimpleStats;}

#define SAS_Reset(a)	{if((a)->Gonio_Base){EmptySimpleAngleStats.Gonio_Base=(a)->Gonio_Base;}(*a)=EmptySimpleAngleStats;}
#define SAS_Reset_(a)	{if((a).Gonio_Base){EmptySimpleAngleStats.Gonio_Base=(a).Gonio_Base;}(a)=EmptySimpleAngleStats;}

#ifdef SS_SKEW
#	define SS_Add_Data_(a,cnt,sm,wght)	{double SS_Add_Data_sum= (double)(sm);\
	SimpleStats *SimpleStatsLocalPtr=&(a);\
	if(!SimpleStatsLocalPtr->count){\
		SimpleStatsLocalPtr->min=SS_Add_Data_sum;\
		SimpleStatsLocalPtr->max=SS_Add_Data_sum;\
	}\
	else if( SS_Add_Data_sum< SimpleStatsLocalPtr->min){\
		SimpleStatsLocalPtr->min=SS_Add_Data_sum;\
	}\
	else if( SS_Add_Data_sum> SimpleStatsLocalPtr->max){\
		SimpleStatsLocalPtr->max=SS_Add_Data_sum;\
	}\
	SimpleStatsLocalPtr->count+=(SimpleStatsLocalPtr->last_count=(long)(cnt));\
	SimpleStatsLocalPtr->sum+=(wght)*(SimpleStatsLocalPtr->last_item=SS_Add_Data_sum);\
	SimpleStatsLocalPtr->sum_sqr+=(wght)*SS_Add_Data_sum*SS_Add_Data_sum;\
	SimpleStatsLocalPtr->sum_cub+=(wght)*SS_Add_Data_sum*SS_Add_Data_sum*SS_Add_Data_sum;\
	SimpleStatsLocalPtr->weight_sum+=(SimpleStatsLocalPtr->last_weight=(double)(wght));\
}
#else
#	define SS_Add_Data_(a,cnt,sm,wght)	{double SS_Add_Data_sum= (double)(sm);\
	SimpleStats *SimpleStatsLocalPtr=&(a);\
	if(!SimpleStatsLocalPtr->count){\
		SimpleStatsLocalPtr->min=SS_Add_Data_sum;\
		SimpleStatsLocalPtr->max=SS_Add_Data_sum;\
	}\
	else if( SS_Add_Data_sum< SimpleStatsLocalPtr->min){\
		SimpleStatsLocalPtr->min=SS_Add_Data_sum;\
	}\
	else if( SS_Add_Data_sum> SimpleStatsLocalPtr->max){\
		SimpleStatsLocalPtr->max=SS_Add_Data_sum;\
	}\
	SimpleStatsLocalPtr->count+=(SimpleStatsLocalPtr->last_count=(long)(cnt));\
	SimpleStatsLocalPtr->sum+=(wght)*(SimpleStatsLocalPtr->last_item=SS_Add_Data_sum);\
	SimpleStatsLocalPtr->sum_sqr+=(wght)*SS_Add_Data_sum*SS_Add_Data_sum;\
	SimpleStatsLocalPtr->weight_sum+=(SimpleStatsLocalPtr->last_weight=(double)(wght));\
}
#endif

#define SS_Add_Data_Array_(SS,data,cnt,weight)	{int i;\
	for(i=0;i<(cnt);i++){\
		SS_Add_Data_((SS),1,(data)[i],(weight));\
	}\
}\

#ifdef SS_SKEW
#	define SS_Add_Squared_Data_(a,cnt,smsq,wght)	{double SS_Add_Data_sumsq= (double)(smsq),\
	SS_Add_Data_sum=sqrt(SS_Add_Data_sumsq);\
	SimpleStats *SimpleStatsLocalPtr=&(a);\
	if(!SimpleStatsLocalPtr->count){\
		SimpleStatsLocalPtr->min=SS_Add_Data_sum;\
		SimpleStatsLocalPtr->max=SS_Add_Data_sum;\
	}\
	else if( SS_Add_Data_sum< SimpleStatsLocalPtr->min){\
		SimpleStatsLocalPtr->min=SS_Add_Data_sum;\
	}\
	else if( SS_Add_Data_sum> SimpleStatsLocalPtr->max){\
		SimpleStatsLocalPtr->max=SS_Add_Data_sum;\
	}\
	SimpleStatsLocalPtr->count+=(SimpleStatsLocalPtr->last_count=(long)(cnt));\
	SimpleStatsLocalPtr->sum+=(wght)*(SimpleStatsLocalPtr->last_item=SS_Add_Data_sum);\
	SimpleStatsLocalPtr->sum_sqr+=(wght)*SS_Add_Data_sumsq;\
	SimpleStatsLocalPtr->sum_cub+=(wght)*SS_Add_Data_sum*SS_Add_Data_sumsq;\
	SimpleStatsLocalPtr->weight_sum+=(SimpleStatsLocalPtr->last_weight=(double)(wght));\
}
#else
#	define SS_Add_Squared_Data_(a,cnt,smsq,wght)	{double SS_Add_Data_sumsq= (double)(smsq),\
	SS_Add_Data_sum=sqrt(SS_Add_Data_sumsq);\
	SimpleStats *SimpleStatsLocalPtr=&(a);\
	if(!SimpleStatsLocalPtr->count){\
		SimpleStatsLocalPtr->min=SS_Add_Data_sum;\
		SimpleStatsLocalPtr->max=SS_Add_Data_sum;\
	}\
	else if( SS_Add_Data_sum< SimpleStatsLocalPtr->min){\
		SimpleStatsLocalPtr->min=SS_Add_Data_sum;\
	}\
	else if( SS_Add_Data_sum> SimpleStatsLocalPtr->max){\
		SimpleStatsLocalPtr->max=SS_Add_Data_sum;\
	}\
	SimpleStatsLocalPtr->count+=(SimpleStatsLocalPtr->last_count=(long)(cnt));\
	SimpleStatsLocalPtr->sum+=(wght)*(SimpleStatsLocalPtr->last_item=SS_Add_Data_sum);\
	SimpleStatsLocalPtr->sum_sqr+=(wght)*SS_Add_Data_sumsq;\
	SimpleStatsLocalPtr->weight_sum+=(SimpleStatsLocalPtr->last_weight=(double)(wght));\
}
#endif

#define SS_Add_Squared_Data_Array_(SS,data,cnt,weight)	{int i;\
	for(i=0;i<(cnt);i++){\
		SS_Add_Squared_Data_((SS),1,(data)[i],(weight));\
	}\
}\

#ifdef SS_SKEW
#	define SS_Add_(a,b)	{SimpleStats *SimpleStatsLocalPtr=&(a);\
	if(!SimpleStatsLocalPtr->count){\
		SimpleStatsLocalPtr->min=(b).min;\
		SimpleStatsLocalPtr->max=(b).max;\
	}\
	else if( (b).min< SimpleStatsLocalPtr->min){\
		SimpleStatsLocalPtr->min=(b).min;\
	}\
	else if( (b).max> SimpleStatsLocalPtr->max){\
		SimpleStatsLocalPtr->max=(b).max;\
	}\
	SimpleStatsLocalPtr->count+= (SimpleStatsLocalPtr->last_count=(b).count);\
	SimpleStatsLocalPtr->weight_sum+= (SimpleStatsLocalPtr->last_weight=(b).weight_sum);\
	SimpleStatsLocalPtr->sum+= (SimpleStatsLocalPtr->last_item=(b).sum);\
	SimpleStatsLocalPtr->sum_sqr+= (b).sum_sqr;\
	SimpleStatsLocalPtr->sum_cub+= (b).sum_cub;\
	(SimpleStatsLocalPtr->takes)++;\
}
#else
#	define SS_Add_(a,b)	{SimpleStats *SimpleStatsLocalPtr=&(a);\
	if(!SimpleStatsLocalPtr->count){\
		SimpleStatsLocalPtr->min=(b).min;\
		SimpleStatsLocalPtr->max=(b).max;\
	}\
	else if( (b).min< SimpleStatsLocalPtr->min){\
		SimpleStatsLocalPtr->min=(b).min;\
	}\
	else if( (b).max> SimpleStatsLocalPtr->max){\
		SimpleStatsLocalPtr->max=(b).max;\
	}\
	SimpleStatsLocalPtr->count+= (SimpleStatsLocalPtr->last_count=(b).count);\
	SimpleStatsLocalPtr->weight_sum+= (SimpleStatsLocalPtr->last_weight=(b).weight_sum);\
	SimpleStatsLocalPtr->sum+= (SimpleStatsLocalPtr->last_item=(b).sum);\
	SimpleStatsLocalPtr->sum_sqr+= (b).sum_sqr;\
	(SimpleStatsLocalPtr->takes)++;\
}
#endif

#define SAS_Add_Data_(a,cnt,sm,wght)	{double SAS_Add_Data_sum=(double)(sm);\
	SimpleAngleStats *SimpleAngleStatsLocalPtr=&(a);\
	if(!(SimpleAngleStatsLocalPtr->pos_count+SimpleAngleStatsLocalPtr->neg_count)){\
		SimpleAngleStatsLocalPtr->min=SAS_Add_Data_sum;\
		SimpleAngleStatsLocalPtr->max=SAS_Add_Data_sum;\
	}\
	else if( SAS_Add_Data_sum< SimpleAngleStatsLocalPtr->min){\
		SimpleAngleStatsLocalPtr->min=SAS_Add_Data_sum;\
	}\
	else if( SAS_Add_Data_sum> SimpleAngleStatsLocalPtr->max){\
		SimpleAngleStatsLocalPtr->max=SAS_Add_Data_sum;\
	}\
	if( SAS_Add_Data_sum>=0)\
		{SimpleAngleStatsLocalPtr->pos_sum+=(wght)*(SimpleAngleStatsLocalPtr->last_item=SAS_Add_Data_sum);\
		 SimpleAngleStatsLocalPtr->pos_weight_sum+=(SimpleAngleStatsLocalPtr->last_weight=(double)(wght));\
		 SimpleAngleStatsLocalPtr->pos_count+=(SimpleAngleStatsLocalPtr->last_count=(long)(cnt));\
	}\
	else\
		{SimpleAngleStatsLocalPtr->neg_sum+=(wght)*(SimpleAngleStatsLocalPtr->last_item=SAS_Add_Data_sum);\
		 SimpleAngleStatsLocalPtr->neg_weight_sum+=(SimpleAngleStatsLocalPtr->last_weight=(double)(wght));\
		 SimpleAngleStatsLocalPtr->neg_count+=(SimpleAngleStatsLocalPtr->last_count=(long)(cnt));\
	}\
	SimpleAngleStatsLocalPtr->sum_sqr+=(wght)*SAS_Add_Data_sum*SAS_Add_Data_sum;\
}

#define SAS_Add_Squared_Data_(a,cnt,sm,smsq,wght)	{double SAS_Add_Data_sumsq=(double)(smsq),\
	SAS_Add_Data_sum=(double)(sm);\
	SimpleAngleStats *SimpleAngleStatsLocalPtr=&(a);\
	if(!(SimpleAngleStatsLocalPtr->pos_count+SimpleAngleStatsLocalPtr->neg_count)){\
		SimpleAngleStatsLocalPtr->min=SAS_Add_Data_sum;\
		SimpleAngleStatsLocalPtr->max=SAS_Add_Data_sum;\
	}\
	else if( SAS_Add_Data_sum< SimpleAngleStatsLocalPtr->min){\
		SimpleAngleStatsLocalPtr->min=SAS_Add_Data_sum;\
	}\
	else if( SAS_Add_Data_sum> SimpleAngleStatsLocalPtr->max){\
		SimpleAngleStatsLocalPtr->max=SAS_Add_Data_sum;\
	}\
	if( SAS_Add_Data_sum>=0)\
		{SimpleAngleStatsLocalPtr->pos_sum+=(wght)*(SimpleAngleStatsLocalPtr->last_item=SAS_Add_Data_sum);\
		 SimpleAngleStatsLocalPtr->pos_weight_sum+=(SimpleAngleStatsLocalPtr->last_weight=(double)(wght));\
		 SimpleAngleStatsLocalPtr->pos_count+=(SimpleAngleStatsLocalPtr->last_count=(long)(cnt));\
	}\
	else\
		{SimpleAngleStatsLocalPtr->neg_sum+=(wght)*(SimpleAngleStatsLocalPtr->last_item=SAS_Add_Data_sum);\
		 SimpleAngleStatsLocalPtr->neg_weight_sum+=(SimpleAngleStatsLocalPtr->last_weight=(double)(wght));\
		 SimpleAngleStatsLocalPtr->neg_count+=(SimpleAngleStatsLocalPtr->last_count=(long)(cnt));\
	}\
	SimpleAngleStatsLocalPtr->sum_sqr+=(wght)*SAS_Add_Data_sumsq;\
}

#define SAS_Add_(a,b)	{SimpleAngleStats *SimpleAngleStatsLocalPtr=&(a);\
	if(SimpleAngleStatsLocalPtr->Gonio_Base==(b).Gonio_Base){\
		if(!(SimpleAngleStatsLocalPtr->pos_count+SimpleAngleStatsLocalPtr->neg_count)){\
			SimpleAngleStatsLocalPtr->min=(b).min;\
			SimpleAngleStatsLocalPtr->max=(b).max;\
		}\
		else if( (b).min< SimpleAngleStatsLocalPtr->min){\
			SimpleAngleStatsLocalPtr->min=(b).min;\
		}\
		else if( (b).max> SimpleAngleStatsLocalPtr->max){\
			SimpleAngleStatsLocalPtr->max=(b).max;\
		}\
		SimpleAngleStatsLocalPtr->pos_count+=(b).pos_count;\
		SimpleAngleStatsLocalPtr->neg_count+=(b).neg_count;\
		SimpleAngleStatsLocalPtr->last_count+=(b).pos_count+(b).neg_count;\
		SimpleAngleStatsLocalPtr->pos_weight_sum+=(b).pos_weight_sum;\
		SimpleAngleStatsLocalPtr->neg_weight_sum+=(b).neg_weight_sum;\
		SimpleAngleStatsLocalPtr->last_weight=(b).last_weight;\
		SimpleAngleStatsLocalPtr->pos_sum+=(b).pos_sum;\
		SimpleAngleStatsLocalPtr->neg_sum+=(b).neg_sum;\
		SimpleAngleStatsLocalPtr->last_item=(b).last_item;\
		SimpleAngleStatsLocalPtr->sum_sqr+=(b).sum_sqr;}\
		(SimpleAngleStatsLocalPtr->takes)++;\
	}

DEFUN( SS_Mean, (SimpleStats *SS), double);
DEFUN( SS_Mean_Div, (SimpleStats *SS,SimpleStats *SSb), double);
#define SS_Mean_Div_(a,b)	SS_Mean_Div(&(a),&(b))
DEFUN( SS_St_Dev, (SimpleStats *SS), double);
#define SS_Mean_(ss)	SS_Mean(&(ss))
#define SS_St_Dev_(ss)	SS_St_Dev(&(ss))

#ifdef SS_SKEW
	DEFUN( SS_Skew, (SimpleStats *SS), double);
#	define SS_Skew_(a)	SS_Skew(&(a))
#endif

DEFUN( SAS_Mean, (SimpleAngleStats *SS), double);
DEFUN( SAS_St_Dev, (SimpleAngleStats *SS), double);
#define SAS_Mean_(ss)	SAS_Mean(&(ss))
#define SAS_St_Dev_(ss)	SAS_St_Dev(&(ss))

typedef struct leaky_int_par{
	double *tau, *delta_t, memory;
	double DeltaT, *state;
} Leaky_Int_Par;

DEFUN( *Init_Leaky_Int_Par, ( Leaky_Int_Par *lip), Leaky_Int_Par);
DEFUN( Leaky_Int_Response, ( Leaky_Int_Par *lip, double *state, double input, double gain), double);
DEFUN( Normalised_Leaky_Int_Response, ( Leaky_Int_Par *lip, double *state, double input), double);

/* Defaults for 'period': */

#define DEG 360.0
#define RAD 6.2831853072
#define GRA 400.0
#define twoPI 6.2831853072
#define hPI 1.5707963278
#ifndef PI
#	define PI M_PI
#endif

#define DOUBLES 1
#define INTS 2
#define LONGS 3

#ifdef sign
	#undef sign
#endif

#ifndef NO_INLINE
#define lround(x)	((x>=0)?((long)(x+0.5)):((long)(x-0.5)))
#define round(x)	((x>=0)?((int)(x+0.5)):((int)(x-0.5)))
#define sign(x)		((x<0)?-1:1)
#define dsign(x)	((x<0)?-1:1)
#else
#define round(x)	_round(x)
#define lround(x)	_lround(x)
#define sign(x)		_sign(x)
#define dsign(x)	_dsign(x)
#endif


#ifdef __cplusplus
	}
#endif
#endif
