/* Header file for SimpleStats functions.
 \ 20050807: this module will also hold some general simple statistics functions,
 \ unrelated to the SimpleStats and SimpleAngleStats functionality.
 */
#ifndef SS_H

#include "defun.h"

typedef struct SS_Sample{
	unsigned long id;
	double value, weight;
} SS_Sample;

typedef struct Range{
	double min, max;
} Range;

typedef struct SimpleStats{
	long count, last_count, takes, Nvalues, curV;
	double weight_sum, last_weight, sum, sum_sqr, sum_cub, last_item;
	  /* values describing the set. pos_min is the minimal, positive number
	   \ that is not a NaN or Inf. The mean, stdv and skew values are calculated
	   \ by calling the appropriate routines, the min/max values are calculated online.
	   */
	double min, pos_min, max, mean, stdv, skew, median, adev, quantile[2];
	  /* numerical (i.e. not NaN or Inf) values:	*/
	double nr_min, nr_max;
	SS_Sample *sample;
	  // 20090626: changed to char from int!
	char exact, meaned, medianed, stdved, adeved, quantiled;
} SimpleStats;
/* An empty SimpleStats structure:	*/
extern SimpleStats EmptySimpleStats;

/* return the mean of a dataset as described by the
 * SimpleStats structure *SS. Note that the SS_ routines
 * expect that both the count and the weight_sum fields are
 * set.
 */
extern double SS_Mean( SimpleStats *SS);
extern double SS_Median( SimpleStats *SS);

/* return the standard deviation of a dataset as described by the
 * SimpleStats structure *SS. The same considerations as given under
 * st_dev and SS_Mean hold.
 */
extern double SS_St_Dev( SimpleStats *SS);
extern double SS_Skew( SimpleStats *SS);
extern double SS_St_Err( SimpleStats *SS);
extern double SS_ADev( SimpleStats *SS);
extern double SS_Quantile( SimpleStats *SS, double prob);

extern SimpleStats *SS_Copy( SimpleStats *a, SimpleStats *b );
extern int SS_Sort_Samples( SS_Sample *a, SS_Sample *b);

extern char *SS_sprint_full( char *buffer, char *format, char *sep, double min_err, SimpleStats *a);
extern char *SS_sprint( char *buffer, char *format, char *sep, double min_err, SimpleStats *a);

extern double TTest( double mean1, double var1, int n1, double mean2, double var2, int n2, double *t);
extern double TTest_uneq( double mean1, double var1, int n1, double mean2, double var2, int n2, double *t);
extern double FTest( double var1, int n1, double var2, int n2, double *f);

extern double SS_TTest( SimpleStats *d1, SimpleStats *d2, double *t);
extern double SS_TTest_uneq( SimpleStats *d1, SimpleStats *d2, double *t);
extern double SS_TTest_paired( SimpleStats *d1, SimpleStats *d2, double *t);
extern double SS_FTest( SimpleStats *d1, SimpleStats *d2, double *f);

#define SS_Reset_(a)	{int nv=(a).Nvalues,exact=(a).exact;\
	SS_Sample *s=(a).sample;\
	EmptySimpleStats.curV=0;\
	(a)=EmptySimpleStats;\
	(a).meaned=False;(a).stdved=False;\
	(a).Nvalues=nv;\
	(a).exact=exact;\
	(a).sample= s;}

#define SS_Reset(a)	SS_Reset_((*a))

#define SS_Init_(a)	{ EmptySimpleStats.curV=0;\
	(a)=EmptySimpleStats;\
	(a).meaned=False;(a).stdved=False;\
	(a).Nvalues=0;\
	(a).exact=0;\
	(a).sample= NULL;}

#define SS_Init(a)	SS_Init_((*a))

#define SS_Add_Data__(a,cnt,sm,wght)	{double SS_Add_Data_sum= (double)(sm);\
	SimpleStats *SimpleStatsLocalPtr=&(a);\
	SimpleStatsLocalPtr->count+=(SimpleStatsLocalPtr->last_count=(long)(cnt));\
	SimpleStatsLocalPtr->sum+=(wght)*(SimpleStatsLocalPtr->last_item=SS_Add_Data_sum);\
	SimpleStatsLocalPtr->sum_sqr+=(wght)*SS_Add_Data_sum*SS_Add_Data_sum;\
	SimpleStatsLocalPtr->sum_cub+=(wght)*SS_Add_Data_sum*SS_Add_Data_sum*SS_Add_Data_sum;\
	SimpleStatsLocalPtr->weight_sum+=(SimpleStatsLocalPtr->last_weight=(double)(wght));\
}

DEFUN( SS_Add_Data, (SimpleStats *a, long count, double sum, double weight), SimpleStats *);

DEFUN( SS_Pop_Data, (SimpleStats *a, long count, long pos, int update ), long );

#define SS_Add_Data_(a,cnt,sm,wght)	SS_Add_Data(&(a),cnt,sm,wght)
#define SS_Pop_Data_(a,cnt,pos,update)	SS_Pop_Data(&(a),(cnt),(pos),(update))

#define SS_Mean_(ss)	SS_Mean(&(ss))
#define SS_Median_(ss)	SS_Median(&(ss))
#define SS_St_Dev_(ss)	SS_St_Dev(&(ss))
#define SS_ADev_(ss)	SS_ADev(&(ss))
#define SS_Skew_(ss)	SS_Skew(&(ss))
#define SS_St_Err_(ss)	SS_St_Err(&(ss))

extern double *SS_exact, *SS_Ignore_NaN, *SS_Empty_Value;
extern int SS_valid_exact( SimpleStats *ss );

/* Simple Statistics on cyclic data (angles). Gonio_Base specifies the
 * range; data should be given within -0.5*Gonio_Base,+0.5*Gonio_Base
 * (i.o.w.) the singularity should be at 0 and in the middle of the range).
 \ Set Gonio_Offset e.g. to 90 (when Gonio_Base==360) when 0 degrees must
 \ point upwards; to -90 to have it point downwards; to 0 (default) to have it
 \ point rightwards.
 */
typedef struct SimpleAngleStats{
	long pos_count, neg_count, last_count, takes, Nvalues, curV;
	double pos_weight_sum, neg_weight_sum, last_weight, pos_sum, neg_sum, sum_sqr, last_item;
	double min, max, mean, stdv;
	double Gonio_Base, Gonio_Offset, Units_per_Radian;
	SS_Sample *sample;
	Boolean exact, meaned, stdved;
} SimpleAngleStats;

extern SimpleAngleStats EmptySimpleAngleStats;
#define SAS_Reset_(a)	{int nv=(a).Nvalues,exact=(a).exact;\
	SS_Sample *s=(a).sample;\
	if((a).Gonio_Base){\
		EmptySimpleAngleStats.Units_per_Radian=(EmptySimpleAngleStats.Gonio_Base=(a).Gonio_Base)/(2*M_PI);\
	}\
	EmptySimpleAngleStats.Gonio_Offset=(a).Gonio_Offset;\
	EmptySimpleAngleStats.curV=0;\
	(a)=EmptySimpleAngleStats;\
	(a).meaned=False;(a).stdved=False;\
	(a).Nvalues=nv;\
	(a).exact=exact;\
	(a).sample=s;}
#define SAS_Reset(a)	SAS_Reset_(*(a))

#define SAS_Init_(a)	{ EmptySimpleAngleStats.curV=0;\
	(a)=EmptySimpleAngleStats;\
	(a).meaned=False;(a).stdved=False;\
	(a).Nvalues=0;\
	(a).exact=0;\
	(a).sample=NULL;}
#define SAS_Init(a)	SAS_Init_(*(a))

DEFUN( SAS_Add_Data_angsincos, (SimpleAngleStats *a, long count, double sum, double sinval, double cosval, double weight), SimpleAngleStats *);
DEFUN( SAS_Add_Data, (SimpleAngleStats *a, long count, double sum, double weight, int convert), SimpleAngleStats *);

DEFUN( conv_angle_, (double angle, double radix), double);

extern double *SAS_converts_angle, *SAS_exact, *SAS_Ignore_NaN;
extern int SAS_valid_exact( SimpleAngleStats *sas );

#define SAS_Add_Data_(a,cnt,sm,wght,cnv)	SAS_Add_Data(&(a),cnt,sm,wght,cnv)

extern SimpleAngleStats *SAS_Copy( SimpleAngleStats *a, SimpleAngleStats *b );

DEFUN( SAS_Mean, (SimpleAngleStats *SS), double);
DEFUN( SAS_St_Dev, (SimpleAngleStats *SS), double);
#define SAS_Mean_(ss)	SAS_Mean(&(ss))
#define SAS_St_Dev_(ss)	SAS_St_Dev(&(ss))

extern char *SAS_sprint_full( char *buffer, char *format, char *sep, double min_err, SimpleAngleStats *a);
extern char *SAS_sprint( char *buffer, char *format, char *sep, double min_err, SimpleAngleStats *a);

typedef struct NormDist{
	double av, stdv, gset;
	int iset;
} NormDist;


/* 20050107: include NaN.h and/or ascanf.h to have the macros available for Check_Ignore_NaN().
 \ If ever this causes problems, that function will have to be defined elsewhere. For now,
 \ defining it here allows it to be inlined in all modules.
 */
#ifndef NaNorInf
#	include "NaN.h"
#endif
#ifndef ASCANF_FALSE
#	include "ascanf.h"
#endif

/* 20050107: Mostly (to be) used in SS_Add_Data, with *SS_Ignore_NaN: */
#ifdef __GNUC__
inline
#endif
static int Check_Ignore_NaN( double flag, double value )
{ int r;
	if( ASCANF_FALSE(flag)
		|| (flag== 2 && !NaNorInf(value) )
		|| (flag!= 2 && !NaN(value))
	){
		r= True;
	}
	else{
		r= False;
	}
	return(r);
}

/* 20050807: find the median of the specified array. Accepts a working buffer
 \ for sorting purposes in sortbuffer, which is assumed to be the same size as values.
 \ If sortbuffer==values, sorting is done directly in values. If sortbuffer==NULL,
 \ it is assumed values is already sorted.
 */
extern double find_Median( double *values, unsigned long N, double *sortbuffer );

extern double lm_fit(
	double *x, double *y, double *stdv, unsigned long N,	/* the N x,y values and optionally the N standard deviations */
	double *slope, double *icept,					/* return values for lm_fit(x,y) = slope * x + icept */
	double *Pslope, double *Picept, double *goodness	/* optionally return the fit probabilities and the goodness-of-fit (1 if stdv==NULL) */
);
extern double rlm_fit(
	double *x, double *y, unsigned long N,	/* x, y, N */
	double *slope, double *icept			/* return slope and intercept values */
);

#define SS_H
#endif
