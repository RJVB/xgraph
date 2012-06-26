/* 20020504: Simple statistics routines */

#include "config.h"
IDENTIFY( "Simple Statistics routines" );

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <unistd.h>
#include <sys/param.h>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <float.h>

#include "cpu.h"
#include "xgALLOCA.h"

/* #include "Macros.h"	*/

#include "xgout.h"
#include "xgraph.h"
#include "new_ps.h"
#include "xtb/xtb.h"

#include "NaN.h"

#include "xgraph.h"
#include "fdecl.h"

  /* get the ascanf definitions:	*/
#include "ascanf.h"
#include "ascanfc-table.h"

#include "Sinc.h"
#include "SS.h"

#if defined(__SSE4_1__) || defined(__SSE4_2__)
#	define USE_SSE4
#	define SSE_MATHFUN_WITH_CODE
#	include "sse_mathfun/sse_mathfun.h"
#endif

extern char *index();

extern int debugFlag, debugLevel;
extern FILE *StdErr;


SimpleStats EmptySimpleStats= {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

extern double *SS_exact;

#define nsqrt(x)	(x<0.0 ? 0.0 : sqrt(x))

#undef SQR
#define SQR(x)	((x)?((double)(x)*(x)):0.0)

extern double *SS_Empty_Value;

/* return the mean of a dataset as described by the
 * SimpleStats structure *SS. Note that the SS_ routines
 * expect that both the count and the weight_sum fields are
 * set.
 */
double SS_Mean( SimpleStats *SS)
{  double weight_sum;
	if( SS->count && (weight_sum= SS->weight_sum) ){
		SS->meaned= True;
		return( SS->mean= SS->sum/ weight_sum);
	}
	else{
		SS->meaned= False;
		return( SS->mean= *SS_Empty_Value);
	}
}

int SS_Sort_Samples( SS_Sample *a, SS_Sample *b)
{ double diff= a->value* a->weight - b->value * b->weight;
  int ret;
	if( diff< 0 ){
		ret= ( -1 );
	}
	else if( diff> 0 ){
		ret= ( 1 );
	}
	else{
		ret= ( 0 );
	}
	return( ret );
}


  /* Lowlevel routine that calculates the median from an SS, given the SS_Sample array
   \ containing the samples (!) and that may be sorted (!).
   \ No checking is done; this should be handled by the higher-level routine!
   */
#ifdef __GNUC__
inline
#endif
static double _SS_Median( SimpleStats *SS, SS_Sample *samples )
{ int N= SS->count/2;
	qsort( samples, SS->curV, sizeof(SS_Sample), (void*) SS_Sort_Samples );
	if( N * 2 == SS->count ){
	  /* even */
		SS->median= ( samples[N-1].value* samples[N-1].weight + samples[N].value* samples[N].weight)/
			(samples[N-1].weight + samples[N].weight);
	}
	else{
		N= (SS->count+ 1)/ 2;
		SS->median= samples[N-1].value* samples[N-1].weight;
	}
	SS->medianed= True;
	return( SS->median );
}

double SS_Median( SimpleStats *SS)
{
	if( SS->count && SS->curV ){
	    /* 20020612: need to sort into a copy, not in the original!!! */
	  ALLOCA( samples, SS_Sample, SS->curV, slen );
		memcpy( samples, SS->sample, SS->curV* sizeof(SS_Sample) );
		return( _SS_Median(SS, samples ) );
	}
	else{
		SS->medianed= False;
		return( SS->median= *SS_Empty_Value );
	}
}

double SS_weight_scaler= 1.0;

/* return the standard deviation of a dataset as described by the
 * SimpleStats structure *SS. The same considerations as given under
 * st_dev and SS_Mean hold.
 */
double SS_St_Dev( SimpleStats *SS)
{	double sqrt(), stdv;
   long count= SS->count;
   double sum= SS->sum, weight_sum= SS->weight_sum,
	sum_sqr= SS->sum_sqr, f= 1.0;

	if( count== 1L ){
		SS->stdved= True;
		return( SS->stdv= 0.0);
	}
	if( count <= 0 || weight_sum<= 0 ){
		SS->stdved= False;
		return( SS->stdv= -1.0);
	}
  /* If the sum of the weight is smaller than one, we multiply all weights with
   \ a common 10-fold, such that their sum becomes larger than 1. This influences the
   \ result (somewhat), so a) the situation should be avoided and b) the factor used
   \ should maybe have a different value.
	if( weight_sum<= 1.0 ){
		while( f* weight_sum<= 1.0 ){
			f*= 10;
		}
		sum_sqr*= f;
		sum*= f;
		weight_sum*= f;
	}
	SS_weight_scaler= f;
   */
	  /* Better still: scale for the average weight. This way, the standard deviation
	   \ of a range becomes independent of the average weight, i.e. the stdev of a range
	   \ is invariant of a fixed weight applied to all values.
	   */
	SS_weight_scaler= f= count/ weight_sum;
	weight_sum= (double) count;	/* weight_sum* f== count	*/
	if( SS->curV ){
	  /* do the calculation with the corrected 2pass formula	*/
	  int i;
#ifdef DEBUG
		sum_sqr*= f;
		sum*= f;
		if( (stdv= (sum_sqr- ((sum* sum)/ weight_sum) )/ (weight_sum - 1.0) )>= 0.0 ){
			stdv= sqrt( stdv);
		}
		else{
			stdv= -1.0;
		}
#endif
		if( !SS->meaned ){
			SS_Mean(SS);
		}
		sum_sqr= 0;
		sum= 0;
		for( i= 0; i< SS->curV; i++ ){
		  double delta= SS->sample[i].value- SS->mean,
				fwd = f * SS->sample[i].weight * delta;
			sum+= fwd;
			sum_sqr+= fwd * delta;
		}
		SS->stdv= sqrt( (sum_sqr- (sum* sum)/ weight_sum)/ (weight_sum- 1) );
#ifdef DEBUG
		if( pragma_unlikely(debugFlag || ascanf_verbose) && !SS->stdved ){
			fprintf( StdErr, " (stdv correction=%s)%s",
				d2str( SS->stdv- stdv, NULL, NULL), (debugFlag && !ascanf_verbose)? "\n" : " "
			);
		}
#endif
		SS->stdved= True;
	}
	else{
		  /* 20050811: same side effect of calculating the mean as in exact mode! */
		if( !SS->meaned ){
			SS_Mean(SS);
		}
		sum_sqr*= f;
		sum*= f;
		if( (stdv= (sum_sqr- ((sum* sum)/ weight_sum) )/ (weight_sum - 1.0) )>= 0.0 ){
			SS->stdved= True;
			SS->stdv= sqrt( stdv);
		}
		else{
			SS->stdved= False;
			SS->stdv= -1.0;
		}
	}
	return( SS->stdv );
}

/* return the mean absolute deviation of a dataset as described by the
 * SimpleStats structure *SS. The same considerations as given under
 * st_dev and SS_Mean hold. *SS_exact must be set for this routine to work.
 */
double SS_ADev( SimpleStats *SS)
{  long count= SS->count;
   double weight_sum= SS->weight_sum, f= 1.0;
#ifdef OLD_ATTEMPT
   double sum_dif= 0;
#endif
   int i;

	if( count== 1L ){
		return( 0.0);
	}
	if( count <= 0 || weight_sum<= 0 || !SS->curV || !SS->Nvalues ){
		return( MIN(-1.0, *SS_Empty_Value) );
	}
	if( !SS->medianed ){
		SS_Median(SS);
	}
	  /* Better still: scale for the average weight. This way, the standard deviation
	   \ of a range becomes independent of the average weight, i.e. the stdev of a range
	   \ is invariant of a fixed weight applied to all values.
	   */
	SS_weight_scaler= f= count/ weight_sum;
	weight_sum= (double) count;
#ifdef OLD_ATTEMPT
  /* This represents an attempt to implement the MAD from how it is described in the Num. Recipes. It
   \ didn't quite give the same results as those obtained with R...
   */
	for( i= 0; i< SS->curV; i++ ){
		sum_dif+= SS->sample[i].weight* fabs( SS->sample[i].value- SS->median );
	}
	SS->adeved= True;
	return( SS->adev= f* sum_dif/ weight_sum );
#else
  /* The MAD is defined as median( sum( abs(Xi-median(X)) ) ) * 1/qnorm(0.75) . The constant is approx. 1.482602218505602031939,
   \ and rescales the value to match the std.dev.
   */
   { ALLOCA( X, SS_Sample, SS->curV, Xlen);
     SimpleStats ss;
	   SS_Reset_(ss);
	   for( i= 0; i< SS->curV; i++ ){
		   X[i].value= fabs( SS->sample[i].value- SS->median );
		   X[i].weight= SS->sample[i].weight;
		   X[i].id= i;
	   }
	     /* initialise the stub SimpleStats that will be used to calculate the 2nd median: */
	   ss.count= ss.curV= SS->curV;
	   ss.sample= X;
	   return( _SS_Median( &ss, X ) * 1.482602218505602031939 );
   }
#endif
}

/* return the requested quantile of a dataset as described by the
 * SimpleStats structure *SS. The same considerations as given under
 * st_dev and SS_Mean hold. *SS_exact must be set for this routine to work.
 */
double SS_Quantile( SimpleStats *SS, double prob )
{  long N= SS->count;

	if( N== 1L ){
		return( 0.0);
	}
	if( N <= 0 || SS->weight_sum<= 0 || !SS->curV || !SS->Nvalues ){
		return( MIN(-1.0, *SS_Empty_Value) );
	}
	{
	    /* 20020612: need to sort into a copy, not in the original!!! */
	  ALLOCA( samples, SS_Sample, SS->curV+1, slen );
	  double r, i, f;
	  SS_Sample *s1, *s2;
		N= SS->curV;
		memcpy( samples, SS->sample, SS->curV* sizeof(SS_Sample) );
		samples[SS->curV]= samples[SS->curV-1];
		qsort( samples, SS->curV, sizeof(SS_Sample), (void*) SS_Sort_Samples );
		  /* corrected from the R manual for the quantile() function: */
		r= 1+(N-1) * prob;
		f= r- (i= floor(r));
		  /* R uses 1-based arrays, C 0-based arrays, so we implicitly decrement i by 1,
		   \ giving i-1 for the 1st sample, and i for the 2nd
		   */
		s1= &samples[(unsigned long) MIN( i-1, N)];
		s2= &samples[(unsigned long) MIN( i, N)];
		SS->quantile[0]= (1-f)* s1->value * s1->weight +
			f* s2->value* s2->weight;
	}
	SS->quantiled= True;
	SS->quantile[1]= prob;
	return( SS->quantile[0]  );
}

/* return the standard error of a dataset as described by the
 * SimpleStats structure *SS. The same considerations as given under
 * st_dev and SS_Mean hold.
 */
double SS_St_Err( SimpleStats *SS)
{	double sqrt();
	double stdv= SS_St_Dev(SS);
	if( stdv>= 0 ){
		return( sqrt( stdv * stdv / SS->count ) );
	}
	else{
		return( -1 );
	}
}

double SS_Skew( SimpleStats *SS)
{ long count= SS->count;
  double weight_sum= SS->weight_sum,
	sum= SS->sum,
	sum_sqr= SS->sum_sqr, sum_cub= SS->sum_cub;

	set_NaN(SS->skew);
	if( count== 1L )
		return( 0.0);
	if( count <= 0.0 || weight_sum<= 0 )
		return( 0.0);
	if( SS_St_Dev(SS)<= 0 ){
		return(0);
	}
	  /* SS_weight_scaler is determined by SS_St_Dev()	*/
	if( SS_weight_scaler!= 1.0 ){
		sum_sqr*= SS_weight_scaler;
		sum_cub*= SS_weight_scaler;
		sum*= SS_weight_scaler;
		weight_sum*= SS_weight_scaler;
	}
	SS->mean= sum/ weight_sum;
/* 	SS->skew= (3* SS->mean* sum_sqr- sum_cub- 2* (sum*sum*sum)/(count*count))/ (SS->stdv*count);	*/
	SS->skew= (3* SS->mean* sum_sqr- sum_cub- 2* (sum*sum*sum)/(weight_sum*weight_sum))/ (SS->stdv*weight_sum);
	return( SS->skew );
}

  /* Copy statbin b into a. This potentially involves copying dynamically allocated
   \ arrays into one another, so a simple assign *a=*b is not good.
   */
SimpleStats *SS_Copy( SimpleStats *a, SimpleStats *b )
{ long i, Nvalues= a->Nvalues;
  SS_Sample *sample= a->sample;
	  /* a simple assign is possible after saving some critical pointer and counter values.	*/
	*a= *b;
	if( Nvalues< a->Nvalues ){
		  /* NOTE that the 1st arg. to realloc is <value> resp. <weight>, a's original pointers!	*/
		if( !(a->sample= (SS_Sample*) realloc( sample, a->Nvalues * sizeof(SS_Sample))) ){
			fprintf( StdErr, "SS_Copy(0x%lx,0x%lx): can't get memory for %ld samples (%s)\n",
				a, b, a->Nvalues, serror()
			);
			a->Nvalues= 0;
			a->curV= 0;
			xfree( a->sample );
			xfree( sample );
		}
	}
	else{
		a->Nvalues= Nvalues;
		a->sample= sample;
	}
	if( a->sample && b->sample ){
		for( i= 0; i< a->curV; i++ ){
			a->sample[i].id= b->sample[i].id;
			a->sample[i].value= b->sample[i].value;
			a->sample[i].weight= b->sample[i].weight;
		}
	}
	return( a );
}

#ifdef __GNUC__
inline
#endif
static void SS_update_minmax( SimpleStats *a, double sum )
{
	if( !a->count ){
		a->min= sum;
		a->max= sum;
		if( !NaNorInf(sum) ){
			if( sum> 0 ){
				a->pos_min= sum;
			}
			else{
				a->pos_min= -1;
			}
			a->nr_min= sum;
			a->nr_max= sum;
		}
		else{
			/* 20050107: dirty trick to make sure the first next numerical value will properly initialise the nr_bounds: */
			set_Inf(a->nr_min,1);
			set_Inf(a->nr_max,-1);
		}
	}
	/* 20050106: redone the if/else structure. */
	else{
		if( sum< a->min ){
			a->min= sum;
		}
		else if( sum> a->max ){
			a->max= sum;
		}
		if( !NaNorInf(sum) ){
			if( sum< a->nr_min ){
				a->nr_min= sum;
			}
			/* no else here! */ if( sum> a->nr_max ){
				a->nr_max= sum;
			}
			if( sum> 0 && (a->pos_min< 0 || sum< a->pos_min) ){
				a->pos_min= sum;
			}
		}
	}
}

SimpleStats *SS_Add_Data(SimpleStats *a, long count, double sum, double weight)
{
	if( !a ){
		return(NULL);
	}
	SS_update_minmax( a, sum );
	a->count+= (a->last_count= count);
	{ double ws;
		a->sum+= (ws= (weight)*(a->last_item= sum));
		a->sum_sqr+= (ws*= sum);
		a->sum_cub+= ws* sum;
		a->weight_sum+= (a->last_weight= weight);
	}
	a->meaned= False;
	a->stdved= False;
	a->medianed= False;
	a->adeved= False;
	  /* re-use an SAS behavioural parameter..	*/
	if( *SS_exact || a->exact ){
	  extern int SAS_basic_size;
		if( !a->Nvalues || a->curV>= a->Nvalues ){
			  /* For a SAS_basic_size == 32 :
			   \ begin with 32 elements, next, if necessary:
			   \ increase to 128, next, if necessary:
			   \ increase with 128
			   \ This may result in more allocations than doubling the arena at each time,
			   \ but gives more efficient memory-use (unless we'd provide an option to either
			   \ specify the final size, or to issue a "terminating" statement allowing
			   \ to free un-needed space.
			   */
			a->Nvalues= (a->Nvalues)?
					((a->Nvalues== SAS_basic_size)? 4* SAS_basic_size : a->Nvalues+ 4* SAS_basic_size) :
					SAS_basic_size;
			if( !(a->sample= (SS_Sample*) realloc( a->sample, a->Nvalues * sizeof(SS_Sample))) ){
				fprintf( StdErr, "SS_Add_Data(0x%lx,%ld,%s,%s): can't get memory for %ld samples (%s)\n",
					a, count, d2str( sum, NULL, NULL), d2str( weight, NULL, NULL), a->Nvalues, serror()
				);
				a->Nvalues= 0;
				a->curV= 0;
				xfree( a->sample );
			}
		}
		if( pragma_likely( a->sample ) ){
			a->sample[ a->curV ].id= (unsigned short) a->curV;
			a->sample[ a->curV ].value= sum;
			a->sample[ a->curV ].weight= weight;
			a->curV+= 1;
		}
	}
	return( a );
}

// pop (remove) <count> items from bin <a>, starting at position <pos>. Makes sense only
// in 'exact' mode.
long SS_Pop_Data( SimpleStats *a, long count, long pos, int update )
{ long i, n= 0;
  SS_Sample *s;
	if( !a || !(a->exact || *SS_exact) ){
		return(0);
	}
	else if( pos< 0 || pos+count > a->curV ){
		return(-1);
	}
	else if( a->count != a->curV ){
		return(-2);
	}
	  // always subtract the popped values from the increment registers that are supposed to be valid always:
	for( i= pos, n= 0; i< a->curV && n< count; i++, n++ ){
	  double ws;
		s= &a->sample[i];
		ws = s->weight * s->value;
		a->sum-= ws;
		a->sum_sqr-= (ws *= s->value);
		a->sum_cub-= ws * s->value;
		a->weight_sum-= s->weight;
	}
	  // let's be paranoid and use <n> instead of <count> (they should be equal):
	a->count-= (a->last_count= n);
	if( i< a->curV ){
		memmove( &a->sample[pos], &a->sample[i], sizeof(SS_Sample) * (a->curV - (pos+n)) );
		a->curV -= n;
	}
	if( update ){
	  long cnt= a->count;
		a->count= 0;
		SS_update_minmax( a, a->sample[0].value );
		a->count= cnt;
		for( i= 1; i< a->curV; i++ ){
			SS_update_minmax( a, a->sample[i].value );
		}
	}
	return(n);
}

char *_SS_sprint( char *buffer, int len, char *format, char *sep, double min_err, SimpleStats *a)
{ double stdv;
   char *mean;
   ALLOCA( form, char, (2+ (format)? strlen(format):0), flen );
   ALLOCA( spr, char, (2+ (sep)? strlen(sep):0), slen );

	if( !a ){
		return( "" );
	}
	if( format ){
		strcpy( form, format );
	}
	if( sep ){
		strcpy( spr, sep );
	}
	parse_codes( form );
	parse_codes( spr );
	  /* 20040215: some versions of snprintf() do not correctly handle something like
	   \ snprintf( buffer, sizeof(buffer), "%s: bla", buffer );
	   \ but will erase everything yet present in <buffer> (instead of appending).
	   \ The easy way around, without any platform checking, is to use one of d2str()'s
	   \ internal buffers...
	   */
	mean= d2str( SS_Mean(a), form, NULL);
	if( (stdv= SS_St_Dev(a))> min_err ){
	  char Format[256];
		snprintf( Format, sizeof(Format), "%%s%s%%s", spr );
		snprintf( buffer, len, Format, mean, d2str( stdv, form, NULL) );
	}
	  /* 20040229 (!): of course we need to do something if ever we fail the min_err test... we no longer
	   \ print SS_Mean() directly into <buffer> with d2str ...
	   */
	else{
		snprintf( buffer, len, "%s", mean );
	}
	return( buffer );
}

char *SS_sprint( char *buffer, char *format, char *sep, double min_err, SimpleStats *a)
{  static char _buffer[100][256];
   static int bnr= 0;
   int len;

	if( !buffer ){
		buffer= _buffer[bnr];
		bnr= (bnr+1) % 100;
		len= 256;
	}
	else{
		len= 1024;
	}
	return( _SS_sprint( buffer, len, format, sep, min_err, a ) );
}

char *SS_sprint_full( char *buffer, char *format, char *sep, double min_err, SimpleStats *a)
{  static char _buffer[100][256];
   static int bnr= 0;
   char Format[256];
   int len;
	if( !a ){
		return( "" );
	}
	if( !buffer ){
		buffer= _buffer[bnr];
		bnr= (bnr+1) % 100;
		len= 256;
	}
	else{
		len= 1024;
	}
	_SS_sprint( buffer, len, format, sep, min_err, a);
	if( a->takes ){
		snprintf( Format, sizeof(Format), "%%s c=%ld(%%s*%%s) [%%s|%%s:%%s] t=%ld", (long) a->count, a->takes );
	}
	else{
		snprintf( Format, sizeof(Format), "%%s c=%ld(%%s*%%s) [%%s|%%s:%%s]", (long) a->count );
	}
	strcat( Format, " skew=%s ");
	SS_Skew( a );
	if( *SS_exact || a->exact ){
		strcat( Format, "cmax=%d " );
	}
	sprintf( buffer, Format, buffer,
		d2str( a->weight_sum, format, NULL),
		d2str( SS_weight_scaler, format, NULL),
		d2str( a->min, format, NULL),
		(a->pos_min>0)? d2str( a->pos_min, format, NULL) : "NaN",
		d2str( a->max, format, NULL),
		d2str( a->skew, format, NULL),
		a->Nvalues
	);
	StringCheck( buffer, len, __FILE__, __LINE__ );
	return( buffer );
}

/* -------------------- T test routines -------------------- */

  /* Returns the value ln(Gamma(x) for xx > 0. */
double gammln(double xx)
{ double x,y,tmp,ser;
  static double cof[6]={
		76.18009172947146,-86.50532032941677,
		24.01409824083091,-1.231739572450155,
		0.1208650973866179e-2,-0.5395239384953e-5};
  int j;
	y=x=xx;
	tmp=x+5.5;
	tmp-= (x+0.5)*log(tmp);
	ser= 1.000000000190015;
	for( j= 0; j< 6; j++){
		ser+= cof[j]/ (++y);
	}
	return( -tmp+ log( 2.5066282746310005* ser/x) );
}

unsigned long MAXIT= 65536;

  /* Used by betai: Evaluates continued fraction for incomplete beta function by modified Lentz's method	*/
double betacf(double a, double b, double x)
{ int m, m2;
  double aa, c, d, del, h, qab, qam, qap;
	qab=a+b;
	  /* These q's will be used in factors that occur
	   \ in the coefficients (6.4.6).
	   */
	qap= a+ 1.0;
	qam= a- 1.0;
	c= 1.0;
	  /* First step of Lentz's method. */
	d= 1.0- qab* x/ qap;
	if( fabs(d)< DBL_MIN ){
		d= DBL_MIN;
	}
	d=1.0/d;
	h=d;
	for( m= 1; m<= MAXIT; m++ ){
		m2= 2* m;
		aa= m* (b-m)* x/( (qam+m2) * (a+m2) );
		d= 1.0+ aa*d;
		  /* One step (the even one) of the recurrence. */
		if( fabs(d) < DBL_MIN ){
			d= DBL_MIN;
		}
		c= 1.0+ aa/ c;
		if( fabs(c) < DBL_MIN ){
			c= DBL_MIN;
		}
		d= 1.0/ d;
		h*= d*c;
		aa= -(a + m) * (qab + m) * x/( (a + m2) * (qap + m2) );
		d= 1.0+ aa* d;
		  /* Next step of the recurrence (the odd one). */
		if( fabs(d) < DBL_MIN ){
			d= DBL_MIN;
		}
		c= 1.0+ aa/ c;
		if( fabs(c) < DBL_MIN ){
			c= DBL_MIN;
		}
		d= 1.0/ d;
		del= d* c;
		h*= del;
		if( fabs(del-1.0) < DBL_EPSILON ){
		  /* Are we done? */
			break;
		}
	}
	if( m > MAXIT ){
		fprintf( StdErr, "betacf(a=%s,b=%s,x=%s): a or b too big, or MAXIT==%lu too small\n",
			d2str( a, NULL, NULL), d2str( b, NULL, NULL), d2str( x,0,0), MAXIT
		);
	}
	return( h );
}

/* Returns the incomplete beta function Ix (a, b).	*/
double betai(double a, double b, double x)
{ double gammln(double xx);
  double bt;
	if (x < 0.0 || x > 1.0){
		fprintf( StdErr, "Bad x==%s in routine betai\n", d2str(x, NULL, NULL) );
	}
	if( x== 0.0 || x== 1.0 ){
		bt=0.0;
	}
	else{
	  /* Factors in front of the continued fraction.	*/
		bt= exp( gammln(a+b)- gammln(a)- gammln(b)+ a* log(x)+ b* log(1.0-x) );
	}
	if( x< (a+1.0)/(a+b+2.0) ){
	  /* Use continued fraction directly.	*/
		return( bt* betacf(a,b,x)/ a );
	}
	else{
	  /* Use continued fraction after making the symmetry transformation.	*/
		return( 1.0- bt* betacf( b,a,1.0-x )/ b );
	}
}


double TTest( double mean1, double var1, int n1, double mean2, double var2, int n2, double *t )
{ double df, svar, lt;
	if( !t ){
		t= &lt;
	}
	  /* Degrees of freedom. */
	df= n1+ n2- 2;
	if( df<= 0 ){
		*t= -1;
		return( 2 );
	}
	  /* Pooled variance. */
	svar= ( (n1-1) * var1 + (n2-1) * var2 )/ df;
	*t= ( mean1 - mean2 )/ sqrt( svar * (1.0/ n1 + 1.0/ n2) );
	return( betai( 0.5 * df, 0.5, df/ (df + (*t)*(*t)) ) );
}

  /* Given d1 and d2, this routine returns Student's t in *t,
   \ and its significance as result, small values indicating that the arrays have significantly
   \ different means. The data arrays are assumed to be drawn from populations with the same
   \ true variance.
   */
double SS_TTest( SimpleStats *d1, SimpleStats *d2, double *t )
{ double var1, var2, svar, df, lt;
  int n1, n2;
	if( !t ){
		t= &lt;
	}
	if( !d1 || !d2 ){
		*t= -1;
		return(2);
	}
	n1= d1->count, n2= d2->count;
	  /* Degrees of freedom. */
	df= n1+ n2- 2;
	if( !d1->curV || !d2->curV || df<= 0 ){
		*t= -1;
		return( 2 );
	}
	if( !d1->stdved ){
		SS_St_Dev(d1);
	}
	var1= d1->stdv * d1->stdv;
	if( !d2->stdved ){
		SS_St_Dev(d2);
	}
	var2= d2->stdv * d2->stdv;

	return( TTest( d1->mean, var1, n1, d2->mean, var2, n2, t ) );

	  /* Pooled variance. */
	svar= ( (n1-1) * var1 + (n2-1) * var2 )/ df;
	*t= ( d1->mean - d2->mean )/ sqrt( svar * (1.0/ n1 + 1.0/ n2) );
	return( betai( 0.5 * df, 0.5, df/ (df + (*t)*(*t)) ) );
}

double TTest_uneq( double mean1, double var1, int n1, double mean2, double var2, int n2, double *t )
{ double df, lt;
	if( !t ){
		t= &lt;
	}
	if( n1< 2 || n2< 2 ){
		*t= -1;
		return( 2 );
	}
	*t= ( mean1 - mean2 )/ sqrt( var1/n1 + var2/n2 );
	df= SQR( var1/n1 + var2/n2 )/ ( SQR(var1/n1)/(n1-1) + SQR(var2/n2)/(n2-1) );
	return( betai( 0.5* df, 0.5, df/(df+ (*t)*(*t)) ) );
}

  /* Given d1 and d2, this routine returns Student's t in *t, and
   \ its significance as the result, small values indicating that the arrays have significantly differ-
   \ ent means. The data arrays are allowed to be drawn from populations with unequal variances.
   */
double SS_TTest_uneq( SimpleStats *d1, SimpleStats *d2, double *t )
{ double var1, var2, df, lt;
  int n1, n2;
	if( !t ){
		t= &lt;
	}
	if( !d1 || !d2 ){
		*t= -1;
		return(2);
	}
	n1= d1->count, n2= d2->count;
	if( !d1->curV || !d2->curV || n1< 2 || n2< 2 ){
		*t= -1;
		return( 2 );
	}
	if( !d1->stdved ){
		SS_St_Dev(d1);
	}
	var1= d1->stdv * d1->stdv;
	if( !d2->stdved ){
		SS_St_Dev(d2);
	}
	var2= d2->stdv * d2->stdv;

	return( TTest_uneq( d1->mean, var1, n1, d2->mean, var2, n2, t) );

//	*t= ( d1->mean - d2->mean )/ sqrt( var1/n1 + var2/n2 );
//	df= SQR( var1/n1 + var2/n2 )/ ( SQR(var1/n1)/(n1-1) + SQR(var2/n2)/(n2-1) );
//	return( betai( 0.5* df, 0.5, df/(df+ (*t)*(*t)) ) );
}

  /* Given the paired d1 and d2, this routine returns Student's t for
   \ paired data in *t, and its significance as the result, small values indicating a significant
   \ difference of means.
   */
double SS_TTest_paired( SimpleStats *d1, SimpleStats *d2, double *t )
{ unsigned long j;
  double var1, var2, lt, sd, df, cov= 0.0, f1, f2;
  int n, n2;
	if( !t ){
		t= &lt;
	}
	if( !d1 || !d2 ){
		*t= -1;
		return(2);
	}
	n= d1->count, n2= d2->count;
	if( !d1->curV || !d2->curV || n< 1 || n2!= n ){
		*t= -1;
		return( 2 );
	}
	if( !d1->stdved ){
		SS_St_Dev(d1);
	}
	var1= d1->stdv * d1->stdv;
	f1= n/ d1->weight_sum;
	if( !d2->stdved ){
		SS_St_Dev(d2);
	}
	var2= d2->stdv * d2->stdv;
	f2= n2/ d2->weight_sum;

	for( j= 0; j< n; j++){
		cov += ( d1->sample[j].value* d1->sample[j].weight* f1 - d1->mean) *
			( d2->sample[j].value* d2->sample[j].weight* f2 - d2->mean);
	}
	cov/= (df= n-1);
	sd= sqrt( (var1 + var2 - 2.0* cov)/ n );
	*t= (d1->mean - d2->mean)/ sd;
	return( betai( 0.5* df, 0.5, df/(df+(*t)*(*t)) ) );
}

double FTest( double var1, int n1, double var2, int n2, double *f )
{ double lf, df1, df2, prob;
	if( !f ){
		f= &lf;
	}
	if( n1< 2 || n2< 2 ){
		*f= -1;
		return( 2 );
	}

	if( var1== var2 && n1== n2 ){
		return(1);
	}
	else if( !var1 && !var2 ){
		return(1);
	}
	else if( !var1 || !var2 ){
		set_Inf(prob, 1);
		return(prob);
	}
	if( var1 > var2 ){
	  /* Make F the ratio of the larger variance to the smaller one.	*/
		*f= var1/var2;
		df1= n1-1;
		df2= n2-1;
	}
	else{
		*f= var2/var1;
		df1= n2-1;
		df2= n1-1;
	}
	prob = 2.0* betai( 0.5* df2, 0.5* df1, df2/(df2 + df1*(*f)) );
	if( prob > 1.0){
		prob= 2.0- prob;
	}
	return( prob );
}

  /* Given d1 and d2, this routine returns the value of f in *f, and
   \ its significance as the result. Small values indicate that the two arrays have significantly
   \ different variances.
   */
double SS_FTest( SimpleStats *d1, SimpleStats *d2, double *f )
{ double var1, var2, lf, df1, df2, prob;
  int n1, n2;
	if( !f ){
		f= &lf;
	}
	if( !d1 || !d2 ){
		*f= -1;
		return(2);
	}
	n1= d1->count, n2= d2->count;
	if( !d1->curV || !d2->curV || n1< 2 || n2< 2 ){
		*f= -1;
		return( 2 );
	}
	if( !d1->stdved ){
		SS_St_Dev(d1);
	}
	var1= d1->stdv * d1->stdv;
	if( !d2->stdved ){
		SS_St_Dev(d2);
	}
	var2= d2->stdv * d2->stdv;

	return( FTest( var1, n1, var2, n2, f ) );

	if( var1 > var2 ){
	  /* Make F the ratio of the larger variance to the smaller one.	*/
		*f= var1/var2;
		df1= n1-1;
		df2= n2-1;
	}
	else{
		*f= var2/var1;
		df1= n2-1;
		df2= n1-1;
	}
	prob = 2.0* betai( 0.5* df2, 0.5* df1, df2/(df2 + df1*(*f)) );
	if( prob > 1.0){
		prob= 2.0- prob;
	}
	return( prob );
}
/* --------------------------------------------------------- */

/* (Simpler) versions for statistics on angles:	*/

SimpleAngleStats EmptySimpleAngleStats= {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

double Gonio_Base_Value= 2*M_PI, Gonio_Base_Value_2= M_PI, Gonio_Base_Value_4= M_PI_2;
double Units_per_Radian= 1.0, Gonio_Base_Offset=0;

#define __DLINE__	(double)__LINE__
extern LocalWin *check_wi( LocalWin **wi, char *caller );

double Gonio_Base( LocalWin *wi, double base, double offset)
{  int pF= check_wi(&wi,"Gonio_Base")->polarFlag;
   double YY;
   extern double Trans_X();
	if( base< 0 ){
		base*= -1;
	}

	YY= Gonio_Base_Value;

	wi->polarFlag= 0;
	if( base ){
		Units_per_Radian= (Gonio_Base_Value= base )/ (2*M_PI);
		Gonio_Base_Value_2= base / 2.0;
		Gonio_Base_Value_4= base / 4.0;
	}
	Gonio_Base_Offset= offset;
	wi->polarFlag= pF;
	if( pragma_unlikely( debugFlag && base!= YY ) ){
		fprintf( StdErr, "Gonio_Base(%g,%g): base=%g => %g units per radian\n",
			base, offset, Gonio_Base_Value, Units_per_Radian
		);
		fflush( StdErr );
	}
	return( Gonio_Base_Value);
}

extern char *matherr_mark();

extern double cus_pow();

#ifndef STR
#	define STR(name)	# name
#endif
#ifndef STRING
#	define STRING(name)	STR(name)
#endif
#define MATHERR_MARK()	matherr_mark(__FILE__ ":" STRING(__LINE__))

/* return arg(x,y) in [0,M_2PI]	*/
double atan3( double x, double y)
{
	if( x> 0.0){
		if( y>= 0.0){
			return( atan( y/x));
		}
		else{
			return( M_2PI+ atan( y/x));
		}
	}
	else if( x< 0.0){
		return( M_PI+ atan( y/x));
	}
	else{
		if( y> 0.0)
			return( M_PI_2 );
		else if( y< 0.0)
			return( M_PI + M_PI_2 );
		else
			return( 0.0);
	}
}

/* return arg(x,y) in [0,M_2PI]	*/
double _atan3( LocalWin *wi, double x, double y)
{
	check_wi( &wi, "_atan3" );
	if( x> 0.0){
		if( y>= 0.0){
			MATHERR_MARK();
			return( (atan( cus_pow(y/x, 1/wi->powAFlag) )));
		}
		else{
			MATHERR_MARK();
			return( M_2PI+ (atan( cus_pow(y/x, 1/wi->powAFlag) )));
		}
	}
	else if( x< 0.0){
		MATHERR_MARK();
		return( M_PI+ (atan( cus_pow(y/x, 1/wi->powAFlag) )));
	}
	else{
		if( y> 0.0)
			return( M_PI_2 );
		else if( y< 0.0)
			return( M_PI + M_PI_2 );
		else
			return( 0.0);
	}
}

int SS_valid_exact( SimpleStats *ss )
{
	if( ss && ss->count && ss->sample && (ss->exact || *SS_exact) ){
		return(True);
	}
	else{
		return(False);
	}
}


/* 981218: declarations below should be extern!! 	*/
extern double *SAS_converts_angle, *SAS_converts_result, *SAS_exact;

extern double conv_angle_();

/* return the mean of a dataset as described by the
 * SimpleAngleStats structure *SAS. Note that the SAS_ routines
 * expect that both the count and the weight_sum fields are
 * set.
 */
double SAS_Mean( SimpleAngleStats *SAS)
{  double gb= Gonio_Base_Value, go= Gonio_Base_Offset, pav, nav, av= *SS_Empty_Value;
   int ok= 0;
	if( SAS->Gonio_Base!= gb){
		Gonio_Base( NULL, SAS->Gonio_Base, SAS->Gonio_Offset );
	}
	else{
		Gonio_Base( NULL, gb, SAS->Gonio_Offset );
	}
	if( *SAS_exact || SAS->exact ){
	  /* See under SAS_Add_Data()	*/
		if( SAS->pos_count ){
			av= Units_per_Radian * atan3( SAS->pos_sum/ SAS->pos_count , SAS->neg_sum/ SAS->pos_count );
			SAS->meaned= True;
		}
	}
	else{
		if( SAS->pos_count && SAS->pos_weight_sum){
			pav= SAS->pos_sum/ SAS->pos_weight_sum;
			ok+= 1;
		}
		if( SAS->neg_count && SAS->neg_weight_sum){
			nav= SAS->neg_sum/ SAS->neg_weight_sum;
			ok+= 2;
		}
		switch( ok){
			case 1:
				av= pav;
				break;
			case 2:
				av= nav;
				break;
			case 3:{
				av= Units_per_Radian * atan3( Cos(pav)+Cos(nav) , Sin(pav)+Sin(nav) );
				break;
			}
		}
	}
	SAS->mean= ((*SAS_converts_result && !NaNorInf(av))? conv_angle_(av, Gonio_Base_Value) : av) - Gonio_Base_Offset;
	Gonio_Base( NULL, gb, go);
	return( SAS->mean );
}

/* return the standard deviation of a dataset as described by the
 * SimpleAngleStats structure *SAS. The same considerations as given under
 * st_dev and SAS_Mean hold.
 */
double SAS_St_Dev( SimpleAngleStats *SAS)
{
	if( *SAS_exact || SAS->exact ){
		if( SAS->pos_count== 1 && SAS->sample ){
			return( SAS->stdv= 0.0 );
		}
		else if( SAS->pos_count && SAS->sample ){
			if( !SAS->meaned ){
				SAS->stdved= False;
				SAS_Mean(SAS);
			}
			if( !SAS->stdved ){
			  int i;
			  double av, sum= 0, sum_sqr= 0.0, f= 1;
#ifdef DEBUG
			  double stdv;
#endif
			  double delta;
			  double gb= Gonio_Base_Value, go= Gonio_Base_Offset;
				if( SAS->Gonio_Base!= gb){
					Gonio_Base( NULL, SAS->Gonio_Base, SAS->Gonio_Offset );
				}
				else{
					Gonio_Base( NULL, gb, SAS->Gonio_Offset );
				}
			/*
				if( SAS->pos_weight_sum<= 1.0 ){
					while( f* SAS->pos_weight_sum<= 1.0 ){
						f*= 10;
					}
				}
			 */
				  /* We *must* use the mean independent of SAS_converts_result here. We're basically
				   \ interested in the spread in the input! We won't convert the result afterwards, neither:
				   \ spread doesn't make any sense when taken modulo some value...!
				   \ SAS_converts_angle has a completely different effect, of course, acting
				   \ on the input...
				   */
				av= Units_per_Radian * atan3( SAS->pos_sum/ SAS->pos_count , SAS->neg_sum/ SAS->pos_count )- Gonio_Base_Offset;

				f= SAS->pos_count/ SAS->pos_weight_sum;
				for( i= 0; i< SAS->curV; i++ ){
					delta= SAS->sample[i].value- SAS->mean;
					  /* determine the smallest difference along the circle:	*/
					if( fabs(delta)> SAS->Gonio_Base && !NaNorInf(delta) ){
						delta= conv_angle_( delta, SAS->Gonio_Base );
					}
					sum_sqr+= f* SAS->sample[i].weight* delta* delta;
					sum+= f* SAS->sample[i].weight* delta;
				}
				SAS->stdv= sqrt( (sum_sqr- (sum* sum)/ (f* SAS->pos_weight_sum))/ (f* SAS->pos_weight_sum- 1) );
#ifdef DEBUG
				stdv= sqrt( sum_sqr/ (f* SAS->pos_weight_sum - 1) );
				if( pragma_unlikely(debugFlag || ascanf_verbose) && !SAS->stdved ){
					fprintf( StdErr, " (stdv correction=%s)%s",
						d2str( SAS->stdv- stdv, NULL, NULL), (debugFlag && !ascanf_verbose)? "\n" : " "
					);
				}
#endif
				SAS->stdved= True;
				Gonio_Base( NULL, gb, go);
			}
			return( SAS->stdv );
		}
		else{
			if( !SAS->meaned ){
				SAS_Mean(SAS);
			}
			SAS->stdved= False;
			return( SAS->stdv= -1.0 );
		}
	}
	else{
	  double sqrt(), stdv,
		  weight_sum= SAS->pos_weight_sum+ SAS->neg_weight_sum,
		  sum= SAS->pos_sum + SAS->neg_sum,	/* SAS_Mean( SAS),	*/
		  sum_sqr= SAS->sum_sqr, f= 1.0;
	  long count= SAS->pos_count+ SAS->neg_count;

		SAS->stdved= False;
		if( !SAS->meaned ){
			SAS_Mean(SAS);
		}
		if( count== 1L ){
			return( SAS->stdv= 0.0);
		}
		if( count <= 0 || weight_sum<= 0){
			return( SAS->stdv= -1.0);
		}
	  /* If the sum of the weight is smaller than one, we multiply all weights with
	   \ a common 10-fold, such that their sum becomes larger than 1. This influences the
	   \ result (somewhat), so a) the situation should be avoided and b) the factor used
	   \ should maybe have a different value.
		if( weight_sum<= 1.0 ){
			while( f* weight_sum<= 1.0 ){
				f*= 10;
			}
			sum_sqr*= f;
			sum*= f;
			weight_sum*= f;
		}
	   */
		  /* See discussion in SS_St_Dev()	*/
		f= count/ weight_sum;
		sum_sqr*= f;
		sum*= f;
		weight_sum= (double) count;
		if( (stdv= (sum_sqr- (sum* sum)/ count)/ (count - 1.0) )>= 0.0 ){
			SAS->stdved= True;
			return( SAS->stdv= sqrt( stdv) );
		}
		else{
			return( SAS->stdv= -1.0);
		}
	}
}

int SAS_basic_size= 32;

  /* Copy angular statbin b into a. This potentially involves copying dynamically allocated
   \ arrays into one another, so a simple assign *a=*b is not good.
   */
SimpleAngleStats *SAS_Copy( SimpleAngleStats *a, SimpleAngleStats *b )
{ long i, Nvalues= a->Nvalues;
  SS_Sample *sample= a->sample;
	  /* a simple assign is possible after saving some critical pointer and counter values.	*/
	*a= *b;
	if( Nvalues< a->Nvalues ){
		  /* NOTE that the 1st arg. to realloc is <value> resp. <weight>, a's original pointers!	*/
		if( !(a->sample= (SS_Sample*) realloc( sample, a->Nvalues * sizeof(SS_Sample))) ){
			fprintf( StdErr, "SAS_Copy(0x%lx,0x%lx): can't get memory for %ld samples (%s)\n",
				a, b, a->Nvalues, serror()
			);
			a->Nvalues= 0;
			a->curV= 0;
			SAS_Reset_(*a);
			return(NULL);
		}
	}
	else{
		a->Nvalues= Nvalues;
		a->sample= sample;
	}
	for( i= 0; i< a->curV; i++ ){
		a->sample[i].id= b->sample[i].id;
		a->sample[i].value= b->sample[i].value;
		a->sample[i].weight= b->sample[i].weight;
	}
	return( a );
}

/* Add data to a SimpleAngleStats structure; lowlevel routine that receives the angle,
 \ and its sine and cosine (for exact mode). The angle is supposed to have been converted
 \ already!!!
 */
SimpleAngleStats *SAS_Add_Data_angsincos( SimpleAngleStats *a, long count,
	double sum, double sinval, double cosval,
	double weight)
{
	if( !a ){
		return(NULL);
	}
	if( *SAS_exact || a->exact ){
	  /* When SAS_exact, angular statistics use a (more) exact scheme. In this case,
	   \ each value added is stored in incremental registers for cosine (the pos_sum..)
	   \ and sine (the neg_sum..) of this value, plus in an array keeping track of all
	   \ individual values added. The weights are applied to the sine and cosine. The mean
	   \ angle is then determined as the arctangent of the average sine value divided by the
	   \ average cosine value. The standard deviation is determined as the square root of the
	   \ (sum over all values of the squared difference of each value minus the mean angle) divided
	   \ by (the number of values [N] minus 1) (i.e. the definition of stdv). The difference of value minus
	   \ mean angle is taken as the minimal difference, i.e. the minimal angle between these two angles.
	   \ Each individual squared difference is multiplied by its individual weight times N/(sum of weights).
	   \ SAS_exact can be controlled by the fascanf() variable "$SAS_exact".
	   */
		if( !a->pos_count ){
			a->min= sum;
			a->max= sum;
		}
		else if( sum< a->min){
			a->min= sum;
		}
		else if( sum> a->max){
			a->max= sum;
		}
		a->last_item= sum;
		a->pos_weight_sum+= (a->last_weight= weight);
		a->pos_sum+= cosval* weight;
		a->neg_sum+= sinval* weight;
		a->pos_count+=(a->last_count=(long)(count));
		if( !a->Nvalues || a->curV>= a->Nvalues ){
			  /* For a SAS_basic_size == 32 :
			   \ begin with 32 elements, next, if necessary:
			   \ increase to 128, next, if necessary:
			   \ increase with 128
			   \ This may result in more allocations than doubling the arena at each time,
			   \ but gives more efficient memory-use (unless we'd provide an option to either
			   \ specify the final size, or to issue a "terminating" statement allowing
			   \ to free un-needed space.
			   */
			a->Nvalues= (a->Nvalues)?
					((a->Nvalues== SAS_basic_size)? 4* SAS_basic_size : a->Nvalues+ 4* SAS_basic_size) :
					SAS_basic_size;
			if( !(a->sample= (SS_Sample*) realloc( a->sample, a->Nvalues * sizeof(SS_Sample))) ){
				fprintf( StdErr, "SAS_Add_Data_angsincos(0x%lx,%ld,%s,%s): can't get memory for %ld samples (%s)\n",
					a, count, d2str( sum, NULL, NULL), d2str( weight, NULL, NULL), a->Nvalues, serror()
				);
				a->Nvalues= 0;
				a->curV= 0;
				SAS_Reset_(*a);
				return(NULL);
			}
		}
		a->sample[ a->curV ].id= (unsigned short) a->curV;
		a->sample[ a->curV ].value= sum;
		a->sample[ a->curV ].weight= weight;
		a->curV+= 1;
		a->meaned= False;
		a->stdved= False;
	}
	else{
		if( !(a->pos_count+a->neg_count) ){
			a->min=sum;
			a->max=sum;
		}
		else if( sum< a->min){
			a->min=sum;
		}
		else if( sum> a->max){
			a->max=sum;
		}
		if( sum>=0){
			a->pos_sum+=weight*(a->last_item=sum);
			a->pos_weight_sum+=(a->last_weight=(double)(weight));
			a->pos_count+=(a->last_count=(long)(count));
		}
		else{
			a->neg_sum+=weight*(a->last_item=sum);
			a->neg_weight_sum+=(a->last_weight=(double)(weight));
			a->neg_count+=(a->last_count=(long)(count));
		}
		a->sum_sqr+=(weight)*sum*sum;
	}
	return(a);
}

/* Add data to a SimpleAngleStats structure	*/
SimpleAngleStats *SAS_Add_Data( SimpleAngleStats *a, long count, double sum, double weight, int convert)
{ double s, c;
	if( !a ){
		return(NULL);
	}
	if( (*SAS_converts_angle || convert) && !NaNorInf(sum) ){
		sum= conv_angle_(sum,a->Gonio_Base);
	}
	if( *SAS_exact || a->exact ){
#if NATIVE_SINCOS
		SinCos( (sum+a->Gonio_Offset)/a->Units_per_Radian, &s, &c );
#else
		{ double v= (sum+a->Gonio_Offset)/a->Units_per_Radian;
			c= cos(v);
			s= sin(v);
		}
#endif
	}
	return( SAS_Add_Data_angsincos( a, count, sum, s, c, weight ) );
}

char *SAS_sprint( char *buffer, char *format, char *sep, double min_err, SimpleAngleStats *a)
{  static char _buffer[100][256];
   static int bnr= 0;
   double stdv;
   ALLOCA( form, char, (2+ (format)? strlen(format):0), flen );
   ALLOCA( spr, char, (2+ (sep)? strlen(sep):0), slen );

	if( !a ){
		return( "" );
	}
	if( !buffer ){
		buffer= _buffer[bnr];
		bnr= (bnr+1) % 100;
	}
	if( format ){
		strcpy( form, format );
	}
	if( sep ){
		strcpy( spr, sep );
	}
	parse_codes( form );
	parse_codes( spr );
	d2str( SAS_Mean(a), form, buffer);
	if( (stdv= SAS_St_Dev(a))> min_err ){
	  char Format[128];
		strcat( buffer, spr);
		sprintf( Format, "%%s%%s" );
		sprintf( buffer, Format, buffer, d2str( stdv, form, NULL) );
	}
	return( buffer );
}

char *SAS_sprint_full( char *buffer, char *format, char *sep, double min_err, SimpleAngleStats *a)
{  static char _buffer[100][256];
   static int bnr= 0;
   char Format[512];
	if( !a ){
		return( "" );
	}
	if( !buffer ){
		buffer= _buffer[bnr];
		bnr= (bnr+1) % 100;
	}
	SAS_sprint( buffer, format, sep, min_err, a);
	if( *SAS_exact || a->exact ){
		if( debugFlag ){
			sprintf( Format, "%%s c=%ld(%s*%s;%ld) [%%s\272:%%s\272] {%s\272,%s\272}",
				(long) a->pos_count,
				d2str( a->pos_weight_sum, format, NULL),
				d2str( a->pos_count/ a->pos_weight_sum, format, NULL),
				a->Nvalues,
				d2str( a->Gonio_Base, format, NULL), d2str( a->Gonio_Offset, format, NULL )
			);
		}
		else{
			sprintf( Format, "%%s c=%ld(%s*%s) [%%s\272:%%s\272] {%s\272,%s\272}",
				(long) a->pos_count,
				d2str( a->pos_weight_sum, format, NULL),
				((a->pos_count)? d2str( a->pos_count/ a->pos_weight_sum, format, NULL) : "0"),
				d2str( a->Gonio_Base, format, NULL), d2str( a->Gonio_Offset, format, NULL )
			);
		}
	}
	else{
		sprintf( Format, "%%s c=%s/%s,%s/%s [%%s,%%s]",
			d2str( a->pos_sum, format, NULL), d2str( a->pos_weight_sum, format, NULL),
			d2str( a->neg_sum, format, NULL), d2str( a->neg_weight_sum, format, NULL)
		);
	}
	if( *SAS_exact || a->exact ){
		strcat( Format, " cmax=%d" );
	}
	sprintf( buffer, Format, buffer, d2str(a->min, format, NULL), d2str(a->max, format, NULL), a->Nvalues );
	return( buffer );
}

int SAS_valid_exact( SimpleAngleStats *sas )
{
	if( sas && sas->pos_count && sas->sample && (sas->exact || *SAS_exact) ){
		return(True);
	}
	else{
		return(False);
	}
}

/* 20050807: some (more) general statistics functions: */

static int sort_doubles( double *a, double *b)
{ double diff= *a - *b;
	int ret;
	if( diff< 0 ){
		ret= ( -1 );
	}
	else if( diff> 0 ){
		ret= ( 1 );
	}
	else{
		ret= ( 0 );
	}
	return( ret );
}

#ifdef __GNUC__
inline
#endif
double find_Median( double *values, unsigned long N, double *sortbuffer )
{ unsigned long N2= N/2;
	double median;
	switch( N ){
		case 0:
			set_NaN(median);
			break;
		case 1:
			median= values[0];
			break;
		case 2:
			median= (values[0]+values[1])/2;
			break;
		default:{
			if( !sortbuffer ){
				sortbuffer= values;
			}
			else if( sortbuffer!= values ){
				memcpy( sortbuffer, values, N*sizeof(double) );
				qsort( sortbuffer, N, sizeof(double), (void*) sort_doubles );
			}
			else{
				qsort( sortbuffer, N, sizeof(double), (void*) sort_doubles );
			}
			if( N2 * 2 == N ){
				/* even */
				median= (sortbuffer[N2-1] + sortbuffer[N2])/ 2;
			}
			else{
				N2= (N+ 1)/ 2;
				median= sortbuffer[N2-1];
			}
			break;
		}
	}
	return( median );
}


/* lm_fit(): do a linear fit by Chi2 minimisation on y~x or y+-stdv~x.
 \ Returns the Chi2 value corresponding to the fit.
 \ Inspired by the "Algorithm Cuisine" 15.2.
 */
double lm_fit(
	double *x, double *y, double *stdv, unsigned long N,	/* the N x,y values and optionally the N standard deviations */
	double *slope, double *icept,					/* return values for lm_fit(x,y) = slope * x + icept */
	double *Pslope, double *Picept, double *goodness	/* optionally return the fit probabilities and the goodness-of-fit (1 if stdv==NULL) */
)
{ extern double incomplete_gammaFunction_complement(double a, double x);
  unsigned long i;
  double wt, t, avX, sx=0.0, sy=0.0, st2=0.0, ss, sigdat, Q= 1.0, chi2= 0.0, a, b= 0.0, siga, sigb, minErr;

	  /* inspired by the Alg. Cuis., so arrays are base-1 ... : */
	x = &x[-1];
	y = &y[-1];
	if( stdv ){
	  double minPosErr;
		  /* Errors are used to weigh data points, inversely. But a zero error doesn't mean
		   \ the corresponding point should have an infinite weight... Thus, find the minimum
		   \ non-zero error first, and then estimate the effective error for the offending points
		   \ as half that minimum positive error.
		   \ Also, do not assume that the specified errors are all positive: take their absolute values.
		   */
		set_Inf(minPosErr,1);
		for( i= 0; i< N; i++ ){
		  double e;
			if( (e= fabs(stdv[i]))!= 0 && e< minPosErr ){
				minPosErr= e;
			}
		}
		if( minPosErr> 0 && !NaNorInf(minPosErr) ){
			minErr= minPosErr/2;
		}
		else{
			stdv= NULL;
			goto lm_no_error;
		}
		stdv = &stdv[-1];
		ss=0.0;
		for( i=1; i<=N; i++ ){
			if( stdv[i] ){
				wt=1.0/SQR(fabs(stdv[i]));
			}
			else{
				wt= 1.0/ SQR(minErr);
			}
			ss += wt;
			sx += x[i]*wt;
			sy += y[i]*wt;
		}
	}
	else{
lm_no_error:;
		for (i=1;i<=N;i++) {
			sx += x[i];
			sy += y[i];
		}
		ss=N;
	}
	avX=sx/ss;
	if( stdv ){
		for( i=1; i<=N; i++ ){
		  double e= (stdv[i])? fabs(stdv[i]) : minErr;
			t=(x[i]-avX)/e;
			st2 += t*t;
			b += t*y[i]/e;
		}
	}
	else{
		for( i=1; i<=N; i++ ){
			t=x[i]-avX;
			st2 += t*t;
			b += t*y[i];
		}
	}
	b /= st2;
	a= (sy-sx*(b))/ss;
	siga= sqrt((1.0+sx*sx/(ss*st2))/ss);
	sigb= sqrt(1.0/st2);

	if( !stdv ){
		for( i=1; i<=N; i++ ){
			chi2 += SQR(y[i]-(a)-(b)*x[i]);
		}
		sigdat= sqrt((chi2)/(N-2));
		siga *= sigdat;
		sigb *= sigdat;
	}
	else{
		for( i=1; i<=N; i++ ){
		  double e= (stdv[i])? fabs(stdv[i]) : minErr;
			chi2 += SQR((y[i]-(a)-(b)*x[i])/e);
		}
		if( N>2 ){
			Q= incomplete_gammaFunction_complement(0.5*(N-2),0.5*(chi2));
		}
	}
	if( slope ){
		*slope= b;
	}
	if( icept ){
		*icept= a;
	}
	if( Picept ){
	  /* fit intercept uncertainty: */
		*Picept= siga;
	}
	if( Pslope ){
	  /* fit slope uncertainty: */
		*Pslope= sigb;
	}
	if( goodness ){
	  /* probability that the fit would have Chi2 this large or larger */
		*goodness= Q;
	}
	return( chi2 );
}

typedef struct rlm_fit_data{
	unsigned long N;
	double *x, *y,
		aa, absDev;
} rlm_fit_data;

static double rlm_median_sum(rlm_fit_data *rfd, double b)
	/* lowlevel workhorse function for rlm_fit */
{ unsigned long j;
  double d, sum=0.0, *xt= rfd->x, *yt= rfd->y, aa= rfd->aa, absDev, *values;
  unsigned long N= rfd->N;
  ALLOCA( Values, double, N, val_len );

	values= &Values[-1];
	for( j= 1; j<= N; j++){
		values[j]= yt[j]- b*xt[j];
	}
	  /* in principle, if xt[] is sorted, values[] will be sorted also,
	   \ so find_Median could skip the sorting step.
	   \ Of course, this would also require that yt[] be sorted along with xt...
	   */
	aa= find_Median( &values[1], N, &values[1] );
	absDev=0.0;
	for( j=1; j<= N; j++ ){
		d= yt[j] - (b*xt[j] + aa);
		absDev += fabs(d);
		if( yt[j] != 0.0 ){
			d /= fabs(yt[j]);
		}
		if( fabs(d) > DBL_EPSILON ){
			sum += (d >= 0.0 ? xt[j] : -xt[j]);
		}
	}
	rfd->aa= aa;
	rfd->absDev= absDev;
	return sum;
}

/* rlm_fit(): do a robust linear fit by by least absolute deviations on y~x.
 \ Returns the mean absolute deviation corresponding to the fit.
 \ Inspired by the "Algorithm Cuisine" 15.7.
 */
double rlm_fit(
	double *x, double *y, unsigned long N,	/* x, y, N */
	double *slope, double *icept			/* return slope and intercept values */
)
{ unsigned long j;
  double bb, b1, b2, del, f, f1, f2, sigb, temp;
  double sx=0.0, sy=0.0, sxy=0.0, sxx=0.0, chisq=0.0;
  rlm_fit_data rfd;

	x= &x[-1];
	y= &y[-1];

	rfd.N= N;
	rfd.x= x;
	rfd.y= y;
	for( j=1; j<= N; j++ ){
		sx += x[j];
		sy += y[j];
		sxy += x[j]*y[j];
		sxx += x[j]*x[j];
	}
	del= N*sxx-sx*sx;
	rfd.aa= (sxx*sy-sx*sxy)/del;
	bb= (N*sxy-sx*sy)/del;
	for( j=1; j<= N; j++ ){
		  /* This surely calculates temp and then increments chisq with temp*temp ?! */
		chisq += (temp=y[j]-(rfd.aa+bb*x[j]),temp*temp);
	}
	sigb= sqrt(chisq/del);
	b1= bb;
	f1= rlm_median_sum( &rfd, b1);
	if( sigb > 0.0 ){
		b2= bb + SIGN(f1) * 3.0*sigb;
		f2= rlm_median_sum( &rfd, b2);
		if( b2 == b1 ){
			goto rlm_return;
		}
		while( f1*f2 > 0.0 ){
			bb= b2 + 1.6 * (b2-b1);
			b1= b2;
			f1= f2;
			b2= bb;
			f2= rlm_median_sum( &rfd, b2);
		}
		sigb= 0.01 * sigb;
		while( fabs(b2-b1) > sigb ){
			bb=b1+0.5*(b2-b1);
			if( bb == b1 || bb == b2 ){
				break;
			}
			f= rlm_median_sum( &rfd, bb);
			if( f*f1 >= 0.0 ){
				f1=f;
				b1=bb;
			}
			else{
				f2=f;
				b2=bb;
			}
		}
	}
rlm_return:;
	if( icept ){
		*icept= rfd.aa;
	}
	if( slope ){
		*slope= bb;
	}
	return( rfd.absDev/N );
}
