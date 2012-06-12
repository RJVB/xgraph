#ifndef _LOWLEVEL_TIMER_H

/****************************************************************************/
/*                                  Timers                                  */
/*  From fftw 2.1.5	*/
/* Adapted by RJVB */
/****************************************************************************/

/*
 * Here, you can use all the nice timers available in your machine.
 */

/*
 *
 Things you should define to include your own clock:

 fftw_time -- the data type used to store a time
 fftw_timers -- a data type for a basic timer; contains 2 fftw_time fields

 extern fftw_time fftw_get_time(fftw_time *t);
 -- a function returning the current time.  (We have implemented this as a macro in most cases.)

 extern fftw_time fftw_time_diff(fftw_timer *diff) -- returns the time difference (diff->t1 - diff->t2).
 If t1 < t2, it may simply return zero (although this is not required).
 (We have implemented this as a macro in most cases.)

 extern double fftw_time_to_sec(fftw_time t);
 -- returns the time t expressed in seconds, as a double.
 (Implemented as a macro in most cases.)

 extern double fftw_time_diff_to_sec(fftw_timer *diff)
 -- returns the the number of seconds elapsed according to the timer diff.

 FFTW_CYCLES_PER_SEC: for calibration: should be set to your processor's clock speed, in Hz
 (so for a 2GHz processor it should be 2000000000L)

 FFTW_TIME_MIN -- a double-precision macro holding the minimum
 time interval (in seconds) for accurate time measurements.
 This should probably be at least 100 times the precision of
 your clock (we use even longer intervals, to be conservative).
 This will determine how long the planner takes to measure
 the speeds of different possible plans.
 */

#if (defined(__MACH__) || defined(__APPLE__))

#include <mach/mach_time.h>

typedef uint64_t fftw_time;
typedef struct fftw_timers{
	fftw_time t1, t2;
} fftw_timers;

extern double Mach_Absolute_Time_Factor;

#	define fftw_get_time(t)  ((*(fftw_time*)t)=mach_absolute_time())
#	define fftw_time_diff(diff) (((diff)->t2) - ((diff)->t1))
#	define fftw_time_to_sec(t) (((double) (*t))*Mach_Absolute_Time_Factor)
#	define fftw_time_diff_to_sec(diff) (((double) ((diff)->t2)- (double)((diff)->t1))*Mach_Absolute_Time_Factor)
#	define FFTW_TIME_MIN (1.0e-4)


#elif defined(USE_PERFORMANCECOUNTER)

//extern void QueryPerformanceCounter(long long *);

typedef long long fftw_time;
typedef struct fftw_timers{
	fftw_time t1, t2;
} fftw_timers;

extern double PerformanceCounter_Calibrator;

static __inline__ fftw_time read_tsc()
{ fftw_time ret;

     __asm__ __volatile__("rdtsc": "=A" (ret));
     /* no input, nothing else clobbered */
     return ret;
}

//static __inline__ fftw_time fftw_get_time(void *t)
//{ long long tt;
//	QueryPerformanceCounter( &tt );
//// 	return *((fftw_time*)t) = tt.QuadPart;
//	return *((fftw_time*)t) = tt;
//}

extern fftw_time fftw_get_time(void *t);

#	define fftw_time_diff(diff) (((diff)->t2) - ((diff)->t1))
#	define fftw_time_to_sec(t) (((double) (*t))*PerformanceCounter_Calibrator)
#	define fftw_time_diff_to_sec(diff) (((double) ((diff)->t2)- (double)((diff)->t1))*PerformanceCounter_Calibrator)
#	define FFTW_TIME_MIN (1.0e-5)

#elif (defined(__GNUC__) || defined(__APPLE_CC__)) && (defined(__i386__) || defined(__x86_64__)) || defined(linux)

// check for clock_gettime function:
#	if defined(CLOCK_MONOTONIC)

typedef double fftw_time;
typedef struct fftw_timers{
	fftw_time t1, t2;
} fftw_timers;

static __inline__ fftw_time fftw_get_time(void *t)
{ struct timespec hrt;
	clock_gettime( CLOCK_MONOTONIC, &hrt );
	return *((fftw_time*)t) = hrt.tv_sec + hrt.tv_nsec * 1e-9;
}

static __inline__ fftw_time fftw_get_res()
{ struct timespec hrt;
	clock_getres( CLOCK_MONOTONIC, &hrt );
	return hrt.tv_sec + hrt.tv_nsec * 1e-9;
}

#		define fftw_time_diff(diff) (((diff)->t2) - ((diff)->t1))
#		define fftw_time_to_sec(t) (*t)
#		define fftw_time_diff_to_sec(diff) (((double) ((diff)->t2)- (double)((diff)->t1)))
#		define FFTW_TIME_MIN (fftw_time_getres())

#	elif defined(CLOCK_REALTIME)

typedef double fftw_time;
typedef struct fftw_timers{
	fftw_time t1, t2;
} fftw_timers;

static __inline__ fftw_time fftw_get_time(void *t)
{ struct timespec hrt;
	clock_gettime( CLOCK_REALTIME, &hrt );
	return *((fftw_time*)t) = hrt.tv_sec + hrt.tv_nsec * 1e-9;
}

static __inline__ fftw_time fftw_get_res()
{ struct timespec hrt;
	clock_getres( CLOCK_REALTIME, &hrt );
	return hrt.tv_sec + hrt.tv_nsec * 1e-9;
}

#		define fftw_time_diff(diff) (((diff)->t2) - ((diff)->t1))
#		define fftw_time_to_sec(t) (*t)
#		define fftw_time_diff_to_sec(diff) (((double) ((diff)->t2)- (double)((diff)->t1)))
#		define FFTW_TIME_MIN (fftw_time_getres())

#	else

/*
 * Use internal Pentium register (time stamp counter). Resolution
 * is 1/FFTW_CYCLES_PER_SEC seconds (e.g. 5 ns for Pentium 200 MHz).
 * (This code was contributed by Wolfgang Reimer)
 * In this implementation, almost 15x faster than gettimeofday() (which has similar resolution).
 */

#		ifndef FFTW_CYCLES_PER_SEC
#error "Must set/define FFTW_CYCLES_PER_SEC = CPU-clockspeed-in-herz to use the Pentium cycle counter"
#		endif

typedef unsigned long long fftw_time;

typedef struct fftw_timers{
	fftw_time t1, t2;
} fftw_timers;

static __inline__ fftw_time read_tsc()
{ fftw_time ret;

     __asm__ __volatile__("rdtsc": "=A" (ret));
     /* no input, nothing else clobbered */
     return ret;
}

#		define fftw_get_time(t)  ((*(fftw_time*)t)=read_tsc())
#		define fftw_time_diff(diff) (((diff)->t2) - ((diff)->t1))
#		define fftw_time_to_sec(t) (((double) (*t)) / FFTW_CYCLES_PER_SEC)
#		define fftw_time_diff_to_sec(diff) (((double) ((diff)->t2)- (double)((diff)->t1)) / FFTW_CYCLES_PER_SEC)
#		define FFTW_TIME_MIN (1.0e-4)	/* for Pentium TSC register */

#	endif // no clock_gettime() function

#elif defined(sgi) && defined(CLOCK_SGI_CYCLE)

/* SGI 'lowlevel' timer routines, using clock_gettime() (as proposed instead of a direct memory-mapping technique
 \  that is surely faster but less portable. Set_Timer *must* have been called for this to work.
 \ Approx. 2x faster on an R5000/200Mhz than gettimeofday() -- but still experimental.
 */

#if FFTW_CYCLES_PER_SEC > 0

#ifndef FFTW_CYCLES_PER_SEC
#error "Must set/define FFTW_CYCLES_PER_SEC (to any value on this platform)"
#endif

#undef FFTW_CYCLES_PER_SEC
#define FFTW_CYCLES_PER_SEC -1

#include <time.h>

#define FFTW_SGI_CLOCK	CLOCK_SGI_CYCLE

typedef struct timespec fftw_time;

extern fftw_time timer_wraps_at;

typedef struct fftw_timers{
	fftw_time t1, t2;
} fftw_timers;

static inline void fftw_get_time(fftw_time *t)
{
	clock_gettime(FFTW_SGI_CLOCK, t);
}

static inline double fftw_time_diff(fftw_timers *diff)
{
	if( diff->t1.tv_sec> diff->t2.tv_sec ){
		  /* remove a wrap. We have no way of knowing here how many wraps occurred: this is the responsibility
		   \ of the user to keep track off!
		   */
		diff->t2.tv_sec+= timer_wraps_at.tv_sec;
		diff->t2.tv_nsec+= timer_wraps_at.tv_nsec;
	}
     return ((double)diff->t2.tv_sec - (double)diff->t1.tv_sec) +
		((double)diff->t2.tv_nsec - (double)diff->t1.tv_nsec)* 1e-9;
}

static inline double fftw_time_to_sec(fftw_time *t)
{
	return( (t->tv_sec + t->tv_nsec*1e-9) );
}

static inline double fftw_time_diff_to_sec(fftw_timers *diff)
{
	if( diff->t1.tv_sec> diff->t2.tv_sec ){
/* 		fprintf( stderr, "fftw_time_diff_to_sec(): wrap at t1=(%lu,%lu) -> t2=(%lu,%lu)",	*/
/* 			diff->t1.tv_sec, diff->t1.tv_nsec,	*/
/* 			diff->t2.tv_sec, diff->t2.tv_nsec	*/
/* 		);	*/
		  /* remove a wrap. We have no way of knowing here how many wraps occurred: this is the responsibility
		   \ of the user to keep track off!
		   */
		diff->t2.tv_sec+= timer_wraps_at.tv_sec;
		diff->t2.tv_nsec+= timer_wraps_at.tv_nsec;
/* 		fprintf( stderr, "=> t2=(%lu,%lu)\n", diff->t2.tv_sec, diff->t2.tv_nsec);	*/
	}
	return ((double)diff->t2.tv_sec - (double)diff->t1.tv_sec) +
		((double)diff->t2.tv_nsec - (double)diff->t1.tv_nsec)*1e-9;
}

  /* educated guess... */
#define FFTW_TIME_MIN (1.0e-4)

#endif

#else

#	undef FFTW_CYCLES_PER_SEC

#endif

#define _LOWLEVEL_TIMER_H
#endif
