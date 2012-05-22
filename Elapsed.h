#ifndef ELAPSED_H
#define ELAPSED_H

#include <sys/types.h>
#include <sys/times.h>
#include <time.h>
#include <sys/time.h>

#ifdef FFTW_CYCLES_PER_SEC
#	include "lowlevel_timer.h"
#endif

/* The times() routine can be pretty darn slow. There is therefore a less expensive timer routine
 \ that does not call it, but uses only either the lowlevel timer, or gettimeofday() (and ignores
 \ ELAPSED_PREFERS_GETTIMEOFDAY). This is the Elapsed_Since_HR() function.
 */
typedef struct Time_Struct{
	double TimeStamp, Tot_TimeStamp, sTimeStamp;
	double Time, Tot_Time;
	double sTime;
#if defined(FFTW_CYCLES_PER_SEC) && !defined(ELAPSED_PREFER_GETTIMEOFDAY)
	fftw_time prev_tv, Tot_tv;
#else
	  /* For use with the (potentially) higher resolution gettimeofday():	*/
	struct timeval prev_tv, Tot_tv;
#endif
	  /* the total time since (re)initialisation of this timer:	*/
	double HRTot_T;
	Boolean do_reset;
}Time_Struct;


#define SECSPERDAY 86400.0

#ifdef linux
#	include <sys/times.h>
	extern clock_t _linux_clk_tck;
#	define CLK_TCK	_linux_clk_tck
#endif

#if defined(CLK_TCK)
#	define TICKS_PER_SECOND (double)(CLK_TCK)
#	define SECONDS_PER_TICK	(1.0/(CLK_TCK))
#else
/* assume 60 ticks..	*/
#	define TICKS_PER_SECOND (60.0)
#	define SECONDS_PER_TICK	(1.0/60.0)
#endif

extern double Elapsed_Since(Time_Struct *timer, int update), Elapsed_Time(), Tot_Time;
extern double Elapsed_Since_HR(Time_Struct *timer, int update);

#endif
