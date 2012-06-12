/*
 *  lowlevel_timer.c
 *  XGraph
 *
 *  Created by René J.V. Bertin on 20120612.
 *  Copyright 2012 RJVB. All rights reserved.
 *
 */


#include "cpu_cycles_per_second.h"
#include "lowlevel_timer.h"

#if defined(USE_PERFORMANCECOUNTER)

#	include <windows.h>

	double PerformanceCounter_Calibrator;

	void init_PerformanceCounter()
	{ LARGE_INTEGER lpFrequency;
		if( !QueryPerformanceFrequency(&lpFrequency) ){
			PerformanceCounter_Calibrator = 0;
		}
		else{
 			PerformanceCounter_Calibrator = 1.0 / ((double) lpFrequency.QuadPart);
		}
	}

	fftw_time fftw_get_time(void *t)
	{ LARGE_INTEGER tt;
		QueryPerformanceCounter( &tt );
		return *((fftw_time*)t) = tt.QuadPart;
	}
#endif //USE_PERFORMANCECOUNTER