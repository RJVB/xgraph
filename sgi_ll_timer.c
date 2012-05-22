#ifdef sgi

#include <stdio.h>
#include <sys/time.h>

#define FFTW_CYCLES_PER_SEC
#include "lowlevel_timer.h"

main()
{ unsigned long i;
  register fftw_time t, pt, tn, ptn;
 
	t.tv_sec= 0;
	t.tv_nsec= 0;
	clock_settime( FFTW_SGI_CLOCK, &t );

	fftw_get_time(&t);
	fprintf( stderr, "%lu.%lu\n", t.tv_sec, t.tv_nsec );
	printf( "%lf\n", fftw_time_to_sec(&t) );
	for( i= 0; i< 10; i++ ){
	  register unsigned long n;
		do{
			pt= t;
			fftw_get_time(&t);
		} while( t.tv_sec>= pt.tv_sec );
		  /* Cache last interval: */
		tn= t;
		ptn= pt;
		n= 0;
		do{
			pt= t;
			fftw_get_time(&t);
			n+= 1;
		} while( t.tv_sec< pt.tv_sec );
		  /* Print here, as we want as little lapse between the larger-than and the smaller-than loops! */
		fprintf( stderr, "t.tv_sec< pt.tv_sec: %lu.%lu -> %lu.%lu\n", ptn.tv_sec, ptn.tv_nsec, tn.tv_sec, tn.tv_nsec );
		fprintf( stderr, "t.tv_sec>= pt.tv_sec: %lu.%lu -> %lu.%lu [%lu]\n", pt.tv_sec, pt.tv_nsec, t.tv_sec, t.tv_nsec, n );
	}
}

#endif
