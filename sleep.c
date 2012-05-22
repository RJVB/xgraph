#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <signal.h>

#ifdef sgi
#	include <sys/time.h>
#else
#	include <time.h>
#endif


static int ascanf_continue= 0;
ascanf_continue_( int sig )
{
	ascanf_continue= 1;
	signal( SIGALRM, ascanf_continue_ );
}

int sleep_once( double time )
{
	if( time> 0 ){
	  struct itimerval rtt, ortt;
		rtt.it_value.tv_sec= (unsigned long) (time);
		rtt.it_value.tv_usec= (unsigned long) ( (time- rtt.it_value.tv_sec)* 1000000 );
		rtt.it_interval.tv_sec= 0;
		rtt.it_interval.tv_usec= 0;
		signal( SIGALRM, ascanf_continue_ );
		ascanf_continue= 0;
		setitimer( ITIMER_REAL, &rtt, &ortt );
		pause();
		  /* restore the previous setting of the timer.	*/
		setitimer( ITIMER_REAL, &ortt, &rtt );
		return( 1 );
	}
	else{
		return( 0 );
	}
}

int set_interval( double time, double interval )
{ static double Initial= -1, Interval= -1;
	if( time> 0 && interval>= 0 ){
	  struct itimerval rtt, ortt;
		if( time!= Initial || interval!= Interval ){
			rtt.it_value.tv_sec= (unsigned long) (time);
			rtt.it_value.tv_usec= (unsigned long) ( (time- rtt.it_value.tv_sec)* 1000000 );
			rtt.it_interval.tv_sec= (unsigned long) (interval);
			rtt.it_interval.tv_usec= (unsigned long) ( (interval- rtt.it_interval.tv_sec)* 1000000 );
			signal( SIGALRM, ascanf_continue_ );
			ascanf_continue= 0;
			Initial= time;
			Interval= interval;
			setitimer( ITIMER_REAL, &rtt, &ortt );
		}
		if( !ascanf_continue ){
		  /* wait for the signal	*/
			pause();
			  /* Make sure that we will pause if no interval has expired since the last call!	*/
			ascanf_continue= 0;
		}
	}
	else{
		return( 0 );
	}
	return( 1 );
}

main( int argc, char *argv[] )
{
  double time= -1, interval= 0;
  char *msg;
	if( argc< 1 ){
		fprintf( stderr, "%s <time> [<interval> [<output-string>]]\n", argv[0] );
		exit(1);
	}
	if( !sscanf( argv[1], "%lf", &time ) || time< 0 ){
		fprintf( stderr, "%s: invalid time %g\n", argv[0], time );
		exit(2);
	}
	if( argc> 1 && (!sscanf( argv[2], "%lf", &interval) || interval< 0) ){
		fprintf( stderr, "%s: invalid interval %g\n", argv[0], interval );
		exit(3);
	}
	if( argc> 2 ){
		msg= argv[3];
	}
	else{
		msg= "";
	}
	if( interval> 0 ){
	 long count= 0;
		while( set_interval( time, interval ) ){
			printf( "%ld %s\n", count++, msg );
			fflush(stdout);
		}
	}
	else{
		set_interval( time, 0.0);
	}
	exit(0);
}
