#include <stdio.h>
#include <errno.h>
#include <sys/file.h>

void flockfile( FILE *fp )
{
	errno= 0;
	if( flock( fileno(fp), LOCK_EX) && !errno ){
		errno= EOPNOTSUPP;
	}
}

void funlockfile( FILE *fp )
{
	errno= 0;
	if( flock( fileno(fp), LOCK_UN) && !errno ){
		errno= EOPNOTSUPP;
	}
}
