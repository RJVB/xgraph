/*
 * generates a family of curves for testing purposes.
 */

#include <stdlib.h>
#include "copyright.h"
#include <math.h>
#include <stdio.h>

static char *setnames[] = {
    "Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta",
    "Theta", "Iota", "Kappa", "Lambda", "Mu", "Nu", "Xi", "Pi",
    "Rho", "Sigma", "Tau", "Upsilon", "Phi", "Chi", "Psi", "Omega"
};

#define MAXNAMES	(sizeof(setnames)/sizeof(char *))

double func(x, i)
double x;
int i;
/* Yeilds a y value given an X value for curve i */
{
     return (x - ((double) i)) * x - ((double) i);
}

main(argc, argv)
int argc;
char *argv[];
{
    int num, index;
    double start, end, step, xval;
    float vals[3];
    FILE *fp;

    if (argc != 5) {
	printf("format: %s curves start finish step\n", argv[0]);
	exit(1);
    }
    num = atoi(argv[1]);
    start = atof(argv[2]);
    end = atof(argv[3]);
    step = atof(argv[4]);
    fp= fopen( "xgtest.binout", "w" );
    for (index = 1;  index <= num;  index++) {
	if (index-1 < MAXNAMES) {
	    printf("\"%s\n", setnames[index-1]);
	}
	    if( fp ){
		    fprintf( fp, "\n\n*LEGEND* %g-%g, step=%g\n", start, end, step );
		    fprintf( fp, "\n*BINARYDATA* lines=%g columns=3 size=%d\n", (double) (end- start)/ step+ 1, sizeof(float) );
	    }
	for (xval = start;  xval <= end;  xval += step) {
	    printf("%G %G\n", xval, func(xval, index));
		if( fp ){
			vals[0]= xval;
			vals[1]= func( xval, index );
			vals[2]= index;
			fwrite( vals, sizeof(float), 3, fp );
		}
	}
	printf("\n");
    }
    if( fp ){
	    fclose( fp );
    }
}
