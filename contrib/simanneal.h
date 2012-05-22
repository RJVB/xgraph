#include <math.h>
#include <limits.h>
#include <float.h>


extern void continuous_simanneal(double **p, double y[], int ndim, double pb[], double *yb, double ftol,
	double (*funk) (double []) , int *iter, double T);
