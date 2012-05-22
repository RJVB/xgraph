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

#define MAXIT 100
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
		fprintf( StdErr, "betacf(a=%s,b=%s): a or b too big, or MAXIT==%d too small\n",
			d2str( a, NULL, NULL), d2str( b, NULL, NULL), MAXIT
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


  /* Given d1 and d2, this routine returns Student's t in *t,
   \ and its significance as result, small values indicating that the arrays have significantly
   \ different means. The data arrays are assumed to be drawn from populations with the same
   \ true variance.
   */
double SS_TTest( SimpleStats *d1, SimpleStats *d2, double *t )
{ double var1, var2, svar, df, lt;
  int n1= d1->count, n2= d2->count;
	if( !t ){
		t= &lt;
	}
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
	  /* Pooled variance. */
	svar= ( (n1-1) * var1 + (n2-1) * var2 )/ df; 
	*t= ( d1->mean - d2->mean )/ sqrt( svar * (1.0/ n1 + 1.0/ n2) );
	return( betai( 0.5 * df, 0.5, df/ (df + (*t)*(*t)) ) ); 
}

  /* Given d1 and d2, this routine returns Student's t in *t, and
   \ its significance as the result, small values indicating that the arrays have significantly differ-
   \ ent means. The data arrays are allowed to be drawn from populations with unequal variances.
   */
double SS_TTest_uneq( SimpleStats *d1, SimpleStats *d2, double *t )
{ double var1, var2, df, lt;
  int n1= d1->count, n2= d2->count;
	if( !t ){
		t= &lt;
	}
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
	*t= ( d1->mean - d2->mean )/ sqrt( var1/n1 + var2/n2 );
	df= SQR( var1/n1 + var2/n2 )/ ( SQR(var1/n1)/(n1-1) + SQR(var2/n2)/(n2-1) );
	return( betai( 0.5* df, 0.5, df/(df+ (*t)*(*t)) ) );
}

  /* Given the paired d1 and d2, this routine returns Student's t for
   \ paired data in *t, and its significance as the result, small values indicating a significant
   \ difference of means.
   */
double SS_TTest_paired( SimpleStats *d1, SimpleStats *d2, double *t )
{ unsigned long j;
  double var1, var2, lt, sd, df, cov= 0.0, f1, f2;
  int n= d1->count, n2= d2->count;
	if( !t ){
		t= &lt;
	}
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

  /* Given the arrays data1[1..n1] and data2[1..n2], this routine returns the value of f, and
   \its significance as prob. Small values of prob indicate that the two arrays have significantly
   \different variances.
   */
double SS_FTest( SimpleStats *d1, SimpleStats *d2, double *f )
{ double var1, var2, lf, df1, df2, prob;
  int n1= d1->count, n2= d2->count;
	if( !f ){
		f= &lf;
	}
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
