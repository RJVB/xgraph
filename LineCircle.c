/* utility functions for finding the roots of a quadratic equation and
 \ for finding the intersection between a line and a circle.
 \ By Mark Wexler.
 */

#define BETWEEN01(s)			((s) >= 0 && (s) <= 1)

/*
 \ assigns the real roots of a x^2 + b x + c = 0 to p1 and p2, so that p1 < p2
 \ returns the number of roots. Uses the method described in the NR, 5-6.
*/
#ifdef __GNUC__
inline
#endif
int quadratic_roots(const double a, const double b, const double c, double *p1, double *p2)
{ double d = b*b - 4.0*a*c;
  int r;
	if( d < 0.0 ){
		r= 0;
	}
	else if( a== 0 ){
		if( b ){
			*p1= -c/ b;
			r= 1;
		}
		else{
			r= 0;
		}
	}
	else if( d == 0.0 ){
		*p1 = -b/(2.0*a);
		r= 1;
	}
	else{
	  double q, x1, x2;
		q = -0.5*(b + ((double) SIGN(b))*sqrt(d));
		x1 = q/a;
		x2 = c/q;
		if(x1 < x2) {
			*p1 = x1;
			*p2 = x2;
		}
		else {
			*p1 = x2;
			*p2 = x1;
		}
		r= 2;
	}
	return(r);
}

#ifdef __GNUC__
inline
#endif
double POW2( double a )
{
	return( a * a );
}

int clip_line_by_circle(const double line0[2], const double line1[2],
					const double centre[2], const double radius,
					double begin[2], double end[2])
{ double a = POW2(line0[0] - line1[0]) + POW2(line0[1] - line1[1]);
  double b = 2*(centre[0] - line0[0])*(line0[0] - line1[0]) +
			   2*(centre[1] - line0[1])*(line0[1] - line1[1]);
  double c = POW2(centre[0] - line0[0]) +
			   POW2(centre[1] - line0[1]) - POW2(radius);
  double s1, s2;
  int n = quadratic_roots(a, b, c, &s1, &s2);
	if( n > 0 ){
		if( n== 1 ){
			s2= s1;
		}
		if( s1 > 1 || s2 < 0 ){
		 /* completely outside */
			n= 0;
		}
		else if( s1 < 0 && s2 > 1 ){
		  /* completely inside */
			memcpy( begin, line0, 2*sizeof(double));
			memcpy( end, line1, 2*sizeof(double));
		}
		else if( s1 < 0 && BETWEEN01(s2) ){
		 /* line0 inside, line1 outside */
			memcpy( begin, line0, 2*sizeof(double));
#ifdef __cplusplus
			end = line0 + s2*(line1 - line0);
#else
			end[0] = line0[0] + s2*(line1[0] - line0[0]);
			end[1] = line0[1] + s2*(line1[1] - line0[1]);
#endif
		}
		else if( BETWEEN01(s1) && s2 > 1 ){
		 /* line0 outside, line1 inside */
#ifdef __cplusplus
			begin = line0 + s1*(line1 - line0);
#else
			begin[0] = line0[0] + s1*(line1[0] - line0[0]);
			begin[1] = line0[1] + s1*(line1[1] - line0[1]);
#endif
			memcpy( end, line1, 2*sizeof(double));
		}
		else if( BETWEEN01(s1) && BETWEEN01(s2) ){
		  /* midsection inside */
#ifdef __cplusplus
			begin = line0 + s1*(line1 - line0);
			end = line0 + s2*(line1 - line0);
#else
			begin[0] = line0[0] + s1*(line1[0] - line0[0]);
			begin[1] = line0[1] + s1*(line1[1] - line0[1]);
			end[0] = line0[0] + s2*(line1[0] - line0[0]);
			end[1] = line0[1] + s2*(line1[1] - line0[1]);
#endif
		}
		else{
#ifdef __cplusplus
			throw(error("geometry", "problem with clip_line_by_circle, s1=%g, s2=%g", s1, s2));
#endif
			n= 0;
		}
	}
	return(n);
}


int clip_line_by_circle2( double slope, double intercept,
					const double centre[2], const double radius,
					double begin[2], double end[2])
{ double line0[2], line1[2];
	
	if( Inf(slope) ){
		line0[0]= line1[0]= intercept;
		line0[1]= centre[1]- radius*2;
		line1[1]= centre[1]+ radius*2;
	}
	else{
		line0[0]= centre[0]- radius*2;
		line0[1]= slope* line0[0]+ intercept;
		line1[0]= centre[0]+ radius*2;
		line1[1]= slope* line1[0]+ intercept;
	}
	return( clip_line_by_circle( line0, line1, centre, radius, begin, end ) );
}
