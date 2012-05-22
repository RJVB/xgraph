
#include <cpu.h>
#ifdef MCH_AMIGA
#	include "graf_header.h"
#	ifndef _MXT_H
#		include <local/mxt.h>
#	endif
#	include "axes_.h"
#else
#	include "cx.h"
#	include "mxt.h"
#	include "graf_tool.h"
#	include "axes_.h"
#	include "macros.h"
#endif

#include "xfig/func.h"
#include "xfig/object.h"
#include "xfig/const.h"

#include "fig_save.h"

#include <sys/stat.h>

IDENTIFY( "FIG Axis and 3D routines");

#define ENT (integer)

grint F_xscreenmaxglb= 11 * PIX_PER_INCH-1 , F_ymaxglb= 8 * PIX_PER_INCH-1;

grint F_ox3d, F_oy3d;
grint F_xpixtot, F_ypixtot, F_zpixtot;
integer F_xpix, F_ypix, F_zpix;
extern integer front, Ax_Debug;
double F_xzrate, F_axglb3, F_ayglb3, F_azglb3, F_bxglb3, F_byglb3, F_bzglb3;
tridscreen F_tri;

#ifdef MCH_AMIGA
#	define GRAFOUT()	{}
#	define GRAFIN()	{}
#	undef Graf_Passive
#	undef Graf_Active
#	define Graf_Active(s)	(s=1)
#	define Graf_Passive(s)	(s=0)
#endif
#ifdef MACINTOSH
#	ifdef _MAC_THINKC_
#		define GRAFOUT()	grafout()
#		define GRAFIN()	grafin()
#	else
#		define GRAFOUT()	{}
#		define GRAFIN()	{}
#	endif
#endif

#ifdef _UNIX_C_
#	define Graf_Active(s)	(s=1)
#	define Graf_Passive(s)	(s=0)
#	define GRAFOUT()	{}
#	define GRAFIN()	{}
#endif

#ifdef _MAC_THINKC_
#	define _MAC_FINDER_
#endif
#ifdef _MAC_MPWC_
#	define _MAC_FINDER_
#endif

FILE *FigFile= NULL;

OpenFigFile(file_name, xe, ye)
char	*file_name;
int xe, ye;
{
	struct stat	file_status;

	if (*file_name == 0) {
	    fprintf( stderr, "No file.  Abort save operation.");
	    return( 0 );
    }
	if (stat(file_name, &file_status) == 0) { /* file exists */
	    if (file_status.st_mode & S_IFDIR) {
			fprintf( stderr, "\"%s\" is a directory\n", file_name);
			return( 0 );
		}
	    if (file_status.st_mode & S_IWRITE) { /* writing is permitted */
			if (file_status.st_uid != geteuid()) {
				fprintf( stderr, "\"%s\" permission is denied\n", file_name);
				return( 0 );
			}
		}
	    else {
			fprintf( stderr, "\"%s\" File is read only\n", file_name);
			return( 0 );
		}
	}
	else if (errno != ENOENT)
	    return( 0 );  /* file does exist but stat fails */

	if( !FigFile && !(FigFile = fopen(file_name, "w")) == NULL ){
	    fprintf( stderr, "Couldn't open file %s, %s\n", file_name, serror() );
	    return( 0 );
	}
	else {
		fprintf(fp, "%s\n", "#FIG 2.0");
		fprintf(fp, "%d %d\n", PIX_PER_INCH, 2);
		if( xe<= 0 ){
			xe= F_xscreenmaxglb;
		}
		else{
			F_xscreenmaxglb= xe;
		}
		if( ye<= 0 ){
			ye= F_ymaxglb;
		}
		else{
			F_ymaxglb= ye;
		}
		fprintf( fp, "%d -2 -2 %d %d\n", O_COMPOUND, xe+ 1, ye+ 1 );
		F_ds( -2, -2, xe+ 2, ye+ 2, 0);
    }
	return( 1);
}

CloseFigFile()
{
	if( FigFile ){
		fprintf( FigFile, "\n%d\n", O_END_COMPOUND);
		fclose( FigFile );
		FigFile= NULL;
	}
}

int F_colorglb_a, F_colorglb_b;
int F_depth;

/* # Drawing functions	*/

F_dp( x, y)
grint x, y;
{ F_line l;
  F_point p;
	p.x= x;
	p.y= y;
	p.next= NULL;
	l.type= T_POLYLINE;
	l.style= SOLID_LINE;
	l.thickness= 1;
	l.color= F_colorglb_a;
	l.depth= F_depth;
	l.style_val= 0.0;
	l.pen= 0; 
	l.area_fill= 0;
	l.radius= 0.0;
	l.for_arrow= NULL;
	l.back_arrow= NULL;
	l.points= &p;
	l.next= NULL;
	Fig_write_line( FigFile, &l);
}

F_ds( xb, yb, xe, ye, fill)
grint xb, yb, xe, ye;
int fill;
{ F_line l;
  F_point p[5];
	p[0].x= xb;
	p[0].y= yb;
	p[0].next= &p[1];
	p[1].x= xe;
	p[1].y= yb;
	p[1].next= &p[2];
	p[2].x= xe;
	p[2].y= ye;
	p[2].next= &p[3];
	p[3].x= xb;
	p[3].y= ye;
	p[3].next= &p[4];
	p[4].x= xb;
	p[4].y= yb;
	p[4].next= NULL;
	l.type= T_BOX;
	l.style= SOLID_LINE;
	l.thickness= 1;
	l.color= F_colorglb_a;
	l.depth= F_depth;
	l.style_val= 0.0;
	l.pen= 0; 
	l.area_fill= (fill)? F_colorglb_a : 0;
	l.radius= 0.0;
	l.for_arrow= NULL;
	l.back_arrow= NULL;
	l.points= p;
	l.next= NULL;
	Fig_write_line( FigFile, &l);
}

F_dl( xb, yb, xe, ye)
grint xb, yb, xe, ye;
{ F_line l;
  F_point p[2];
	p[0].x= xb;
	p[0].y= yb;
	p[0].next= &p[1];
	p[1].x= xe;
	p[1].y= ye;
	p[1].next= NULL;
	l.type= T_POLYLINE;
	l.style= SOLID_LINE;
	l.thickness= 1;
	l.color= F_colorglb_a;
	l.depth= F_depth;
	l.style_val= 0.0;
	l.pen= 0; 
	l.area_fill= 0;
	l.radius= 0.0;
	l.for_arrow= NULL;
	l.back_arrow= NULL;
	l.points= p;
	l.next= NULL;
	Fig_write_line( FigFile, &l);
}

FigFlush()
{
	fflush( FigFile );
}

F_drawtext( x, y, scale, txt, orn, align)
grint x, y;
int scale;
char *txt;
int orn, align;
{ F_text t;
	t.type= align;
	t.font= 14;
	t.size= 10;
	t.color= F_colorglb_a;
	t.depth= F_depth;
	t.angle= (orn)? 0.0 : 1.57;

	t.style= PLAIN;
	t.height= 7;	/* pixels */
	t.length= 6* strlen(txt);	/* pixels */
		t.height= 1;
		t.length= strlen(txt);
	t.base_x= x;
	t.base_y= y+ t.height;
	t.pen= 0;
	t.cstring= txt;
	t.next= NULL;
	Fig_write_text( FigFile, &t);
}

F_GetcolorRGB( cola, rgb)
int cola;
GrafRGB *rgb;
{
	Setcolorglb( cola, colorglb_b);
	return( GetcolorRGB( colorglb_a, rgb) );
}

F_Setcolorglb( cola, colb)
int cola, colb;
{ GrafRGB rgb;
	F_GetcolorRGB( cola, &rgb);
	F_colorglb_a= 21 - ((int)(20.0* (rgb.luminance / 256.0 )+ 0.5 ) );
	F_GetcolorRGB( colb, &rgb);
	F_colorglb_b= 21 - ((int)(20.0* (rgb.luminance / 256.0 )+ 0.5 ) );
}

F_point *Poly_F_point= NULL;
int Poly_F_points= 0;

Fig_AllocPoly( count)
int count;
{
	if( !Poly_F_point || count > Poly_F_points ){
		if( Poly_F_point ){
			free( Poly_F_point );
		}
		if( calloc_error( Poly_F_point, F_point, count) ){
			fprintf( stderr, "Fig_AllocPoly(%d): can't allocate buffer (%s)\n",
				count, serror()
			);
			Poly_F_point= NULL;
			Poly_F_points= 0;
			return( 0 );
		}
		else{
			Poly_F_points= count;
		}
	}
	return( count );
}

int F_FillPolygon( x, y, count)				/* draw and fill a polygon */
grint *x, *y;
int count;
{	int k;
	F_line l;
	
	if( count== 0 ){
		return( 0);
	}
	if( !Fig_AllocPoly( count ) ){
		return(0);
	}
	for( k= 0; k< count; k++){
		Poly_F_point[k].x= x[k];
		Poly_F_point[k].y= y[k];
		if( k< count- 1 ){
			Poly_F_point[k].next= &Poly_F_point[k+1];
		}
		else{
			Poly_F_point[k].next= NULL;
		}
	}
	l.type= T_POLYGON;
	l.style= SOLID_LINE;
	l.thickness= 1;
	l.color= F_colorglb_a;
	l.depth= F_depth;
	l.style_val= 0.0;
	l.pen= 0; 
	l.area_fill= F_colorglb_a;
	l.radius= 0.0;
	l.for_arrow= NULL;
	l.back_arrow= NULL;
	l.points= Poly_F_point;
	l.next= NULL;
	Fig_write_line( FigFile, &l);
	return( k);						/* return the number of points drawn */
}


int F_DrawPolygon( x, y, count)				/* draw a polygon; clipping */
grint *x, *y;
int count;
{	int k;
	F_line l;
	
	if( count== 0){
		return( 0);
	}
	if( !Fig_AllocPoly( count ) ){
		return(0);
	}
	for( k= 0; k< count; k++){
		Poly_F_point[k].x= x[k];
		Poly_F_point[k].y= y[k];
		if( k< count- 1 ){
			Poly_F_point[k].next= &Poly_F_point[k+1];
		}
		else{
			Poly_F_point[k].next= NULL;
		}
	}
	l.type= T_POLYGON;
	l.style= SOLID_LINE;
	l.thickness= 1;
	l.color= F_colorglb_a;
	l.depth= F_depth;
	l.style_val= 0.0;
	l.pen= 0; 
	l.area_fill= 0;
	l.radius= 0.0;
	l.for_arrow= NULL;
	l.back_arrow= NULL;
	l.points= Poly_F_point;
	l.next= NULL;
	return( k);
}


	
DEFUN( getenv_int, (char *var), int);

#define sqr(z)	(((z)<0)?-(z)*(z):(z)*(z))
#define cus_sqrt(z)	(((z)>=0)?sqrt((z)):-sqrt(-(z)))

DEFUN( axes_logcheck,( integer *log, double *mini, double *max, int a), void);

static grint XB= GRINT 0, YB= GRINT 0
/* 
	, XE= GRINT 639, YE= GRINT 511,
	*xb= &XB, *yb= &YB, *xe= &F_xscreenmaxglb, *ye= &F_ymaxglb
 */
;


  /* 921117: log==1 -> logscale log==2 -> square_root scale	*/

extern integer log_x_axis, log_y_axis, log_z_axis;
double F_MinX, F_MaxX, F_MinY, F_MaxY, F_MinZ, F_MaxZ;


DEFUN( big_checks, ( double mn, double mx, double mini, double max,
		double *y, integer *size, char *inde, integer *start, int log), integer);

void F_viewpoint( x, y)
grint x, y;
{
	if( x< GRINT 0 )
		x= GRINT( ( (double) F_xscreenmaxglb- GRINT 54)/ 2. + 0.5)+ GRINT 54- 
			GRINT( 0.1*F_xscreenmaxglb + 0.5);
	if( y< GRINT 0)
		y= GRINT( ( (double) F_ymaxglb- 20)/ 2. + 0.5)+ GRINT 10+
			GRINT( 0.1* F_ymaxglb + 0.5);
	F_ox3d= x;
	F_oy3d= y;
	F_ypixtot= ( F_xscreenmaxglb- F_ox3d);
	F_xpixtot= F_ox3d- GRINT 54;
	F_zpixtot= F_oy3d- GRINT 10;
	F_xzrate= (double) ( F_ymaxglb- GRINT 10- F_oy3d)/ F_xpixtot;
	if( Ax_Debug>= 2 ){
		GRAFOUT();
		fprintf( stderr, "F_viewpoint(%d,%d) pixels[%d,%d,%d] x/z ratio= %g\n",
			(int) F_ox3d, (int) F_oy3d,
			(int) F_xpixtot, (int) F_ypixtot, (int) F_zpixtot,
			F_xzrate
		);
		fflush( stderr);
		GRAFIN();
	}
}

void F_tricoords( x, y, z)
double x, y, z;
{	integer xpix, ypix, zpix;
	xpix= round( ( x- F_axglb3)* F_bxglb3);
	ypix= round( ( y- F_ayglb3)* F_byglb3);
	zpix= round( ( z- F_azglb3)* F_bzglb3);
	F_tri.x= F_ox3d+ GRINT( ypix- xpix);
	F_tri.y= GRINT round(  ((double)F_oy3d- (double)zpix+ F_xzrate* xpix) );
	if( Ax_Debug>= 3 ){
		GRAFOUT();
		fprintf( stderr, "(%g,%g,%g) [%d,%d,%d] -> (%d,%d)\n", x, y, z, 
			xpix, ypix, zpix,
			F_tri.x, F_tri.y);
		fflush( stderr);
		GRAFIN();
	}
}

void F_definetriworld( minix, miniy, miniz, maxx, maxy, maxz)
double minix, miniy, miniz, maxx, maxy, maxz;
{
	F_axglb3= minix;
	F_ayglb3= miniy;
	F_azglb3= miniz;
	F_bxglb3= F_xpixtot/ ( maxx- minix);
	F_byglb3= F_ypixtot/ ( maxy- miniy);
	F_bzglb3= F_zpixtot/ ( maxz- miniz);
}

void F_tri_line( xb, yb, zb, xe, ye, ze)
double xb, yb, zb, xe, ye, ze;
{
	grint xx, yy;
	
	F_tricoords( xb, yb, zb); 
	xx= F_tri.x; yy= F_tri.y;
	F_tricoords( xe, ye, ze);
	F_dl( xx, yy, F_tri.x, F_tri.y);
	FigFlush();
}

void F_tri_point( x, y, z)
double x, y, z;
{

	F_tricoords( x, y, z);
	F_dp( F_tri.x, F_tri.y);
}

void F_draw_front( dm)
int dm;
{ int fd= F_depth;
	F_depth= 0;
	F_tri_line( MaxX, MinY, MaxZ, MaxX, MaxY, MaxZ);
	F_tri_line( MaxX, MaxY, MaxZ, MinX, MaxY, MaxZ);
	F_tri_line( MaxX, MaxY, MaxZ, MaxX, MaxY, MinZ);
	F_tri_point( MaxX, MaxY, MaxZ);
	F_tri_point( MaxX, MaxY, MinZ);
	F_tri_point( MinX, MaxY, MaxZ);
	FigFlush();
	F_depth= fd;
}

F_ax_drawtext( x, y, scale, txt, orn, align)
grint x, y;
int scale;
char *txt;
int orn, align;
{  int fd= F_depth;
	F_depth= 0;
	F_drawtext( x, y, scale, txt, orn, align );
	F_depth= fd;
}

void F_triaxis( logx, logy, logz, minix, miniy, miniz, maxx, maxy, maxz)
int *logx, *logy, *logz;
double minix, miniy, miniz, maxx, maxy, maxz;
{
	grint i, j, zyb, yxe;
	double x, y, z, xx, yy, zz;
	char indeX[ 10], indeY[ 10], indeZ[ 10];
	double low, high, mnx, mny, mnz, mxx, mxy, mxz, dx, dy, dz;
	integer nx, ny, nz, size, start;

	Ax_Debug= getenv_int( "AXES-DEBUG");
	F_viewpoint( F_ox3d, F_oy3d);
	axes_logcheck( logx, &minix, &maxx, 'x');
	axes_logcheck( logy, &miniy, &maxy, 'y');
	axes_logcheck( logz, &miniz, &maxz, 'z');
	if( *logx== 1 ) { 
		mnx= log10( minix);
		mxx= log10( maxx);
	}
	else if( *logx== 2 ) { 
		mnx= cus_sqrt( minix);
		mxx= cus_sqrt( maxx);
	}
	else { 
		 mnx= minix;
		 mxx= maxx; 
	}
	if( *logy== 1 ) { 
		mny= log10( miniy);
		mxy= log10( maxy);
	}
	else if( *logy== 2 ) { 
		mny= cus_sqrt( miniy);
		mxy= cus_sqrt( maxy);
	}
	else { 
		mny= miniy;
		mxy= maxy; 
	}
	if( *logz== 1 ) { 
		mnz= log10( miniz);
		mxz= log10( maxz); 
	}
	else if( *logz== 2 ) { 
		mnz= cus_sqrt( miniz);
		mxz= cus_sqrt( maxz); 
	}
	else { 
		mnz= miniz;
		mxz= maxz; 
	}
	log_x_axis= *logx;
	log_y_axis= *logy;
	log_z_axis= *logz;
	MaxX= maxx= mxx;
	MaxY= maxy= mxy;
	MaxZ= maxz= mxz;
	MinX= minix= mnx;
	MinY= miniy= mny;
	MinZ= miniz= mnz;
	F_definetriworld(  mnx, mny, mnz, mxx, mxy, mxz);
	F_depth= 1000;
	F_tri_line( mnx, mny, mnz,
			  mxx, mny, mnz);
	F_tri_line( mnx, mxy, mnz,
			  mnx, mny, mnz);
	F_tri_line( mnx, mny, mnz,
			  mnx, mny, mxz);
	F_tri_line( mxx, mny, mxz,
			  mnx, mny, mxz);
	F_tri_line( mnx, mxy, mnz,
			  mnx, mxy, mxz);
	F_tri_line( mnx, mxy, mxz,
			  mnx, mny, mxz);
	F_tri_line( mxx, mxy, mnz,
			  mxx, mny, mnz);
	F_tri_line( mxx, mny, mnz,
			  mxx, mny, mxz);
	zyb= F_tri.y;
	F_tri_line( mnx, mxy, mnz,
			  mxx, mxy, mnz);
	yxe= F_tri.x;
	if( front ){
		F_draw_front( 2);
	}
	adapt_ranges( &mny, &mxy, &ny);
	y= mny; 
	if( *logy== 1)
		strcpy( indeY, sround( pow(10.0,mny), 1));
	else if( *logy== 2)
		strcpy( indeY, sround( sqr(mny), 1));
	else
		strcpy( indeY, sround(mny, 1));
	x= maxx;
	z= miniz;
	F_tricoords( x, y, z);
	F_dl( F_tri.x, F_tri.y, F_tri.x, F_tri.y+ GRINT 10);
	size= 6* strlen(indeY)+ 2;
	F_ax_drawtext( F_tri.x+ GRINT 3, F_ymaxglb- GRINT 7, 1, indeY, 1, T_LEFT_JUSTIFIED );
	y= mny;
	Ax_Debug= getenv_int( "AXES-DEBUG");
	start= 1;
	while( big_checks(mny, mxy, miniy, maxy, &y, &size, indeY, &start, *logy)){
		F_tricoords( maxx, y, miniz);
		F_dl( F_tri.x, F_tri.y, F_tri.x, F_tri.y+ GRINT 10);
		F_ax_drawtext( F_tri.x- GRINT 3, F_ymaxglb- GRINT 7, 1, indeY, 1, T_RIGHT_JUSTIFIED );
#ifdef MCH_AMIGA
		Chk_Abort();
#endif
	} 
	j= 0;
	dy= ( mxy- mny) / (double)ny;
	i= ( ny< 0)? -1 : 1;
	yy= MIN( mny, mxy);
	high= ( MAX( maxy, miniy)- dy);
	low= ( MIN( maxy, miniy)- dy);
	if( Ax_Debug){
		fprintf( stderr, "Y: putting some %d ticklets; low=%le high=%le start=%le delta=%le\n",
			ny, low, high, yy, dy
		);
		fflush(stderr);
	}
	do {			    /* put smal ticks */
		yy+= dy;
		F_tricoords( maxx, yy, miniz);
		F_dl( F_tri.x, F_tri.y- GRINT 2, F_tri.x, F_tri.y+ GRINT 2);
		j+= 1;
#ifdef MCH_AMIGA
		Chk_Abort();
#endif
	}
	while( dy!= 0 && yy+ dy<= high );
	yy= MIN( mny, mxy)- dy;
	F_tricoords( maxx, yy, miniz);
	while( F_tri.x>= GRINT 54 && F_tri.x <= yxe){
		F_dl(F_tri.x, F_tri.y- GRINT 2, F_tri.x, F_tri.y+ GRINT 2);
		yy-= dy;
		F_tricoords( maxx, yy, miniz);
#ifdef MCH_AMIGA
		Chk_Abort();
#endif
	}
	adapt_ranges( &mnz, &mxz, &nz);
	z= mnz; 
	if( *logz== 1)
		strcpy( indeZ, sround( pow(10.0,mnz), 1));
	else if( *logz== 2)
		strcpy( indeZ, sround( sqr(mnz), 1));
	else
		strcpy( indeZ, sround(mnz, 1));
	x= maxx;
	y= miniy;
	F_tricoords( x, y, z);
	F_dl( F_tri.x- GRINT 10, F_tri.y, F_tri.x, F_tri.y);
	size= 6* strlen(indeZ)+ 2;
	F_ax_drawtext( F_tri.x- GRINT 2, F_tri.y- GRINT 9, 1, indeZ, 1, T_RIGHT_JUSTIFIED );
	z= mnz;
	Ax_Debug= getenv_int( "AXES-DEBUG");
	start= 1;
	while( big_checks( mnz, mxz, miniz, maxz, &z, &size, indeZ, &start, *logz)){
		F_tricoords( maxx, miniy, z);
		F_dl( F_tri.x, F_tri.y, F_tri.x- GRINT 10, F_tri.y);
		F_ax_drawtext( F_tri.x- GRINT 2, F_tri.y+ GRINT 2, 1, indeZ, 1, T_RIGHT_JUSTIFIED );
#ifdef MCH_AMIGA
		Chk_Abort();
#endif
	}
	j= 0;
	dz= ( mxz- mnz) / nz;
	i= ( nz< 0)? -1 : 1;
	zz= MIN( mnz, mxz);
	high= ( MAX( maxz, miniz)- dz);
	low= ( MIN( maxz, miniz)- dz);
	if( Ax_Debug){
		fprintf( stderr, "Z: putting some %d ticklets; low=%le high=%le start=%le delta=%le\n",
			nz, low, high, zz, dz
		);
		fflush(stderr);
	}
	do {			    /* put smal ticks */
		zz+= dz;
		F_tricoords( maxx, miniy, zz);
		F_dl( F_tri.x- GRINT 2, F_tri.y, F_tri.x+ GRINT 2, F_tri.y);
		j+= 1;
#ifdef MCH_AMIGA
		Chk_Abort();
#endif
	}
	while( dz!= 0.0 && zz+dz<= high );
	zz= MIN( mnz, mxz)- dz;
	F_tricoords( maxx, miniy, zz);
	while( F_tri.y<= F_ymaxglb- GRINT 10 && F_tri.y>= zyb){
		F_dl(F_tri.x- GRINT 2, F_tri.y, F_tri.x+ GRINT 2, F_tri.y);
		zz-= dz;
		F_tricoords( maxx, miniy, zz);
#ifdef MCH_AMIGA
		Chk_Abort();
#endif
	}
	adapt_ranges( &mnx, &mxx, &nx);
	x= mnx; 
	if( *logx== 1)
		strcpy( indeX, sround( pow(10.0,mnx), 1));
	else if( *logx== 2)
		strcpy( indeX, sround( sqr(mnx), 1));
	else
		strcpy( indeX, sround(mnx, 1));
	y= miniy;
	z= maxz;
	F_tricoords( x, y, z);
	F_dl( F_tri.x-GRINT 15, F_tri.y, F_tri.x, F_tri.y);
	size= 6* strlen(indeX)- 8;
	F_ax_drawtext( GRINT round( (double)F_tri.x- 10.0/ F_xzrate)- GRINT 10,
					 F_tri.y+ GRINT 1, 1, indeX, 1, T_LEFT_JUSTIFIED
	);
	x= mnx;
	Ax_Debug= getenv_int( "AXES-DEBUG");
	start= 1;
	while( big_checks( mnx, mxx, minix, maxx, &x, &size, indeX, &start, *logx)){
		F_tricoords( x, miniy, maxz);
		if( Ax_Debug>= 2 ){
			fprintf( stderr, "label at (%g,%g,%g) -> [%d,%d]\n",
				x, miniy, maxz,
				(int) F_tri.x, (int) F_tri.y
			);
		}
		F_dl( F_tri.x, F_tri.y, F_tri.x- GRINT 15, F_tri.y);
		F_ax_drawtext( F_tri.x- GRINT 2, F_tri.y- GRINT 9, 1, indeX, 1, T_RIGHT_JUSTIFIED );
#ifdef MCH_AMIGA
		Chk_Abort();
#endif
	}
	j= 0;
	dx= ( mxx- mnx) / nx;
	i= ( nx< 0)? -1 : 1;
	xx= MIN( mnx, mxx);
	high= ( MAX( maxx, minix)- dx);
	low= ( MIN( maxx, minix)- dx);
	if( Ax_Debug){
		fprintf( stderr, "X: putting some %d ticklets; low=%le high=%le start=%le delta=%le\n",
			nx, low, high, xx, dx
		);
		fflush(stderr);
	}
	do {			    /* put smal ticks */
		xx+= dx;
		F_tricoords( xx, miniy, maxz);
		F_dl( F_tri.x- GRINT 4, F_tri.y, F_tri.x+ GRINT 4, F_tri.y);
		j+= 1;
#ifdef MCH_AMIGA
		Chk_Abort();
#endif
	}
	while( dx!= 0.0 && xx+ dx<= high );
	xx= MIN(mnx, mxx)- dx;
	F_tricoords( xx, miniy, maxz);
	while( F_tri.x<= F_ox3d && F_tri.x>= GRINT 54){
		F_dl(F_tri.x- GRINT 3, F_tri.y, F_tri.x+ GRINT 2, F_tri.y);
		xx-= dx;
		F_tricoords( xx, miniy, maxz);
#ifdef MCH_AMIGA
		Chk_Abort();
#endif
	}
	FigFlush();
	F_depth= 0;
}

#define clog10(z) ((z>0)?log10(z):z)

#define check_tri() tri_ok++

void F_funplot3( fun, minix, miniy, maxx, maxy, dx, dy, transform)
DEFMETHOD( fun,(double *x, double *y, Poly3D *p), double), minix, miniy, maxx, maxy;
int dx, dy;
DEFMETHOD( transform,(integer n, double *x, double *y, double *z), int);
{
	grint X[5], Y[5];
	grint *xyX, *xyY;
	double x, y, ddx, ddy, sum= 0.0;
	integer k, j;
	int col, acol, bcol;
	int i;
	Poly3D *p, *q, *r;
	Poly3D pol, pol2, pol3;
	GrafRGB rgb;
	char *axes_debug;
	int tri_ok;
	DEFUN( no_transform, (integer, double *, double *, double *), int);

	if( !fun)
		return;
	if( !transform)
		transform= no_transform;	
	axes_debug= getenv( "AXES-DEBUG");
	ddx= (maxx- minix)/ dx;
	ddy= ( maxy- miniy)/ dy;
	acol= F_colorglb_a;
	bcol= F_colorglb_b;
	p= &pol;
	q= &pol2;
	r= &pol3;
	f3x= Choose_f3_plot_method( NULL, log_x_axis );
	f3y= Choose_f3_plot_method( NULL, log_y_axis );
	f3z= Choose_f3_plot_method( NULL, log_z_axis );
	F_depth= dx;
	for( x= minix, i= 0; i< dx; i++, x+= ddx){
#ifdef MCH_AMIGA
		Chk_Abort();
#endif
		for( y= miniy, j= 0; j< dy; j++, y+= ddy){
			p->x[3]= p->x[0]= x;
			p->y[1]= p->y[0]= y;
			p->x[2]= p->x[1]= x+ ddx;
			p->y[3]= p->y[2]= y+ ddy;
			p->points= 1;
			p->z[0]= (*fun)( p->x, p->y, p);
			p->points= 2;
			p->z[1]= (*fun)( &p->x[1], p->y, p);
			p->points= 3;
			p->z[2]= (*fun)( &p->x[2], &p->y[2], p);
			p->points= 4;
			p->z[3]= (*fun)( p->x, &p->y[3], p);
			col= funplot3_colour( p);

			for( k= 0; k< 4; k++){
				p->x[k]= (*f3x)( p->x[k] );
				p->y[k]= (*f3y)( p->y[k] );
				p->z[k]= (*f3z)( p->z[k] );
			}
			if( !(*transform)( 4, p->x, p->y, p->z))
				goto next;
			xyX= X; xyY= Y;
			tri_ok= 0;
			F_tricoords( p->x[0], p->y[0], p->z[0]);
			check_tri();
			sum= p->z[0]; 
			*xyX++= F_tri.x;
			*xyY++= F_tri.y;
			X[4]= F_tri.x;
			Y[4]= F_tri.y;
			F_tricoords( p->x[1], p->y[1], p->z[1]);
			check_tri();
			sum+= p->z[1];
			*xyX++= F_tri.x;
			*xyY++= F_tri.y;
			F_tricoords( p->x[2], p->y[2], p->z[2]);
			check_tri();
			sum+= p->z[2];
			*xyX++= F_tri.x;
			*xyY++= F_tri.y;
			F_tricoords( p->x[3], p->y[3], p->z[3]);
			check_tri();
			sum+= p->z[3];
			*xyX++= F_tri.x;
			*xyY++= F_tri.y;
			if( axes_debug ){
				fprintf( stderr,
					"point (%le,%le,%le) Poly\n\t{(%ld,%ld)\n\t (%ld,%ld)\n\t (%ld,%ld)\n\t (%ld,%ld)\n\t (%ld,%ld)\n\t}\n",
						p->x[0], p->y[0], p->z[0],
						(long) X[0], (long) Y[0], (long) X[1], (long) Y[1],
						(long) X[2], (long) Y[2], (long) X[3], (long) Y[3], (long) X[4], (long) Y[4]
				);
			}
			if( funplot3_colours){
				F_Setcolorglb( col, 0);
			}
			else{
				F_Setcolorglb( maxcolors-1, 0);
			}
			if( tri_ok){
				F_FillPolygon( X, Y, 5);
			}
			acol= F_colorglb_a;
			bcol= F_colorglb_b;
			F_GetcolorRGB( acol, &rgb);
			if( rgb.luminance< MinVisLuminance )
				col= 0;					/* r,g,b < MAXCOLOURS: black	*/
			else
				col= 1;
			if( funplot3_colours && !col)
				col= maxcolors- 1;
			else
				col= 0;
			if( tri_ok){
				F_Setcolorglb( col, 0);
				F_DrawPolygon( X, Y, 5);
				F_Setcolorglb( acol, bcol);
			}
			FigFlush();
		}
next:;
		F_depth-= 1;
	}
	if( front){
		F_Setcolorglb( 1, 0);
		F_draw_front( 1);
	}
}

extern struct Remember *GraftoolKey;

extern int tripoint_array_colours, tripoint_array_cross, tripoint_array_normal;
extern unsigned long tripoint_drawn_edges;

void F_plot_tripoint_array( tp, xelem, yelem, N)
tridpoint **tp;
int xelem, yelem;
unsigned long N;
{ integer i, k, j;
  unsigned long n;
  grint X[5], Y[5];
  grint *xyX, *xyY;
  double sum= 0.0;
  int col, acol, bcol;
  Poly3D *p, *q, *r;
  Poly3D pol, pol2, pol3;
  GrafRGB rgb;
  int Colour, tri_ok;

	if( !tp)
		return;
	acol= F_colorglb_a;
	bcol= F_colorglb_b;
	p= &pol;
	q= &pol2;
	r= &pol3;
	tripoint_drawn_edges= 0;
	f3x= Choose_f3_plot_method( NULL, log_x_axis );
	f3y= Choose_f3_plot_method( NULL, log_y_axis );
	f3z= Choose_f3_plot_method( NULL, log_z_axis );
	F_depth= xelem-1;
	for( n= 0, j= 0; j< xelem-1 && n< N; j++){
	  tridpoint *TP= tp[j], *TP_1= tp[j+1];	/* TP: gy[i] - TP_1: gy+1[i]	*/
	  int visible;
#ifdef MCH_AMIGA
		Chk_Abort();
#endif
		for( i= 0; i< yelem-1 && n< N; i++, n++ ){
			p->x[0]= TP[i].x;		/* gx,gy	*/
			p->y[0]= TP[i].y;
			p->z[0]= TP[i].z;
			visible= TP[i].visible;

			p->x[1]= TP_1[i].x;		/* gx,gy+1	*/
			p->y[1]= TP_1[i].y;
			p->z[1]= TP_1[i].z;
			visible*= TP_1[i].visible;

			p->x[2]= TP_1[i+1].x;	/* gx+1,gy+1	*/
			p->y[2]= TP_1[i+1].y;
			p->z[2]= TP_1[i+1].z;
			visible*= TP_1[i+1].visible;

			p->x[3]= TP[i+1].x;		/* gx+1,gy	*/
			p->y[3]= TP[i+1].y;
			p->z[3]= TP[i+1].z;
			visible*= TP[i+1].visible;

			switch( tripoint_array_colours){
				case 0:
					p->points= 4;
					Colour= funplot3_colour( p);
					break;
				case 1:
					Colour= (int) TP[i].colour;
					Colour+= (int) TP_1[i].colour;
					Colour+= (int) TP_1[i+1].colour;
					Colour+= (int) TP[i+1].colour;
					Colour/= 4;
					break;
				default:
					Colour= (int) TP[i].colour;
					break;
			}

			if( !visible ){
				goto next3dpoint;
			}

			for( k= 0; k< 4; k++ ){
				p->x[k]= (*f3x)( p->x[k] );
				p->y[k]= (*f3y)( p->y[k] );
				p->z[k]= (*f3z)( p->z[k] );
			}

			xyX= X; xyY= Y;
			tri_ok= 0;
			F_tricoords( p->x[0], p->y[0], p->z[0]);
			check_tri();
			sum= p->z[0]; 
			*xyX++= F_tri.x;
			*xyY++= F_tri.y;
			X[4]= F_tri.x;
			Y[4]= F_tri.y;

			F_tricoords( p->x[1], p->y[1], p->z[1]);
			check_tri();
			sum+= p->z[1];
			*xyX++= F_tri.x;
			*xyY++= F_tri.y;

			F_tricoords( p->x[2], p->y[2], p->z[2]);
			check_tri();
			sum+= p->z[2];
			*xyX++= F_tri.x;
			*xyY++= F_tri.y;

			F_tricoords( p->x[3], p->y[3], p->z[3]);
			check_tri();
			sum+= p->z[3];
			*xyX++= F_tri.x;
			*xyY++= F_tri.y;

			if( funplot3_colours){
				col= Colour;
				F_Setcolorglb( col, 0);
			}
			else{
				F_Setcolorglb( maxcolors-1, 0);
			}
			if( tri_ok){
				tripoint_drawn_edges+= (unsigned long) F_FillPolygon( X, Y, 5)- 1;
			}
			acol= F_colorglb_a;
			bcol= F_colorglb_b;
			F_GetcolorRGB( acol, &rgb);
			if( rgb.luminance< MinVisLuminance )
				col= 0;					/* r,g,b < MAXCOLOURS: black	*/
			else
				col= 1;
			if( funplot3_colours && !col)
				col= maxcolors- 1;
			else
				col= 0;
			if( tri_ok){
				F_Setcolorglb( col, 0);
				tripoint_drawn_edges+= (unsigned long) F_DrawPolygon( X, Y, 5)- 1;
				if( tripoint_array_cross ){
					F_dl( X[3], Y[3], X[1], Y[1]);
					F_dl( X[0], Y[0], X[2], Y[2]);
				}
				if( tripoint_array_normal ){
					F_Setcolorglb( maxcolors-1, 0);
					F_dl( X[0], Y[0],
						(grint) (X[0]+ TP[i].distance* Cos( TP[i].normal )),
						(grint) (Y[0]- TP[i].distance* Sin( TP[i].normal ))
					);
				}
				F_Setcolorglb( acol, bcol);
			}
			FigFlush();
next3dpoint:;
		}
		F_depth-= 1;
	}
	if( front){
		F_Setcolorglb( 1, 0);
		F_draw_front( 1);
	}
}
