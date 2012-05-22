#ifndef _FIG_GRAPHICS_H

#include "fig_save.h"

extern grint F_xscreenmaxglb, F_ymaxglb;
extern grint F_ox3d, F_oy3d;
extern grint F_xpixtot, F_ypixtot, F_zpixtot;
extern integer F_xpix, F_ypix, F_zpix;
extern double F_xzrate, F_axglb3, F_ayglb3, F_azglb3, F_bxglb3, F_byglb3, F_bzglb3;
extern tridscreen F_tri;


extern FILE *FigFile;
A_FUN( OpenFigFile, (char *name, int xe, int ye) );
A_FUN( CloseFigFile, () );

DEFUN( F_DrawPolygon,( grint *x, grint *y, int count), int);
DEFUN( F_FillPolygon,( grint *x, grint *y, int count), int);
A_FUN( F_Setcolorglb, ( int cola, int colb));
DEFUN( F_definetriworld,( double minix, double miniy, double miniz, double maxx, double maxy, double maxz), void);
A_FUN( F_dl,( grint xb, grint yb, grint xe, grint ye) );
A_FUN( F_ds,( grint xb, grint yb, grint xe, grint ye, int fill) );
DEFUN( F_draw_front, ( int dm), void);
A_FUN( F_drawtext, ( grint x, grint y, int scale, char *txt, int orn, int align) );
DEFUN( F_tri_line, ( double xb, double yb, double zb, double xe, double ye, double ze), void);
DEFUN( F_tri_point, ( double x, double y, double z), void);
DEFUN( F_triaxis, ( int *logx, int *logy, int *logz,
		double minix, double miniy, double miniz, double maxx, double maxy, double maxz), void);
DEFUN( F_tricoords, ( double x, double y, double z), void);
DEFUN( F_viewpoint, ( grint x, grint y), void);
A_FUN( FigFlush, ());

DEFUN( F_funplot3, (DEFMETHOD(fun,(double *x, double *y, Poly3D *p),double), double minx, double miny, double maxx,\
			double maxy, int dx, int dy,\
			DEFMETHOD(transform,(integer n, double *x, double *y, double *z), int )\
		), void); 
DEFUN( F_plot_tripoint_array, (tridpoint **tp, int xelem, int yelem, unsigned long N), void);

#define _FIG_GRAPHICS_H
#endif
