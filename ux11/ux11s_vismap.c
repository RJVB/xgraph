/*
 * X11 Utility Functions
 */

#include <stdio.h>
#include "ux11.h"
#include "ux11_internal.h"

extern FILE *StdErr;

#define UX11_MIN_DEPTH	8

#undef ux11_min_depth
#undef ux11_vis_class
#undef ux11_useDBE

static int ux11_min_depth= UX11_MIN_DEPTH, ux11_vis_class= -1, ux11_useDBE= 0;

int *__ux11_min_depth()
{
	return( &ux11_min_depth );
}

int *__ux11_vis_class()
{
	return( &ux11_vis_class );
}

int *__ux11_useDBE()
{
	return( &ux11_useDBE );
}

static int ok_depth;

static int sort_xvi( XdbeVisualInfo *a, XdbeVisualInfo *b )
{
	return( a->perflevel - b->perflevel );
}

static int sort_xsvi( XdbeScreenVisualInfo *a, XdbeScreenVisualInfo *b )
{
	return( a->visinfo->perflevel - b->visinfo->perflevel );
}

int ux11_std_vismap( Display *disp, Visual **rtn_vis, Colormap *rtn_cmap, int *rtn_scrn, int *rtn_depth, int search)
/*
 * This routine tries to find a visual/colormap pair that
 * supports color for `disp'.  The following steps are
 * used to determine this pair:
 *  1.  The default depth of the default screen is examined.
 *      If it is more than four,  the default visual and
 *      colormap for the display is returned, if also
 *      user didn't request a specific visual type.
 *  2.  ux11_find_visual is used to see if there is a good
 *      alternate visual available (better than the default).
 *      If so,  a new colormap is made for the visual
 *      and it is returned.  If no good alternative is
 *      found,  the routine returns the default visual/colormap.
 * The routine returns zero if unsuccessful.  It returns UX11_DEFAULT
 * if the default is returned,  and UX11_ALTERNATE if a non-defualt
 * visual/colormap is returned (UX11_ALTERNATE_RW if a read/write
 * colourmap has been allocated).
 */
{
    int def_depth;
    XVisualInfo info;

    def_depth = DefaultDepth(disp, DefaultScreen(disp));
    memset( &info, 0, sizeof(XVisualInfo) );
    if( def_depth >= ux11_min_depth && !ux11_useDBE &&
		(ux11_vis_class< 0 || ux11_vis_class== DefaultVisual(disp,DefaultScreen(disp))->class)
	){
	/* Plenty and sufficient default resources */
		*rtn_vis = DefaultVisual(disp, DefaultScreen(disp));
		*rtn_cmap = DefaultColormap(disp, DefaultScreen(disp));
		*rtn_scrn = DefaultScreen(disp);
		*rtn_depth = DefaultDepth(disp, DefaultScreen(disp));
		return UX11_DEFAULT;
    }
	else if( ux11_useDBE ){
	  int maj, min;
		if( XdbeQueryExtension( disp, &maj, &min) ){
		  int i, j, n= 0, screendepth= (ux11_min_depth> 0)? ux11_min_depth : def_depth, perflevel= 0;
		  XdbeScreenVisualInfo *xsvi;
		  Drawable screen;
		  XVisualInfo *vi, templ;
		  long mask= VisualIDMask;
		  int N= 0;
			if( (xsvi= XdbeGetVisualInfo( disp, &screen, &n)) ){
				  /* Sort the returned visual structs according to increasing performance */
				for( i= 0; i< n; i++ ){
					qsort( xsvi[i].visinfo, xsvi[i].count, sizeof(XdbeVisualInfo), sort_xvi );
				}
				templ.visualid= 0;
				i= 0;
				  /* Find the visual with at least the required depth, and the highest perflevel
				   \ performance hint:
				   */
				for( j= 0; j< n; j++ ){
					do{
//						fprintf( StdErr, "xsvi[%d].visinfo[%d].visual=%d depth=%d perflevel=%d\n",
//							   j, i, xsvi[j].visinfo[i].visual, xsvi[j].visinfo[i].depth, xsvi[j].visinfo[i].perflevel
//						);
#ifdef CRASH_XQUARTZ
						if( xsvi[j].visinfo[i].depth>= screendepth &&
							(!templ.visualid || xsvi[j].visinfo[i].perflevel> perflevel)
						){
							templ.visualid = xsvi[j].visinfo[i].visual;
							perflevel = xsvi[j].visinfo[i].perflevel;
						}
#else
						if( xsvi[j].visinfo[i].depth>= screendepth &&
							(!templ.visualid || xsvi[j].visinfo[i].perflevel>= perflevel)
						){
							templ.visualid = xsvi[j].visinfo[i].visual;
							perflevel = xsvi[j].visinfo[i].perflevel;
							// 20101116: select the visual with the deepest screen:
							screendepth = xsvi[j].visinfo[i].depth;
						}
#endif
						i++;
					} while( i< xsvi[j].count );
				}
				if( !templ.visualid ){
					templ.visualid= xsvi[0].visinfo[0].visual;
				}
				XdbeFreeVisualInfo( xsvi );
				if( (vi= XGetVisualInfo( disp, mask, &templ, &N)) ){
					info= *vi;
					ux11_useDBE= -1;
					goto found_required;
				}
				else{
					ux11_useDBE= False;
				}
			}
			else{
				ux11_useDBE= False;
			}
			if( !ux11_useDBE ){
				fprintf( stderr,
					"ux11_std_vismap(): display has the Double-Buffer extension,"
					" but no visual supporting it.\n"
				);
				goto find_required;
			}
		}
		else{
			ux11_useDBE= False;
			fprintf( stderr, "ux11_std_vismap(): display does not have the Double-Buffer extension\n" );
			goto find_required;
		}
	}
	else{
	  /* Try to find another suitable visual */
	  int found= 0;
find_required:;
		if( !(search || ux11_vis_class< 0) ){
			goto use_specified;
		}
		ok_depth= 0;
		found= ux11_find_visual(disp, ux11_color_vis, &info);
		if( !found && ux11_vis_class>= 0 ){
		  int uvc= ux11_vis_class;
			ux11_vis_class= -1;
			ok_depth= 0;
			found= ux11_find_visual(disp, ux11_color_vis, &info);
			ux11_vis_class= uvc;
		}
		if( found ){
		  int alloc;
found_required:;
			*rtn_vis = info.visual;
			*rtn_scrn = info.screen;
			*rtn_depth = info.depth;

			  /* New colormap required */
			switch( info.class ){
				case GrayScale:
				case PseudoColor:
				case DirectColor:
/* 						alloc= AllocAll;	*/
/* 						break;	*/
				case StaticGray:
				case StaticColor:
				case TrueColor:
				default:
					alloc= AllocNone;
					break;
			}
			*rtn_cmap = XCreateColormap(disp,
							RootWindow(disp, info.screen),
							info.visual, alloc
			);
			XInstallColormap( disp, *rtn_cmap );
			if (*rtn_cmap) {
				return ( (alloc== AllocAll)? UX11_ALTERNATE_RW : UX11_ALTERNATE);
			} else {
				return 0;
			}
		} else {
			/* Back to the default */
use_specified:;
				*rtn_vis = DefaultVisual(disp, DefaultScreen(disp));
				*rtn_cmap = DefaultColormap(disp, DefaultScreen(disp));
				*rtn_scrn = DefaultScreen(disp);
				*rtn_depth = DefaultDepth(disp, DefaultScreen(disp));
				return UX11_DEFAULT;
			}
    }
    return 0;
}



int ux11_color_vis(vis)
XVisualInfo *vis;		/* Visual to examine */
/*
 * Returns a desirability index for the passed visual.
 * This functions preference list is:
 *   PsuedoColor
 *   DirectColor
 *   TrueColor
 *   StaticColor
 *   GrayScale
 *   StaticGray
 */
{
	if( vis->depth< ux11_min_depth ){
		return(0);
	}
	else if( !ok_depth ){
		ok_depth= vis->depth;
	}
	else if( vis->depth> ok_depth && ux11_vis_class< 0 ){
		return(0);
	}
	if( ux11_vis_class >= 0 && vis->class!= ux11_vis_class ){
		return(0);
	}
	switch (vis->class) {
		case PseudoColor:
				return(6);
/* 			return vis->colormap_size * 100;	*/
			break;
		case DirectColor:
				return(5);
/* 			return vis->depth * 1000;	*/
			break;
		case StaticColor:
				return(4);
/* 			return vis->colormap_size * 50;	*/
			break;
		case TrueColor:
				return(3);
/* 			return vis->depth * 500;	*/
			break;
		case GrayScale:
				return(2);
/* 			return vis->colormap_size * 25;	*/
			break;
		case StaticGray:
				return(1);
/* 			return vis->depth * 250;	*/
			break;
		default:
			fprintf(stderr, "Unknown visual type: %d\n", vis->class);
			abort();
	}
}
