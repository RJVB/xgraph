/*
 * X11 Utility functions
 */

#define _MAIN_C

#include <stdio.h>
#include "xgerrno.h"

#include "ux11.h"
#include "ux11_internal.h"

#ifdef _HAVE_XINERAMA_
#	include <X11/extensions/Xinerama.h>
#endif

Display *ux11_open_display(argc, argv, display_specified )
int argc;
char *argv[];
int *display_specified;
/* Looks for -display option */
{
    Display *disp= NULL;
    char *disp_name= NULL;
    extern char *getenv();

      /* see user specified a "-display <name>" argument.	*/
    if( (disp_name = ux11_get_value(argc, argv, "-display", NULL ) ) ){
		*display_specified= 1;
	}
	else{
	  char *c= getenv("DISPLAY");
	  /* he didn't - check the environment.	*/
		if( c ) {
			disp_name= strdup( c );
			*display_specified= 0;
		}
	}
    if( disp_name )
	    disp = XOpenDisplay( disp_name );
    if (!disp) {
		fprintf(stderr, "%s: cannot open display `%s' (%s)\n",
			argv[0], (disp_name)? disp_name : "<none given>",
			serror()
		);
		exit(10);
    }
    if( disp_name && !*display_specified ){
	    free( disp_name );
    }
	XSynchronize( disp, 0);
    return disp;
}

#ifdef _HAVE_XINERAMA_
XineramaScreenInfo *ui_HeadInfo= NULL;
int	ui_NumHeads= 1;

int ux11_multihead_DisplayWidth( Display *d, int scr, int x_centre, int y_centre, int *base_x, int *base_y, int *head )
{ int i;
	
	if( !ui_HeadInfo ){
	  int event_base, error_base;
		if( XineramaQueryExtension( d, &event_base, &error_base ) ){
			ui_HeadInfo= XineramaQueryScreens( d, &ui_NumHeads );
		}
		else{
			ui_HeadInfo= NULL;
			ui_NumHeads= -1;
		}
	}
	if( ui_HeadInfo && ui_NumHeads> 0 ){
	  long dist, min_dist;
	  int best= 0;
		  /* Find the head containing the window's specified centre, and
		   \ return the head's width and offset on the 'total screen'.
		   */
		min_dist= dist= abs( (ui_HeadInfo[0].x_org + ui_HeadInfo[0].width/2 - x_centre) +
			(ui_HeadInfo[0].y_org + ui_HeadInfo[0].height/2 - y_centre) );
		for( i= 1; i< ui_NumHeads; i++ ){
			dist= abs( (ui_HeadInfo[i].x_org + ui_HeadInfo[i].width/2 - x_centre) +
				(ui_HeadInfo[i].y_org + ui_HeadInfo[i].height/2 - y_centre) );
			if( dist< min_dist ){
				min_dist= dist;
				best= i;
			}
		}
		if( head ){
			*head= ui_HeadInfo[best].screen_number;
		}
		if( base_x ) *base_x= ui_HeadInfo[best].x_org;
		if( base_y ) *base_y= ui_HeadInfo[best].y_org;
		return( ui_HeadInfo[best].width );
	}
	if( head ){
		*head= 0;
	}
	if( base_x ) *base_x= 0;
	if( base_y ) *base_y= 0;
	return( XDisplayWidth(d, scr) );
}

int ux11_multihead_DisplayHeight( Display *d, int scr, int x_centre, int y_centre, int *base_x, int *base_y, int *head )
{ int i;

	if( !ui_HeadInfo ){
	  int event_base, error_base;
		if( XineramaQueryExtension( d, &event_base, &error_base ) ){
			ui_HeadInfo= XineramaQueryScreens( d, &ui_NumHeads );
		}
		else{
			ui_HeadInfo= NULL;
			ui_NumHeads= -1;
		}
	}
	if( ui_HeadInfo && ui_NumHeads> 0 ){
	  long dist, min_dist;
	  int best= 0;
		  /* Find the head containing the window's specified centre, and
		   \ return the head's width and offset on the 'total screen'.
		   */
		min_dist= dist= abs( (ui_HeadInfo[0].x_org + ui_HeadInfo[0].width/2 - x_centre) +
			(ui_HeadInfo[0].y_org + ui_HeadInfo[0].height/2 - y_centre) );
		for( i= 1; i< ui_NumHeads; i++ ){
			dist= abs( (ui_HeadInfo[i].x_org + ui_HeadInfo[i].width/2 - x_centre) +
				(ui_HeadInfo[i].y_org + ui_HeadInfo[i].height/2 - y_centre) );
			if( dist< min_dist ){
				min_dist= dist;
				best= i;
			}
		}
		if( head ){
			*head= ui_HeadInfo[best].screen_number;
		}
		if( base_x ) *base_x= ui_HeadInfo[best].x_org;
		if( base_y ) *base_y= ui_HeadInfo[best].y_org;
		return( ui_HeadInfo[best].height );
	}
	if( head ){
		*head= 0;
	}
	if( base_x ) *base_x= 0;
	if( base_y ) *base_y= 0;
	return( XDisplayHeight(d, scr) );
}

#else

int ux11_multihead_DisplayWidth( Display *d, int scr, int x_centre, int y_centre, int *base_x, int *base_y, int *head )
{
	return( XDisplayWidth(d, scr) );
}

int ux11_multihead_DisplayHeight( Display *d, int scr, int x_centre, int y_centre, int *base_x, int *base_y, int *head )
{
	return( XDisplayHeight(d, scr) );
}

#endif
