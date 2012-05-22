/*
 * UX11 Utility Functions
 */

#include <stdio.h>
#include "ux11.h"
#include "ux11_internal.h"

static XVisualInfo *get_all_visuals();

int ux11_find_visual(disp, good_func, rtn_vis)
Display *disp;			/* What display to examine */
int (*good_func)();		/* Desirability function   */
XVisualInfo *rtn_vis;		/* VisualInfo to fill in   */
/*
 * Locates an appropriate color visual.  Uses `good_func' to evaluate 
 * all visuals.  The function has the following form:
 *   int good_func(vis)
 *   XVisualInfo *vis;
 * Should return the desirability of the visual (larger values
 * mean better visuals).  Returns a non-zero status if successful.
 */
{
    XVisualInfo *vlist;
    int num_vis, idx, max_cost, cost;
    XVisualInfo *chosen;

    vlist = get_all_visuals(disp, &num_vis);
    if (!vlist) return 0;

    max_cost = 0;
    chosen = NULL;
    memset( rtn_vis, 0, sizeof(XVisualInfo) );
    for (idx = 0;  idx < num_vis;  idx++) {
		if( (cost = (*good_func)(&(vlist[idx]))) && cost > max_cost) {
			max_cost = cost;
			chosen = &(vlist[idx]);
		}
    }
	if( chosen ){
		*rtn_vis = *chosen;
	    XFree(vlist);
		return(1);
	}
	else{
	    XFree(vlist);
		return(0);
	}
}

static XVisualInfo *get_all_visuals(disp, num)
Display *disp;
int *num;
/*
 * Gets the visual list for for the specified display.  Number
 * of items returned in `num'.
 */
{
    return XGetVisualInfo(disp, VisualNoMask, (XVisualInfo *) 0, num);
}

