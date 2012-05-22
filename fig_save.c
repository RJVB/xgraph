/*
vi:set ts=4|set sw=4:
:ts=4
*/
/* 
 *	FIG : Facility for Interactive Generation of figures
 *
 *	Copyright (c) 1985, 1988 by Supoj Sutanthavibul (supoj@sally.UTEXAS.EDU)
 *	January 1985.
 *	1st revision : August 1985.
 *	2nd revision : March 1988.
 *
 *	%W%	%G%
*/
/* Modified for "programmer" use by RJB, 931209 ...	*/

/* #include "xfig/fig.h"	*/
/* #include "xfig/resources.h"	*/
#include "xfig/func.h"
#include "xfig/object.h"
#include "xfig/const.h"

#include <local/cx.h>
#include <sys/stat.h>

IDENTIFY( "Save FIG objects");

extern F_compound	objects;

extern int		figure_modified;
extern int		num_object;
extern 			null_proc();

Fig_write_arc(fp, a)
FILE	*fp;
F_arc	*a;
{
	F_arrow	*f, *b;

	fprintf(fp, "%d %d %d %d %d %d %d %d %.3f %d %d %d %.3f %.3f %d %d %d %d %d %d\n",
		O_ARC, a->type, a->style, a->thickness, 
		a->color, a->depth, a->pen, a->area_fill,
		a->style_val, a->direction,
		((f = a->for_arrow) ? 1 : 0), ((b = a->back_arrow) ? 1 : 0),
		a->center.x, a->center.y, 
		a->point[0].x, a->point[0].y, 
		a->point[1].x, a->point[1].y, 
		a->point[2].x, a->point[2].y);
	if (f)
	    fprintf(fp, "\t%d %d %.3f %.3f %.3f\n", f->type, f->style,
			f->thickness, f->wid, f->ht);
	if (b)
	    fprintf(fp, "\t%d %d %.3f %.3f %.3f\n", b->type, b->style,
			b->thickness, b->wid, b->ht);
}

Fig_write_compound(fp, com)
FILE		*fp;
F_compound	*com;
{
	F_arc		*a;
	F_compound	*c;
	F_ellipse	*e;
	F_line		*l;
	F_spline	*s;
	F_text		*t;

	fprintf(fp, "%d %d %d %d %d\n", O_COMPOUND, com->nwcorner.x,
		com->nwcorner.y, com->secorner.x, com->secorner.y);
	for (a = com->arcs; a != NULL; a = a-> next)Fig_write_arc(fp, a);
	for (c = com->compounds; c != NULL; c = c-> next)Fig_write_compound(fp, c);
	for (e = com->ellipses; e != NULL; e = e-> next)Fig_write_ellipse(fp, e);
	for (l = com->lines; l != NULL; l = l-> next)Fig_write_line(fp, l);
	for (s = com->splines; s != NULL; s = s-> next)Fig_write_spline(fp, s);
	for (t = com->texts; t != NULL; t = t-> next)Fig_write_text(fp, t);
	fprintf(fp, "%d\n", O_END_COMPOUND);
}

Fig_write_ellipse(fp, e)
FILE		*fp;
F_ellipse	*e;
{
	if( e->radiuses.x == 0 || e->radiuses.y == 0 )
		return;
	
	fprintf(fp, "%d %d %d %d %d %d %d %d %.3f %d %.3f %d %d %d %d %d %d %d %d\n",
		O_ELLIPSE, e->type, e->style, e->thickness, 
		e->color, e->depth, e->pen, e->area_fill,
		e->style_val, e->direction, e->angle,
		e->center.x, e->center.y, 
		e->radiuses.x, e->radiuses.y, 
		e->start.x, e->start.y, 
		e->end.x, e->end.y);
	}

Fig_write_line(fp, l)
FILE	*fp;
F_line	*l;
{
	F_point	*p;
	F_arrow	*f, *b;

	if( l->points == NULL )
		return;
#ifndef TFX
	if (l->type == T_ARC_BOX)
	    fprintf(fp, "%d %d %d %d %d %d %d %d %.3f %d %d %d\n",
		O_POLYLINE, l->type, l->style, l->thickness,
		l->color, l->depth, l->pen, l->area_fill, l->style_val, l->radius,
		((f = l->for_arrow) ? 1 : 0), ((b = l->back_arrow) ? 1 : 0));
	else
#endif TFX
	    fprintf(fp, "%d %d %d %d %d %d %d %d %.3f %d %d\n",
		O_POLYLINE, l->type, l->style, l->thickness,
		l->color, l->depth, l->pen, l->area_fill, l->style_val,
		((f = l->for_arrow) ? 1 : 0), ((b = l->back_arrow) ? 1 : 0));
	if (f)
	    fprintf(fp, "\t%d %d %.3f %.3f %.3f\n", f->type, f->style,
			f->thickness, f->wid, f->ht);
	if (b)
	    fprintf(fp, "\t%d %d %.3f %.3f %.3f\n", b->type, b->style,
			b->thickness, b->wid, b->ht);
	fprintf(fp, "\t");
	for (p = l->points; p!= NULL; p = p->next) {
	    fprintf(fp, " %d %d", p->x, p->y);
	    };
	fprintf(fp, " 9999 9999\n");
}

Fig_write_spline(fp, s)
FILE		*fp;
F_spline	*s;
{
	F_control	*cp;
	F_point		*p;
	F_arrow		*f, *b;

	if( s->points == NULL )
		return;
	fprintf(fp, "%d %d %d %d %d %d %d %d %.3f %d %d\n",
		O_SPLINE, s->type, s->style, s->thickness,
		s->color, s->depth, s->pen, s->area_fill, s->style_val,
		((f = s->for_arrow) ? 1 : 0), ((b = s->back_arrow) ? 1 : 0));
	if (f)
	    fprintf(fp, "\t%d %d %.3f %.3f %.3f\n", f->type, f->style,
			f->thickness, f->wid, f->ht);
	if (b)
	    fprintf(fp, "\t%d %d %.3f %.3f %.3f\n", b->type, b->style,
			b->thickness, b->wid, b->ht);
	fprintf(fp, "\t");
	for (p = s->points; p != NULL; p = p->next) {
	    fprintf(fp, " %d %d", p->x, p->y);
	    };
	fprintf(fp, " 9999 9999\n");  /* terminating code  */

	if (s->controls == NULL) return;
	fprintf(fp, "\t");
	for (cp = s->controls; cp != NULL; cp = cp->next) {
	    fprintf(fp, " %.3f %.3f %.3f %.3f",
			cp->lx, cp->ly, cp->rx, cp->ry);
	    };
	fprintf(fp, "\n");
	}

Fig_write_text(fp, t)
FILE	*fp;
F_text	*t;
{
	if( t->length == 0 )
		return;
	fprintf(fp, "%d %d %d %d %d %d %d %.3f %d %d %d %d %d %s\1\n", 
		O_TEXT, t->type, t->font, t->size, t->pen,
		t->color, t->depth, t->angle,
		t->style, t->height, t->length, 
		t->base_x, t->base_y, t->cstring);
	}
