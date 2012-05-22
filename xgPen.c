#ifndef XG_DYMOD_SUPPORT
#	define XG_DYMOD_SUPPORT
#endif

#include "config.h"
IDENTIFY( "XGPens lowlevel routines and user-interface" );

#include <stdio.h>
#include <stdlib.h>

extern FILE *StdErr;

#include "copyright.h"

  /* Include a whole bunch of headerfiles. Not all of them are strictly necessary, but if
   \ we want to have fdecl.h to know all functions we possibly might want to call, this
   \ list is needed.
   */
#include "xgout.h"
#include "xgraph.h"
#include "new_ps.h"
#include "xtb/xtb.h"

#include "NaN.h"

#include "fdecl.h"

  /* get the ascanf definitions:	*/
#include "ascanf.h"

  /* Definitions, externals, etc:	*/

#include "XXseg.h"
extern XSegment *Xsegs, *lowYsegs, *highYsegs;		/* Point space for X */
extern XXSegment *XXsegs;
#define LYsegs	lowYsegs
#define HYsegs	highYsegs
extern XSegment_error *XsegsE;
extern long XsegsSize, XXsegsSize, YsegsSize, XsegsESize;

extern int use_ps_LStyle, ps_LStyle;

#define __DLINE__	(double)__LINE__

extern int maxitems;

extern char *xgraph_NameBuf;
extern int xgraph_NameBufLen;

extern LocalWin *ActiveWin, StubWindow;

extern int XGStoreColours;

#define loX	win_geo.bounds._loX
#define loY	win_geo.bounds._loY
#define lopX	win_geo.bounds._lopX
#define hinY	win_geo.bounds._hinY
#define lopY	win_geo.bounds._lopY
#define hiX	win_geo.bounds._hiX
#define hiY	win_geo.bounds._hiY
#define XOrgX	win_geo._XOrgX
#define XOrgY	win_geo._XOrgY
#define XOppX	win_geo._XOppX
#define XOppY	win_geo._XOppY
#define UsrOrgX	win_geo._UsrOrgX
#define UsrOrgY	win_geo._UsrOrgY
#define UsrOppX	win_geo._UsrOppX
#define UsrOppY	win_geo._UsrOppY
#define WinOrgX	win_geo._WinOrgX
#define WinOrgY	win_geo._WinOrgY
#define WinOppX	win_geo._WinOppX
#define WinOppY	win_geo._WinOppY
#define R_UsrOrgX	win_geo.R_UsrOrgX
#define R_UsrOrgY	win_geo.R_UsrOrgY
#define R_UsrOppX	win_geo.R_UsrOppX
#define R_UsrOppY	win_geo.R_UsrOppY
#define XUnitsPerPixel	win_geo._XUnitsPerPixel
#define YUnitsPerPixel	win_geo._YUnitsPerPixel
#define R_XUnitsPerPixel	win_geo.R_XUnitsPerPixel
#define R_YUnitsPerPixel	win_geo.R_YUnitsPerPixel

#ifndef ABS
#	define ABS(x)		(((x)<0)?-(x):(x))
#endif
#ifndef MIN
#	define MIN(a, b)	((a) < (b) ? (a) : (b))
#endif
#ifndef MAX
#	define MAX(a, b)	((a) > (b) ? (a) : (b))
#endif


extern char *ParsedColourName;
#define StoreCName(name)	xfree(name);name=XGstrdup(ParsedColourName)

inline double pLINEWIDTH(LocalWin *wi, double lineWidth)
{ double r;
	if( !wi ){
		r= (lineWidth>=0)? lineWidth : -lineWidth;
	}
	else{
		if( lineWidth>= 0 ){
			r= lineWidth;
		}
		else{
			  // RJVB 20081210: sometimes we'd like to specify a "real-world" width ...
			  // be aware that XDrawSegments (used by seg_X()) gives those a horrible look!
			if( wi->aspect ){
				r= SCREENXDIM(wi, -lineWidth);
			}
			else{
				r= (SCREENXDIM(wi, -lineWidth) + SCREENYDIM(wi, -lineWidth)) / 2.0;
			}
		}
	}
	return(r);
}

/* =============== low- and midlevel drawing routines, extracted from xgraph.c =============== */

void PenTextDimensions( LocalWin *wi, XGPenPosition *pos );

void CollectPenPosStats( LocalWin *wi, XGPenPosition *pos )
{
	switch( pos->command ){
		default:
			CollectPointStats( wi, NULL, 0, pos->x, pos->y, 0,0,0,0 );
			break;
		case XGPenTextBox:
		case XGPenText:
			if( !pos->attr.text_outside ){
				switch( pos->textJust ){
					case T_CENTER:
						CollectPointStats( wi, NULL, 0, pos->x- pos->w/2, pos->y- pos->h/2, 0,0,0,0 );
						CollectPointStats( wi, NULL, 0, pos->x+ pos->w/2, pos->y+ pos->h/2, 0,0,0,0 );
						break;
					case T_LEFT:
						CollectPointStats( wi, NULL, 0, pos->x, pos->y- pos->h/2, 0,0,0,0 );
						CollectPointStats( wi, NULL, 0, pos->x+ pos->w, pos->y+ pos->h/2, 0,0,0,0 );
						break;
					case T_VERTICAL:
						CollectPointStats( wi, NULL, 0, pos->x, pos->y, 0,0,0,0 );
						CollectPointStats( wi, NULL, 0, pos->x+ pos->w, pos->y+ pos->h, 0,0,0,0 );
						break;
					case T_UPPERLEFT:
						CollectPointStats( wi, NULL, 0, pos->x, pos->y, 0,0,0,0 );
						CollectPointStats( wi, NULL, 0, pos->x+ pos->w, pos->y- pos->h, 0,0,0,0 );
						break;
					case T_TOP:
						CollectPointStats( wi, NULL, 0, pos->x- pos->w/2, pos->y, 0,0,0,0 );
						CollectPointStats( wi, NULL, 0, pos->x+ pos->w/2, pos->y- pos->h, 0,0,0,0 );
						break;
					case T_UPPERRIGHT:
						CollectPointStats( wi, NULL, 0, pos->x- pos->w, pos->y, 0,0,0,0 );
						CollectPointStats( wi, NULL, 0, pos->x, pos->y- pos->h, 0,0,0,0 );
						break;
					case T_RIGHT:
						CollectPointStats( wi, NULL, 0, pos->x, pos->y- pos->h/2, 0,0,0,0 );
						CollectPointStats( wi, NULL, 0, pos->x- pos->w, pos->y+ pos->h/2, 0,0,0,0 );
						break;
					case T_LOWERRIGHT:
						CollectPointStats( wi, NULL, 0, pos->x, pos->y, 0,0,0,0 );
						CollectPointStats( wi, NULL, 0, pos->x- pos->w, pos->y+ pos->h, 0,0,0,0 );
						break;
					case T_BOTTOM:
						CollectPointStats( wi, NULL, 0, pos->x+ pos->w/2, pos->y, 0,0,0,0 );
						CollectPointStats( wi, NULL, 0, pos->x- pos->w/2, pos->y+ pos->h, 0,0,0,0 );
						break;
					case T_LOWERLEFT:
						CollectPointStats( wi, NULL, 0, pos->x, pos->y, 0,0,0,0 );
						CollectPointStats( wi, NULL, 0, pos->x+ pos->w, pos->y+ pos->h, 0,0,0,0 );
						break;
				}
			}
			break;
		case XGPenDrawRect:
		case XGPenFillRect:
			CollectPointStats( wi, NULL, 0, pos->x, pos->y, 0,0,0,0 );
			CollectPointStats( wi, NULL, 0, pos->x+ pos->w, pos->y+ pos->h, 0,0,0,0 );
			break;
		case XGPenDrawCRect:
		case XGPenFillCRect:
			CollectPointStats( wi, NULL, 0, pos->x- pos->w/2, pos->y- pos->h/2, 0,0,0,0 );
			CollectPointStats( wi, NULL, 0, pos->x+ pos->w/2, pos->y+ pos->h/2, 0,0,0,0 );
			break;
		case XGPenDrawEllps:
			CollectPointStats( wi, NULL, 0, pos->x+ pos->w, pos->y+ pos->h, 0,0,0,0 );
			CollectPointStats( wi, NULL, 0, pos->x- pos->w, pos->y- pos->h, 0,0,0,0 );
			break;
		case XGPenDrawPoly:
		case XGPenFillPoly:{
		  int i;
			CollectPointStats( wi, NULL, 0, pos->x, pos->y, 0,0,0,0 );
			for( i= 0; i< pos->polyN_1; i++ ){
				CollectPointStats( wi, NULL, 0, pos->polyX[i], pos->polyY[i], 0,0,0,0 );
			}
			break;
		}
	}
}

/* 20010724: The routine that determines that adaptation of the data-plotting bounding box necessary
 \ to accommodate pentext marked as text_outside in the image plane. Initial implementation that
 \ undoubtedly needs modifications. For simple text, it seems to work. I haven't yet tested it with
 \ vertical text!
 */
int Outside_PenText( LocalWin *wi, int *redo_return )
{ int n_outside= 0, redo= False;
	if( wi->pen_list && !wi->no_pens ){
	  XGPen *Pen= wi->pen_list;
	  int pnt_nr;
	  char msg[128];
		while( Pen ){
			if( !Pen->floating && (Pen->drawn || !PENSKIP(wi,Pen)) ){
			  XGPenPosition *pos= Pen->position;
			  int ok;
				sprintf( msg, "Pen #%d, %d points", Pen->pen_nr, Pen->current_pos );
				for( pnt_nr= 0; pnt_nr< Pen->positions && pnt_nr< Pen->current_pos; pnt_nr++, pos++ ){
					if( pos->attr.text_outside && (pos->command== XGPenTextBox || pos->command== XGPenText) ){
					  int ulx, uly, frx, fry;
						pos->tx= pos->x;
						pos->ty= pos->y;
						ok= 1; do_transform( wi,
							msg, __DLINE__, "Outside_PenText()", &ok, NULL, &pos->tx, NULL, NULL, &pos->ty, NULL, NULL,
							NULL, NULL, 1, pnt_nr, 1.0, 1.0, 1.0, -1, 0, 0
						);
						if( ok ){
						  int twidth, theight;
						  extern int Use_HO_Previous_TC;
							ulx= SCREENX( wi, pos->tx);
							uly= SCREENY( wi, pos->ty);
							PenTextDimensions( wi, pos );
							twidth= pos->TextBox->w;
							theight= pos->TextBox->h;
							if( pos->command== XGPenTextBox ){
							  int lw= pos->TextBox->attr.lineWidth* wi->dev_info.var_width_factor;
								twidth+= lw;
								theight+= lw;
							}
							  /* 20010724: We probably need to obtain the true (ulx,uly) co-ordinate pair
							   \ where the drawing will start here.
							   */
							frx= ulx+ twidth;
							fry= uly+ theight;

							if( CShrinkArea( wi, pos->tx, pos->ty, ulx, uly, frx, fry, Use_HO_Previous_TC ) ){
								redo+= 1;
							}
						}

						n_outside+= 1;
					}
				}
				Pen->drawn= 0;
			}
			Pen= Pen->next;
		}
	}
	if( n_outside && redo ){
		*redo_return= redo;
	}
	return( n_outside );
}

/* Draw pending Pen line segments, with the specified attributes.	*/
void DrawPenSegments( LocalWin *wi, XGPen *Pen, int nondrawn_s, int linestyle, double lineWidth, int pixvalue, Pixel pixelValue )
{ double lw= pLINEWIDTH(wi,lineWidth);
	if( Pen->highlight ){
	  Pixel hlp= highlightPixel;
		if( Pen->hlpixelCName ){
			highlightPixel= Pen->hlpixelValue;
		}
		HighlightSegment( wi, -1, nondrawn_s, Xsegs, HL_WIDTH(lw), L_VAR);
		highlightPixel= hlp;
	}
	wi->dev_info.xg_seg(wi->dev_info.user_state,
			 nondrawn_s, Xsegs,
			 lw, L_VAR,
			 linestyle, pixvalue, pixelValue, NULL
	);
	Pen->drawn += 1;
}

/* Flush pending Pen line segments by drawing them, and then update the attributes with the
 \ values for position *pos, and reset the XSegment pointer structure. If CheckFirst is true,
 \ verify first whether the flush is necessary.
 */
int FlushPenSegments( LocalWin *wi, XGPen *Pen, int *nondrawn_s,
	int *linestyle, double *lineWidth, int *pixvalue, Pixel *pixelValue,
	XGPenPosition *pos, XSegment **xseg, int CheckFirst )
{ int ret= False;
  static int pen_nr= -1;
  static char *ppCName= NULL;
	if( CheckFirst ){
		if( pen_nr== Pen->pen_nr && !(*linestyle!= pos->attr.linestyle || *pixvalue!= pos->colour.pixvalue ||
				*pixelValue!= pos->colour.pixelValue || *lineWidth!= pos->attr.lineWidth)
		){
			return(0);
		}
	}
	if( *nondrawn_s ){
		DrawPenSegments( wi, Pen, *nondrawn_s, *linestyle, *lineWidth, *pixvalue, *pixelValue );
	}
/* 	Xsegs->x1= (*xseg)->x2;	*/
/* 	Xsegs->y1= (*xseg)->y2;	*/
	if( *xseg> Xsegs ){
		Xsegs->x1= (*xseg)[-1].x2;
		Xsegs->y1= (*xseg)[-1].y2;
		*xseg= Xsegs;
		ret= True;
	}
	if( debugFlag && debugLevel> 0 ){
		fprintf( StdErr, "\tPen #%d:", Pen->pen_nr );
		if( *nondrawn_s ){
			fprintf( StdErr, " %d segment(s) drawn with s=%d,w=%s,c=%s",
				*nondrawn_s, *linestyle, d2str( *lineWidth, 0,0),
				(ppCName)? ppCName : d2str(*pixelValue, 0,0)
			);
		}
	}
	*nondrawn_s= 0;
	*linestyle= pos->attr.linestyle;
	*pixvalue= pos->colour.pixvalue;
	*pixelValue= pos->colour.pixelValue;
	ppCName= (*pixvalue<0)? pos->pixelCName : AllAttrs[*pixvalue].pixelCName;
	*lineWidth= pos->attr.lineWidth;
	if( debugFlag && debugLevel> 0 ){
		fprintf( StdErr, "; new attr s=%d,w=%s,c=%s\n",
			*linestyle, d2str( *lineWidth, 0,0),
			(ppCName)? ppCName : d2str( (double) *pixelValue, 0,0)
		);
	}
	return(ret);
}

char *PenCommandName[]= { "void", "PenMoveTo", "PenLineTo",
	"PenDrawRect", "PenFillRect", "PenDrawCRect", "PenFillCRect",
	"PenDrawEllps", "PenFillEllps(N/A!)", "PenDrawPoly", "PenFillPoly", "PenText", "PenTextBox", "void!", "void!!", "void!!!"
};

void xg_DrawPenRectangle( LocalWin *wi, XGPen *Pen, XGPenPosition *pos, XRectangle *rec,
	int linestyle, double lineWidth, int pixvalue, Pixel pixelValue )
{ int pv; Pixel pV;
  double lw;
	if( (pos->command== XGPenFillRect || pos->command== XGPenFillCRect) && pos->not_outlined ){
		pv= pos->flcolour.pixvalue;
		pV= pos->flcolour.pixelValue;
		lw= 0;
	}
	else{
		pv= pixvalue;
		pV= pixelValue;
		lw= pLINEWIDTH(wi,lineWidth);
	}
	if( Pen->highlight ){
	  int upls= use_ps_LStyle;
		use_ps_LStyle= 0;
		wi->dev_info.xg_rect( wi->dev_info.user_state,
			rec,
			HL_WIDTH(lw), L_VAR, linestyle,
			-1, (Pen->hlpixelCName)? Pen->hlpixelValue : highlightPixel,
			0, 0,0, NULL
		);
		use_ps_LStyle= upls;
	}
	wi->dev_info.xg_rect( wi->dev_info.user_state,
		rec,
		lw, L_VAR, linestyle, pv, pV,
		(pos->command== XGPenFillRect || pos->command== XGPenFillCRect)? 1 : 0,
		pos->flcolour.pixvalue, pos->flcolour.pixelValue,
		NULL
	);
}

/* 20031113: quick hack to allow for a form of clipping for PenRectangles. Only
 \ rectangles are drawn that fit fully within the drawing region (or in/on the window/paper
 \ in case of a floating pen).
 */
int NotClipped( LocalWin *wi, double x, double y, int floating )
{ double cminx, cmaxx, cminy, cmaxy;
	if( floating ){
	  double X= x, Y= y;
		cminx= wi->WinOrgX;
		cminy= wi->WinOrgY;
		cmaxx= wi->WinOppX;
		cmaxy= wi->WinOppY;
		  /* For some reason or another, we need to reconvert back to real world co-ordinates here
		   \ (as opposed to axes-transformed values), here but not in ClipWindow?
		   \ I'm not in the mood to figure out: this works.
		   */
		x= Reform_X(wi, X, Y);
		y= Reform_Y(wi, Y, X);
	}
	else{
		cminx= wi->UsrOrgX;
		cminy= wi->UsrOrgY;
		cmaxx= wi->UsrOppX;
		cmaxy= wi->UsrOppY;
	}
	return( (x< cminx || x> cmaxx || y< cminy || y> cmaxy)? 0 : 1 );
}

void _DrawPenRectangle( LocalWin *wi, XGPen *Pen, XGPenPosition *pos, int i,
	int linestyle, double lineWidth, int pixvalue, Pixel pixelValue,
	char *msg, char *fcname )
{ XRectangle rec;
  double ex, ey;
  XGPenCommand op= pos->command;
  int ok, isRect;
	  /* Determine the world co-ordinates of the rectangles corners. This of course depends
	   \ on the type of rectangle... Whether (x,y) is its centre in the middle of the width,height,
	   \ or whether (x,y) is one of the 2 diagonal base-points (lower left).
	   */
	if( pos->w && pos->h ){
		isRect= True;
	}
	else{
		isRect= False;
	}
	if( op== XGPenDrawRect || op== XGPenFillRect ){
	  /* Rectangle specified with basepoint (x,y)	*/
		  /* Transform the rectangle: base point:	*/
		pos->tx= pos->x;
		pos->ty= pos->y;
		ok= 1; do_transform( wi,
			msg, __DLINE__, "_DrawPenRectangle()", &ok, NULL, &pos->tx, NULL, NULL, &pos->ty, NULL, NULL,
			NULL, NULL, 1, i, 1.0, 1.0, 1.0, -1, 0, 0
		);
		if( ok && NotClipped(wi, pos->tx, pos->ty, Pen->floating) ){
			  /* Now transform the opposite point:	*/
			if( isRect ){
				ex= pos->x+ pos->w;
				ey= pos->y+ pos->h;
				ok= 1; do_transform( wi,
					msg, __DLINE__, "_DrawPenRectangle()", &ok, NULL, &ex, NULL, NULL, &ey, NULL, NULL,
					NULL, NULL, 1, i, 1.0, 1.0, 1.0, -1, 0, 0
				);
				if( !ok || !NotClipped(wi, ex, ey, Pen->floating) ){
					return;
				}
			}
			else{
				ex= pos->tx, ey= pos->ty;
			}
		}
		else{
			return;
		}
		if( debugFlag && debugLevel> 0 ){
			fprintf( StdErr, "_DrawPenRectangle(%d,%d): %s (%s,%s)=(%s,%s) + (%s,%s)=>(%s,%s) %d fc=%s\n", Pen->pen_nr, i,
				PenCommandName[op], d2str( pos->x, NULL, NULL), d2str( pos->y, NULL, NULL),
				d2str( pos->tx, NULL, NULL), d2str( pos->ty, NULL, NULL),
				d2str( pos->w, NULL, NULL), d2str( pos->h, NULL, NULL),
				d2str( ex, NULL, NULL), d2str( ey, NULL, NULL),
				ok, fcname
			);
		}
		pos->sx= SCREENX( wi, pos->tx );
		pos->sy= SCREENY( wi, pos->ty );
		pos->OK= True;
	}
	else{
	  /* rectangle centered on (x,y)	*/
	  double x, y;
		pos->tx= pos->x- pos->w/2;
		pos->ty= pos->y- pos->h/2;
		ok= 1; do_transform( wi,
			msg, __DLINE__, "_DrawPenRectangle()", &ok, NULL, &pos->tx, NULL, NULL, &pos->ty, NULL, NULL,
			NULL, NULL, 1, i, 1.0, 1.0, 1.0, -1, 0, 0
		);
		if( ok && NotClipped( wi, pos->tx, pos->ty, Pen->floating) ){
			  /* Now transform the opposite point:	*/
			if( isRect ){
				ex= pos->x+ pos->w/2;
				ey= pos->y+ pos->h/2;
				ok= 1; do_transform( wi,
					msg, __DLINE__, "_DrawPenRectangle()", &ok, NULL, &ex, NULL, NULL, &ey, NULL, NULL,
					NULL, NULL, 1, i, 1.0, 1.0, 1.0, -1, 0, 0
				);
				if( !ok || !NotClipped( wi, ex, ey, Pen->floating) ){
					return;
				}
			}
			else{
				ex= pos->tx, ey= pos->ty;
			}
		}
		else{
			return;
		}
		if( isRect ){
			x= pos->x;
			y= pos->y;
			ok= 1; do_transform( wi,
				msg, __DLINE__, "_DrawPenRectangle()", &ok, NULL, &x, NULL, NULL, &y, NULL, NULL,
				NULL, NULL, 1, i, 1.0, 1.0, 1.0, -1, 0, 0
			);
		}
		else{
			x= pos->tx, y= pos->ty;
		}
		if( ok ){
			pos->sx= SCREENX( wi, x );
			pos->sy= SCREENY( wi, y );
			pos->OK= True;
		}
		else{
			  /* No need to skip this rect if for whatever reason only the to-be-marked
			   \ centre is invalid..
			   */
			pos->OK= False;
		}
		if( debugFlag && debugLevel> 0 ){
			fprintf( StdErr, "_DrawPenRectangle(%d,%d): %s (%s,%s)=(%s,%s) +/- (%s,%s)=>(%s,%s) %d fc=%s\n", Pen->pen_nr, i,
				PenCommandName[op], d2str( pos->x, NULL, NULL), d2str( pos->y, NULL, NULL),
				d2str( x, NULL, NULL), d2str( y, NULL, NULL),
				d2str( pos->w/2, NULL, NULL), d2str( pos->h/2, NULL, NULL),
				d2str( ex, NULL, NULL), d2str( ey, NULL, NULL),
				ok, fcname
			);
		}
	}
	  /* Construct a rectangle (x,y,w,h) from the corresponding screen co-ordinates:	*/
	if( isRect ){
		rec= *rect_diag2xywh( SCREENX( wi, pos->tx), SCREENY( wi, pos->ty), SCREENX( wi, ex), SCREENY( wi, ey) );
	}
	else{
		rec.x= SCREENX(wi, pos->tx);
		rec.y= SCREENY(wi, pos->ty);
		rec.width= rec.height= 0;
	}
	xg_DrawPenRectangle( wi, Pen, pos, &rec, linestyle, lineWidth, pixvalue, pixelValue );
	Pen->drawn += 1;
}

void _DrawPenTextBoxRectangle( LocalWin *wi, XGPen *Pen, XGPenPosition *pos, int i,
	int sx, int sy, int lheight,
	int linestyle, double lineWidth, int pixvalue, Pixel pixelValue )
{ XGPenPosition *lpos= pos[i].TextBox;
  int xlw, ylw, x_offset, y_offset, tw, th;
  XRectangle rect;
	  /* xlw,ylw: correction for the lineWidth with which the box is outlined: */
	ylw= xlw= 2* pLINEWIDTH(wi,lpos->attr.lineWidth)* wi->dev_info.var_width_factor;
	rect.x= sx;
	rect.y= sy;
	  /* lpos->[wh] contains the width and height of the string in screen co-ordinates. */
	tw= lpos->w;
	th= lpos->h;
	  /* Size of the box is the string size plus the linewidth: */
	rect.width= tw+ xlw;
	rect.height= th+ ylw;
	  /* offset values that give some margins around the string: 2.5* bdr_pad */
	x_offset= wi->dev_info.bdr_pad/ 2.5;
	y_offset= wi->dev_info.bdr_pad/ 2.5;
	if( lpos->textJust== T_VERTICAL ){
		  /* Fine-tuning by hand... The vertical printing routine offsets the string by a certain amount,
		   \ that seems to be well estimated by 0.15* width (the used font's height!) horizontally, and
		   \ -1.5* width vertically. I hope these values are universal - and valid until I get a better
		   \ grasp on what the text rotation actually does!
		   */
		rect.x-= 0.5* xlw- 0.15* tw+ x_offset;
		/* alternative multilines adjustment:
		rect.y-= 0.5* ylw+ 1.5* tw+ y_offset;
		 */
		rect.y-= 0.2* ylw+ th+ y_offset;
		rect.width+= 2* x_offset;
		rect.height+= 2* y_offset;
	}
	else{
		switch( lpos->textJust2[0] ){
		  /* Great! No additional correction is necessary for the horizontal position:
		   \ just incorporate the margin, linewidth, and determine the left corner position
		   \ as a function of justification:
		   */
			case -1:
				rect.x-= tw+ 0.5* xlw+ x_offset;
				break;
			case 0:
				rect.x= rect.x- rect.width/2- x_offset;
				break;
			default:
			case 1:
				rect.x-= 0.5* xlw+ x_offset;
				break;
		}
		switch( lpos->textJust2[1] ){
			default:
			case -1:
				rect.y-= y_offset+ 1.15* lheight;
				break;
			case 0:
			  /* Some correction for the extra margin seems to be necessary?! */
/* 									rect.y+= rect.height/ 2+ 0.75* y_offset;	*/
				rect.y-= y_offset+ 0.75* lheight;
				break;
			case 1:
			  /* Vertically, the necessary shift seems to be well approximated by 0.1* th	*/
/* 									rect.y+= (1-0.10)* th+ 0.5* ylw+ y_offset;	*/
				rect.y-= y_offset+ 0.35* lheight;
				break;
		}
		rect.width+= 2* x_offset;
		rect.height+= 2* y_offset;
	}
	xg_DrawPenRectangle( wi, Pen, lpos, &rect, linestyle, lineWidth, pixvalue, pixelValue );
}

void DrawPen( LocalWin *wi, XGPen *Pen )
{ int i, mark_inside1, mark_inside2, clipcode1, clipcode2, nondrawn_s= 0, ok, lifted= False, liftpos, N= 0, flushed= 0;
  int linestyle, pixvalue;
  Pixel pixelValue;
  double lineWidth;
  XSegment *xseg= Xsegs;
  double ptx, pty;
  XGPenPosition *pos= Pen->position;
  char msg[128];
  static char *fcname= "?";
  XGPenCommand op;

	sprintf( msg, "Pen #%d, %d points", Pen->pen_nr, Pen->current_pos );

	  /* 20030328: this is *really* necessary!! */
	if( Pen->positions> maxitems ){
		maxitems= Pen->positions;
		realloc_Xsegments();
		xseg= Xsegs;
	}

#ifdef DEBUG
	memset( Xsegs, (short) MAX( wi->dev_info.area_h, wi->dev_info.area_w), XsegsSize );
#endif

	Pen->drawn = 0;
	for( i= 0; i< Pen->current_pos; i++ ){
		if( wi->fit_after_draw && !wi->fitting && !Pen->floating ){
			CollectPenPosStats( wi, &pos[i] );
		}
		pos[i].OK= False;
		if( debugFlag && debugLevel> 0 ){
			fcname= (pos[i].flcolour.pixvalue<0)? (pos[i].flpixelCName)? pos[i].flpixelCName : fcname :
				AllAttrs[pos[i].flcolour.pixvalue].pixelCName;
		}
		switch( (op= pos[i].command) ){
			case XGPenText:
			case XGPenTextBox:
			case XGPenMoveTo:{
xgpenMoveTo:;
				pos[i].tx= pos[i].x;
				pos[i].ty= pos[i].y;
				ok= 1; do_transform( wi,
					msg, __DLINE__, "DrawPen()", &ok, NULL, &pos[i].tx, NULL, NULL, &pos[i].ty, NULL, NULL,
					NULL, NULL, 1, i, 1.0, 1.0, 1.0, -1, 0, 0
				);
				if( debugFlag && debugLevel> 0 ){
					fprintf( StdErr, "DrawPen(%d,%d): %s: set pen (%s,%s)=(%s,%s|%d)\n", Pen->pen_nr, i,
						PenCommandName[op], d2str( pos[i].x, NULL, NULL), d2str( pos[i].y, NULL, NULL),
						d2str( pos[i].tx, NULL, NULL), d2str( pos[i].ty, NULL, NULL), ok
					);
				}
				if( ok && !(NaN(pos[i].tx) || NaN(pos[i].ty)) ){
					switch( op ){
						default:
							  /* The start co-ordinates can only be really stored when the end
							   \ co-ordinates are known - only then is clipping possible.
							   */
							pos[i].sx= xseg->x2= xseg->x1= SCREENX( wi, (ptx= pos[i].tx) );
							pos[i].sy= xseg->y2= xseg->y1= SCREENY( wi, (pty= pos[i].ty) );
							lifted= True;
							liftpos= i;
							break;
						case XGPenTextBox:
						case XGPenText:
							  /* For text commands, only set the co-ordinates. And check if
							   \ segments need to be flushed (generic command; below).
							   */
							pos[i].sx= SCREENX( wi, (ptx= pos[i].tx) );
							pos[i].sy= SCREENY( wi, (pty= pos[i].ty) );
							break;
					}
					pos[i].OK= True;
					if( FlushPenSegments( wi, Pen, &nondrawn_s, &linestyle, &lineWidth, &pixvalue, &pixelValue,
						&pos[i], &xseg, True )
					){
						flushed= True;
					}
				}
				break;
			}
			case XGPenLineTo:{
				if( nondrawn_s || lifted || flushed ){
					if( nondrawn_s== XsegsSize/sizeof(XSegment) ||
						linestyle!= pos[i].attr.linestyle || pixvalue!= pos[i].colour.pixvalue ||
						pixelValue!= pos[i].colour.pixelValue || lineWidth!= pos[i].attr.lineWidth
					){
						if( FlushPenSegments( wi, Pen, &nondrawn_s, &linestyle, &lineWidth, &pixvalue, &pixelValue,
							&pos[i], &xseg, False )
						){
							flushed= True;
						}
					}
					pos[i].tx= pos[i].x;
					pos[i].ty= pos[i].y;
					ok= 1; do_transform( wi,
						msg, __DLINE__, "DrawPen()", &ok, NULL, &pos[i].tx, NULL, NULL, &pos[i].ty, NULL, NULL,
						NULL, NULL, 1, i, 1.0, 1.0, 1.0, -1, 0, 0
					);
					if( debugFlag && debugLevel> 0 ){
						fprintf( StdErr, "DrawPen(%d,%d): %s (%s,%s)=(%s,%s|%d)", Pen->pen_nr, i,
							PenCommandName[op], d2str( pos[i].x, NULL, NULL), d2str( pos[i].y, NULL, NULL),
							d2str( pos[i].tx, NULL, NULL), d2str( pos[i].ty, NULL, NULL), ok
						);
					}
					if( ok && !(NaN(pos[i].tx) || NaN(pos[i].ty)) ){
						  /* Clip the segment. The previous co-ordinate pair is stored in (tx,ty)	*/
						if( ClipWindow( wi, NULL, Pen->floating, &ptx, &pty, &pos[i].tx, &pos[i].ty,
								&mark_inside1, &mark_inside2, &clipcode1, &clipcode2 )
						){
							  /* 20050508: temporary solution to handle clipping of pen lines a bit better: */
							if( mark_inside2 || pos[i].attr.noclip ){
								if( lifted ){
									xseg->x1= SCREENX( wi, ptx );
									xseg->y1= SCREENY( wi, pty );
									lifted= False;
									pos[liftpos].OK= True;
								}
								else if( xseg> Xsegs ){
								  /* Link to previous point/position:	*/
									xseg->x1= xseg[-1].x2;
									xseg->y1= xseg[-1].y2;
								}
								pos[i].sx= xseg->x2= SCREENX( wi, pos[i].tx );
								pos[i].sy= xseg->y2= SCREENY( wi, pos[i].ty );
								pos[i].OK= True;
								nondrawn_s+= 1;
								N+= 1;
								xseg++;
								flushed= False;
								if( debugFlag && debugLevel> 0 ){
									fprintf( StdErr, ", clipOK, %d cached segments, %d max of %d\n",
										nondrawn_s, N, maxitems
									);
								}
							}
							else{
								  /* 20050508: an out-the-window pen position can best be handled as if it were
								   \ a PenMoveTo statement instead of a LineTo. At least this seems to
								   \ work for now.
								   */
								if( debugFlag && debugLevel> 0 ){
									fprintf( StdErr, ", clipOK but current point (2) marked outside, %d cached segments, %d max of %d: handling as PenMoveTo.\n",
										nondrawn_s, N, maxitems
									);
								}
								goto xgpenMoveTo;
							}
						}
						else{
							if( debugFlag && debugLevel> 0 ){
								fputc( '\n', StdErr );
							}
							  /* 20050508: an out-the-window pen position can best be handled as if it were
							   \ a PenMoveTo statement instead of a LineTo. At least this seems to
							   \ work for now.
							   */
							goto xgpenMoveTo;
						}
					}
					else{
						if( debugFlag && debugLevel> 0 ){
							fputc( '\n', StdErr );
						}
						  /* 20050508: an out-the-window pen position can best be handled as if it were
						   \ a PenMoveTo statement instead of a LineTo. At least this seems to
						   \ work for now.
						   */
						goto xgpenMoveTo;
					}
				}
				else{
					goto xgpenMoveTo;
				}
				break;
			}
			case XGPenDrawRect:
			case XGPenFillRect:
			case XGPenDrawCRect:
			case XGPenFillCRect:{
				if( FlushPenSegments( wi, Pen, &nondrawn_s, &linestyle, &lineWidth, &pixvalue, &pixelValue,
					&pos[i], &xseg, True )
				){
					flushed= True;
				}
				_DrawPenRectangle( wi, Pen, &pos[i], i, linestyle, lineWidth, pixvalue, pixelValue, msg, fcname );
				break;
			}
			case XGPenDrawEllps:{
			  int rad_x, rad_y;
			  double x, y, ex, ey;
				if( FlushPenSegments( wi, Pen, &nondrawn_s, &linestyle, &lineWidth, &pixvalue, &pixelValue,
					&pos[i], &xseg, True )
				){
					flushed= True;
				}
				  /* The centre:	*/
				pos[i].tx= pos[i].x;
				pos[i].ty= pos[i].y;
				ok= 1; do_transform( wi,
					msg, __DLINE__, "DrawPen()", &ok, NULL, &pos[i].tx, NULL, NULL, &pos[i].ty, NULL, NULL,
					NULL, NULL, 1, i, 1.0, 1.0, 1.0, -1, 0, 0
				);
				if( ok ){
					  /* The true, requested centre will be used to put a marker, if one is requested.	*/
					pos[i].sx= SCREENX( wi, pos[i].tx );
					pos[i].sy= SCREENY( wi, pos[i].ty );
				}
				else{
					break;
				}
				  /* Now determine the position of the bounding rectangle. If the axes are untransformed,
				   \ these will be at x+-w and y+-h
				   */
				x= pos[i].x- pos[i].w;
				y= pos[i].y- pos[i].h;
				ok= 1; do_transform( wi,
					msg, __DLINE__, "DrawPen()", &ok, NULL, &x, NULL, NULL, &y, NULL, NULL,
					NULL, NULL, 1, i, 1.0, 1.0, 1.0, -1, 0, 0
				);
				if( ok ){
					  /* Now transform the opposite point:	*/
					ex= pos[i].x+ pos[i].w;
					ey= pos[i].y+ pos[i].h;
					ok= 1; do_transform( wi,
						msg, __DLINE__, "DrawPen()", &ok, NULL, &ex, NULL, NULL, &ey, NULL, NULL,
						NULL, NULL, 1, i, 1.0, 1.0, 1.0, -1, 0, 0
					);
					if( !ok ){
						break;
					}
				}
				else{
					break;
				}
				if( debugFlag && debugLevel> 0 ){
					fprintf( StdErr, "DrawPen(%d,%d): %s (%s,%s)+/-(%s,%s) => (%s,%s)+/-(%s,%s) %d\n", Pen->pen_nr, i,
						PenCommandName[op], d2str( pos[i].x, NULL, NULL), d2str( pos[i].y, NULL, NULL),
						d2str( pos[i].w, NULL, NULL), d2str( pos[i].h, NULL, NULL),
						d2str( (x+ex)/2, 0,0 ), d2str( (y+ey)/2, 0,0),
						d2str( fabs(ex-x)/2, 0,0), d2str( fabs(ey-y)/2, 0,0),
						ok
					);
				}
				  /* the radii can be changed due to e.g. log transformations:	*/
				rad_x= SCREENXDIM( wi, fabs(ex- x)/2 );
				rad_y= SCREENYDIM( wi, fabs(ey- y)/2 );
				  /* Likewise, the "centre" will no longer be in the middle, in that case. Since
				   \ we have no routine to draw such an "egg", we approximate it by a shifted ellipse.
				   \ Note that x and y are doubles.
				   */
				x= SCREENX( wi, (x+ ex)/ 2 );
				y= SCREENY( wi, (y+ ey)/ 2 );

				{ double lw= pLINEWIDTH(wi,lineWidth);
					if( Pen->highlight ){
					  int upls= use_ps_LStyle;
						use_ps_LStyle= 0;
						wi->dev_info.xg_arc( wi->dev_info.user_state,
							(int) x, (int) y, rad_x, rad_y,
							0, wi->radix, HL_WIDTH(lw),
							L_VAR, 0, -1, (Pen->hlpixelCName)? Pen->hlpixelValue : highlightPixel
						);
						use_ps_LStyle= upls;
					}
					wi->dev_info.xg_arc( wi->dev_info.user_state,
						(int) x, (int) y, rad_x, rad_y,
						0, wi->radix,
						lw, L_VAR, linestyle, pixvalue, pixelValue
					);
				}
				Pen->drawn += 1;
				break;
			}
			case XGPenDrawPoly:
			case XGPenFillPoly:{
			  int pN= pos[i].polyN_1+ 1;
			  ALLOCA( pts, XPoint, pN, pts_len);
			  double x, y;
			  int p;
				if( FlushPenSegments( wi, Pen, &nondrawn_s, &linestyle, &lineWidth, &pixvalue, &pixelValue,
					&pos[i], &xseg, True )
				){
					flushed= True;
				}
				pos[i].OK= False;
				pos[i].tx= pos[i].x;
				pos[i].ty= pos[i].y;
				ok= 1; do_transform( wi,
					msg, __DLINE__, "DrawPen()", &ok, NULL, &pos[i].tx, NULL, NULL, &pos[i].ty, NULL, NULL,
					NULL, NULL, 1, i, 1.0, 1.0, 1.0, -1, 0, 0
				);
				if( debugFlag && debugLevel> 0 ){
					fprintf( StdErr, "DrawPen(%d,%d): %s %d (%s,%s)=(%s,%s|%d)", Pen->pen_nr, i,
						PenCommandName[op], 0, d2str( pos[i].x, NULL, NULL), d2str( pos[i].y, NULL, NULL),
						d2str( pos[i].tx, NULL, NULL), d2str( pos[i].ty, NULL, NULL), ok
					);
					fflush( StdErr );
				}
				if( ok ){
					pos[i].sx= pts[0].x= SCREENX( wi, pos[i].tx );
					pos[i].sy= pts[0].y= SCREENY( wi, pos[i].ty );
					for( p= 0; p< pos[i].polyN_1 && ok; p++ ){
						x= pos[i].polyX[p];
						y= pos[i].polyY[p];
						ok= 1; do_transform( wi,
							msg, __DLINE__, "DrawPen()", &ok, NULL, &x, NULL, NULL, &y, NULL, NULL,
							NULL, NULL, 1, i, 1.0, 1.0, 1.0, -1, 0, 0
						);
						if( ok ){
							pts[p+1].x= SCREENX( wi, x );
							pts[p+1].y= SCREENY( wi, y );
							if( debugFlag && debugLevel> 0 ){
								fprintf( StdErr, "/ %d (%s,%s)=(%s,%s|%d)",
									p+1, d2str( pos[i].polyX[p], NULL, NULL), d2str( pos[i].polyY[p], NULL, NULL),
									d2str( x, NULL, NULL), d2str( y, NULL, NULL), ok
								);
								fflush( StdErr );
							}
						}
					}
					if( debugFlag && debugLevel> 0 ){
						fprintf( StdErr, " fc=%s\n", fcname);
					}
				}
				if( !ok ){
					break;
				}
				{ int pv; Pixel pV;
				  double lw;
					if( (op== XGPenFillPoly) && pos[i].not_outlined ){
						pv= pos[i].flcolour.pixvalue;
						pV= pos[i].flcolour.pixelValue;
						lw= 0;
					}
					else{
						pv= pixvalue;
						pV= pixelValue;
						lw= pLINEWIDTH(wi,lineWidth);
					}
					if( Pen->highlight ){
					  int upls= use_ps_LStyle;
						use_ps_LStyle= 0;
						wi->dev_info.xg_polyg( wi->dev_info.user_state,
							pts, pN,
							HL_WIDTH(lw), L_VAR, linestyle, -1, (Pen->hlpixelCName)? Pen->hlpixelValue : highlightPixel,
							0, 0,0, NULL
						);
						use_ps_LStyle= upls;
					}
					wi->dev_info.xg_polyg( wi->dev_info.user_state,
						pts, pN,
						lw, L_VAR, linestyle, pv, pV,
						(op== XGPenFillPoly)? 1 : 0,
						pos[i].flcolour.pixvalue, pos[i].flcolour.pixelValue,
						NULL
					);
				}
				Pen->drawn += 1;
				break;
			}
			default:
				break;
		}
	}
	if( nondrawn_s ){
		DrawPenSegments( wi, Pen, nondrawn_s, linestyle, lineWidth, pixvalue, pixelValue );
	}
	if( N> maxitems ){
		if( debugFlag && debugLevel> 0 ){
			fprintf( StdErr, "DrawPen(%d): expanding segment cache from %d to %d\n",
				Pen->pen_nr, maxitems, N
			);
		}
		maxitems= N;
		realloc_Xsegments();
	}
	  /* Dots are always overwrite type.	*/
	if( debugFlag && debugLevel> 0 ){
		fprintf( StdErr, "DrawPen(%d): markers/text:", Pen->pen_nr );
		fflush( StdErr );
	}
	for( i= 0; i< Pen->current_pos; i++ ){
		if( pos[i].OK ){
			if( debugFlag && debugLevel> 0 ){
				fcname= (pos[i].flcolour.pixvalue<0)? (pos[i].flpixelCName)? pos[i].flpixelCName : fcname :
					AllAttrs[pos[i].flcolour.pixvalue].pixelCName;
			}
			op= pos[i].command;
			if( pos[i].attr.markFlag ){
			  double ms= AllSets[0].markSize;
				  /* Currently, xg_dot() infers the marker size from the markSize
				   \ field in the dataset array. So we temporarily use entry 0 to
				   \ cache this point's marker size.
				   */
				AllSets[0].markSize= pos[i].attr.markSize;
				wi->dev_info.xg_dot(wi->dev_info.user_state,
						pos[i].sx, pos[i].sy,
						P_MARK, ABS(pos[i].attr.markType)-1,
						pos[i].colour.pixvalue, pos[i].colour.pixelValue, 0, NULL);
				AllSets[0].markSize= ms;
				if( debugFlag && debugLevel> 0 ){
					fprintf( StdErr, " M%d(%d,%d,%s)", i, (int) pos[i].sx, (int) pos[i].sy, d2str( pos[i].attr.markSize, 0,0) );
					fflush( StdErr );
				}
				Pen->drawn += 1;
			}
			if( pos[i].textSet && pos[i].text ){
			  Pixel colour0= AllAttrs[0].pixelValue;
			  Pixel colour1= AllAttrs[1].pixelValue;
			  int twidth, theight, multilines;
				  /* We need to (re)determine the text's dimensions. This is because we cannot be sure
				   \ that the dimensions determined add the time off the call to PenText[] or PenTextBox[]
				   \ are appropriate for the actual situation. For instance, we may be printing now, or
				   \ maybe we're in raw or Quick mode.
				   */
				PenTextDimensions( wi, &pos[i] );
				twidth= pos[i].TextBox->w;
				theight= pos[i].TextBox->h;
#ifdef OLD_CRIPPLE_PENTEXTBOXRECT
				  /* 20010909: There is no need to pass via world-scaled co-ordinates to determine the box's
				   \ dimensions around the text!! It is much easier, and much more robust against different
				   \ axes scalings and transformations, to do that right here, in pixels (screen co-ordinates)!!
				   */
				if( op== XGPenTextBox ){
				  XGPenPosition *lpos= pos[i].TextBox;
				  double r[4], xlw, ylw, x_offset, y_offset, tw, th;
					if( FlushPenSegments( wi, Pen, &nondrawn_s, &linestyle, &lineWidth, &pixvalue, &pixelValue,
						&pos[i], &xseg, True )
					){
						flushed= True;
					}
					  /* xlw,ylw: correction for the lineWidth with which the box is outlined: */
					xlw= pLINEWIDTH(wi,lpos->attr.lineWidth)* wi->dev_info.var_width_factor;
					  /* We must have those in world co-ordinates! */
					ylw= xlw* wi->YUnitsPerPixel/ wi->Yscale;
					xlw*= wi->XUnitsPerPixel/ wi->Xscale;
					r[0]= lpos->x;
					r[1]= lpos->y;
					  /* lpos->[wh] contains the width and height of the string in screen co-ordinates. Convert
					   \ those to world co-ordinates:
					   */
					tw= lpos->w* wi->win_geo._XUnitsPerPixel/ wi->Xscale;
					th= lpos->h* wi->win_geo._YUnitsPerPixel/ wi->Yscale;
					  /* Size of the box is the string size plus the linewidth: */
					r[2]= tw+ xlw;
					r[3]= th+ ylw;
					  /* offset values that give some margins around the string: 2.5* bdr_pad in world co-ordinates	*/
					x_offset= wi->dev_info.bdr_pad/ 2.5* wi->win_geo._XUnitsPerPixel/ wi->Xscale;
					y_offset= wi->dev_info.bdr_pad/ 2.5* wi->win_geo._YUnitsPerPixel/ wi->Yscale;
					if( lpos->textJust== T_VERTICAL ){
#ifdef DEBUG
					  int ox, oy;
#endif
						  /* Fine-tuning by hand... The vertical printing routine offsets the string by a certain amount,
						   \ that seems to be well estimated by 0.15* width (the used font's height!) horizontally, and
						   \ -1.5* width vertically. I hope these values are universal - and valid until I get a better
						   \ grasp on what the text rotation actually does!
						   */
						r[0]-= 0.5* xlw- 0.15* tw+ x_offset;
						r[1]-= 0.5* ylw+ 1.5* tw+ y_offset;
#ifdef DEBUG
						ox= SCREENXDIM( wi, 0.15* tw );
						oy= SCREENYDIM( wi, 1.5* tw );
#endif
						r[2]+= 2* x_offset;
						r[3]+= 2* y_offset;
					}
					else{
						switch( lpos->textJust2[0] ){
						  /* Great! No additional correction is necessary for the horizontal position:
						   \ just incorporate the margin, linewidth, and determine the left corner position
						   \ as a function of justification:
						   */
							case -1:
								r[0]-= tw+ 0.5* xlw+ x_offset;
								r[2]+= 2* x_offset;
								break;
							case 0:
								r[0]= r[0]- r[2]/2- x_offset;
								r[2]+= 2* x_offset;
								break;
							case 1:
								r[0]-= 0.5* xlw+ x_offset;
								r[2]+= 2* x_offset;
								break;
						}
						switch( lpos->textJust2[1] ){
							case -1:
								r[1]-= 1* y_offset+ 0.5* ylw;
								r[3]+= 2* y_offset;
								break;
							case 0:
							  /* Some correction for the extra margin seems to be necessary?! */
								r[1]= r[1]- r[3]/ 2- 0.75* y_offset;
								r[3]+= 2* y_offset;
								break;
							case 1:
							  /* Vertically, the necessary shift seems to be well approximated by 0.1* th	*/
								r[1]-= (1-0.10)* th+ 0.5* ylw+ y_offset;
								r[3]+= 2* y_offset;
								break;
						}
					}
					{ double x= lpos->x, y= lpos->y;
						  /* Store the co-ordinates we just determined: */
						lpos->x= r[0], lpos->y= r[1], lpos->w= r[2], lpos->h= r[3];
						  /* And draw a rectangle... */
						_DrawPenRectangle( wi, Pen, lpos, i, linestyle, lineWidth, pixvalue, pixelValue, msg, fcname );
						  /* And restore the co-ordinates, since we still need them. For instance, when we're in
						   \ raw mode...!
						   */
						lpos->x= x, lpos->y= y, lpos->w= twidth, lpos->h= theight;
					}
				}
#endif
				if( pos[i].colour.pixvalue< 0 ){
					AllAttrs[0].pixelValue= pos[i].colour.pixelValue;
				}
				else{
					AllAttrs[0].pixelValue= AllAttrs[pos[i].colour.pixvalue % MAXATTR].pixelValue;
				}
				if( Pen->highlight_text ){
					AllAttrs[1].pixelValue= (Pen->hlpixelCName)? Pen->hlpixelValue : highlightPixel;
					xg_text_highlight= True;
					xg_text_highlight_colour= 1;
				}
				textPixel= AllAttrs[0].pixelValue;
				use_textPixel= 1;
				  /* for a single line of text, theight should equal textFntHeight! */
				if( pos[i].textJust== T_VERTICAL ){
					multilines= (twidth> pos[i].TextBox->textFntHeight)? True : False;
				}
				else{
					multilines= (theight> pos[i].TextBox->textFntHeight)? True : False;
				}
				if( multilines ){
				  int sx= pos[i].sx, sy= pos[i].sy, lheight= pos[i].TextBox->textFntHeight;
				  char *tbuf, *line;
					if( pos[i].textJust== T_VERTICAL ){
					  /* Some hand-tuned justification: */
						sx+= 0.1* lheight;
						sy+= 0.175* lheight;
					}
					else{
					  /* Compensate for the full height, but take into account the fact that
					   \ each line will be vertically adjusted by xg_text(). Thus, we must compensate
					   \ for the full height minus one line. This also is fine-tuned by hand...
					   */
						switch( pos[i].TextBox->textJust2[1] ){
							case 1:
								sy-= lheight* 0.1;
								break;
							case 0:
								sy-= (theight- lheight* 0.9)/2;
								break;
							case -1:
								sy-= (theight- lheight* 0.9);
								break;
						}
					}
					  /* 20010909: There is no need to pass via world-scaled co-ordinates to determine the box's
					   \ dimensions around the text!! It is much easier, and much more robust against different
					   \ axes scalings and transformations, to do that right here, in pixels (screen co-ordinates)!!
					   */
					if( op== XGPenTextBox ){
						if( FlushPenSegments( wi, Pen, &nondrawn_s, &linestyle, &lineWidth, &pixvalue, &pixelValue,
							&pos[i], &xseg, True )
						){
							flushed= True;
						}
						_DrawPenTextBoxRectangle( wi, Pen, pos, i, sx, sy, lheight, linestyle, lineWidth, pixvalue, pixelValue );
					}
					if( !xgraph_NameBuf || strlen(pos[i].text)> xgraph_NameBufLen ){
						xfree( xgraph_NameBuf );
						xgraph_NameBuf= XGstrdup( pos[i].text );
						xgraph_NameBufLen= strlen( xgraph_NameBuf );
					}
					else{
						strcpy( xgraph_NameBuf, pos[i].text );
					}
					tbuf= xgraph_NameBuf;
					while( tbuf && xtb_getline( &tbuf, &line ) ){
						if( Pen->highlight_text ){
							xg_text_highlight= True;
							xg_text_highlight_colour= 1;
						}
						wi->dev_info.xg_text( wi->dev_info.user_state,
							sx, sy,
							line, pos[i].textJust, pos[i].textFntNr,
							(pos[i].textFntNr<0 && pos[i].cfont_var)? pos[i].cfont_var->cfont : NULL
						);
						if( pos[i].textJust== T_VERTICAL ){
							sx+= lheight;
						}
						else{
							sy+= lheight;
						}
						Pen->drawn += 1;
					}
				}
				else{
				  int sx= pos[i].sx, sy= pos[i].sy, lheight= pos[i].TextBox->textFntHeight;
/* 				  int lheight= pos[i].TextBox->textFntHeight;	*/
/* 					if( pos[i].textJust== T_VERTICAL ){	*/
/* 						sy+= 0.175* lheight;	*/
/* 					}	*/
					  /* 20010909: There is no need to pass via world-scaled co-ordinates to determine the box's
					   \ dimensions around the text!! It is much easier, and much more robust against different
					   \ axes scalings and transformations, to do that right here, in pixels (screen co-ordinates)!!
					   */
					if( op== XGPenTextBox ){
						if( FlushPenSegments( wi, Pen, &nondrawn_s, &linestyle, &lineWidth, &pixvalue, &pixelValue,
							&pos[i], &xseg, True )
						){
							flushed= True;
						}
						_DrawPenTextBoxRectangle( wi, Pen, pos, i, sx, sy, lheight, linestyle, lineWidth, pixvalue, pixelValue );
					}
					wi->dev_info.xg_text( wi->dev_info.user_state,
						sx, sy,
						pos[i].text, pos[i].textJust, pos[i].textFntNr,
						(pos[i].textFntNr<0 && pos[i].cfont_var)? pos[i].cfont_var->cfont : NULL
					);
					Pen->drawn += 1;
				}
				AllAttrs[0].pixelValue= colour0;
				AllAttrs[1].pixelValue= colour1;
				  /* we currently take use_textPixel as a once-only switch, so we don't
				   \ save/restore its state!
				   */
				use_textPixel= 0;
				if( debugFlag && debugLevel> 0 ){
					fprintf( StdErr, " T%d(%d,%d,%s)", i, (int) pos[i].sx, (int) pos[i].sy,
						pos[i].text
					);
					fflush( StdErr );
				}
			}
		}
	}
	if( debugFlag && debugLevel> 0 ){
		fputc( '\n', StdErr );
	}
}

/* ================ Here start the ascanf callback routines for manipulating XGPens, extracted from xgsupport.c ================ */

DEFUN( ascanf_SelectPen, ( ASCB_ARGLIST ), int );
DEFUN( ascanf_PenNumPoints, ( ASCB_ARGLIST ), int );
DEFUN( ascanf_PenOverwrite, ( ASCB_ARGLIST ), int );
DEFUN( ascanf_PenMoveTo, ( ASCB_ARGLIST ), int );
DEFUN( ascanf_PenLineTo, ( ASCB_ARGLIST ), int );
DEFUN( ascanf_PenLineStyleWidth, ( ASCB_ARGLIST ), int );
DEFUN( ascanf_PenColour, ( ASCB_ARGLIST ), int );

// 20080710: pens are created with (and reset to) a hard-coded default initial allocation size
// (PenAllocSize can grow
#define PENALLOCSIZE	16
int PenAllocSize= PENALLOCSIZE;

XGPen *CheckPens( LocalWin *wi, int pen_nr )
{ XGPen *npen= NULL, *pen_list= wi->pen_list;
  int n= 1;
  static char active= 0;
	if( pen_nr< 0 ){
		return( NULL );
	}
	if( !pen_list || pen_nr>= wi->numPens ){
		  /* Find the end of an existing list	*/
		while( pen_list && pen_list->next ){
			pen_list= pen_list->next;
		}
		active= 1;
		while( pen_nr>= wi->numPens ){
			if( (npen= (XGPen*) calloc( 1, sizeof(XGPen))) ){
				npen->wi= wi;
				npen->allocSize= PENALLOCSIZE;
				npen->pen_nr= wi->numPens;
				npen->set_link= -1;
				npen->before_set= npen->after_set= -1;
				if( !wi->pen_list ){
					pen_list= wi->pen_list= npen;
				}
				else{
					pen_list->next= npen;
					pen_list= pen_list->next;
				}
				n+= 1;
				wi->numPens+= 1;
			}
			else{
				fprintf( StdErr, "CheckPens(): can't allocate pen #%d (new %d): %s\n",
					pen_nr, n, (active)? "called recursively" : serror()
				);
				active= 0;
				return(NULL);
			}
		}
		active= 0;
		return( npen );
	}
	else{
		n= 0;
		while( pen_list && pen_nr> n ){
			pen_list= pen_list->next;
			n+= 1;
		}
		return( pen_list );
	}
}

XGPenPosition *Alloc_PenPositions( XGPen *Pen, int N )
{ XGPenPosition *pos= NULL;
  static char active= 0;
	if( Pen ){
		if( N< 0 ){
			N= Pen->positions+ Pen->allocSize;
		}
		else if( N== 0 ){
			Pen->allocSize= N= 1;
		}
		else{
			PenAllocSize= Pen->allocSize= N;
			if( N== Pen->positions && Pen->position ){
				return( Pen->position );
			}
		}
		if( !active ){
			active= 1;
			pos = Pen->position;
			if( pos ){
			  int i;
				if( N< Pen->positions ){
					for( i= 0; i< Pen->positions; i++ ){
						if( pos[i].pixelCName ){
							FreeColor( &(pos[i].colour.pixelValue), &pos[i].pixelCName );
						}
						if( pos[i].flpixelCName ){
							FreeColor( &(pos[i].flcolour.pixelValue), &pos[i].flpixelCName );
						}
						pos[i].cfont_var= NULL;
						if( pos[i].TextBox ){
							xfree( pos[i].TextBox );
						}
						// 20080710:
						xfree( pos[i].text );
					}
				}
				if( debugFlag && debugLevel > 0 ){
					fprintf( StdErr, "Alloc_PenPosition(pen %d): expanding pen from %d to %d positions\n",
						    Pen->pen_nr, Pen->positions, N
					);
				}
				if( (pos= (XGPenPosition*) realloc( pos, N* sizeof(XGPenPosition) )) ){
					if( N> Pen->positions ){
					  /* zero out the new part:	*/
						memset( &pos[Pen->positions], 0, sizeof(XGPenPosition)* (N- Pen->positions) );
					}
					Pen->position= pos;
					Pen->positions= N;
					if( Pen->current_pos>= N || Pen->current_pos< 0 ){
						Pen->current_pos= N- 1;
					}
				}
				else{
					fprintf( StdErr, "Alloc_PenPosition(pen %d): could not reallocate %d positions to %d: %s\n",
						Pen->pen_nr, Pen->positions, N, serror()
					);
					Pen->position= NULL;
					Pen->positions= 0;
					Pen->current_pos= 0;
				}
			}
			else{
				if( debugFlag && debugLevel > 1 ){
					fprintf( StdErr, "Alloc_PenPosition(pen %d): initialising pen to %d positions\n",
						    Pen->pen_nr, N
					);
				}
				if( (pos= (XGPenPosition*) calloc( N, sizeof(XGPenPosition))) ){
					Pen->position= pos;
					Pen->positions= N;
					Pen->current_pos= 0;
				}
				else{
					fprintf( StdErr, "Alloc_PenPosition(pen %d): could not allocate %d positions: %s\n",
						Pen->pen_nr, N, serror()
					);
					Pen->positions= 0;
					Pen->current_pos= 0;
				}
			}
			active= 0;
		}
		return( pos );
	}
	else{
		return( NULL );
	}
}

// 20100615: check if Pen has place for N additional positions, and expand if necessary.
// the global and Pen's default allocation sizes are restored.
XGPenPosition *CheckExpand_PenPositions( XGPen *Pen, int N )
{ int PAS = PenAllocSize, PaS;
	if( Pen ){
	  XGPenPosition *ret;
		PaS = Pen->allocSize;
		if( !Pen->position || !Pen->positions ){
			ret = Alloc_PenPositions( Pen, N );
		}
		else if( Pen->current_pos+N >= Pen->positions ){
//			ret = Alloc_PenPositions( Pen, Pen->Positions + N - (Pen->positions - Pen->current_pos) );
			ret = Alloc_PenPositions( Pen, N + Pen->current_pos );
		}
		PenAllocSize = PAS;
		Pen->allocSize = PaS;
		return( ret );
	}
	else{
		return( NULL );
	}
}

static ascanf_Function *PenPolyX= NULL, *PenPolyY= NULL;

int AddPenPosition( XGPen *Pen, XGPenPosition *vals, XGPenOperation mask )
{ XGPenPosition *pos;
  int ret= -1;
  static char active= 0;
	if( Pen && vals && !active ){
		active= 1;
		if( !Pen->position || !Pen->positions || Pen->current_pos== Pen->positions ){
			Alloc_PenPositions( Pen, -1 );
			if( Pen->current_pos ){
			  /* Forward copy the settings of the current pen to the next.
			   \ This should take care of propagating the current attributes.
			   */
				pos= &(Pen->position[Pen->current_pos]);
				if( Pen->current_pos ){
					pos[0].attr= pos[-1].attr;
					  /* Some things must not be inherited.	*/
					if( !pos[0].pixelCName ){
						pos[0].colour= pos[-1].colour;
					}
					if( !pos[0].flpixelCName ){
						pos[0].flcolour= pos[-1].flcolour;
					}
					if( !pos[0].cfont_var ){
						pos[0].cfont_var= pos[-1].cfont_var;
					}
				}
			}
		}
		pos= &(Pen->position[Pen->current_pos]);
		if( CheckMask( mask, XGP_position) ){
			  /* If all goes well, we will return the new pen-position	*/
			ret= (Pen->current_pos+ 1);
			  /* Store the x and y, common for all commands (Polygons are a little different):	*/
			pos->x= vals->x;
			pos->y= vals->y;
			  /* And store the type of command	*/
			pos->command= vals->command;
			  /* Now do command specific things:	*/
			switch( pos->command ){
				default:
					ret= 1;
					break;
				case XGPenMoveTo:
				case XGPenLineTo:
					if( !NaN(vals->orn) ){
						pos->orn= vals->orn;
					}
					break;
				case XGPenDrawRect:
				case XGPenFillRect:
				case XGPenDrawCRect:
				case XGPenFillCRect:
				case XGPenDrawEllps:
				case XGPenText:
				case XGPenTextBox:
					pos->w= vals->w;
					pos->h= vals->h;
					pos->not_outlined= vals->not_outlined;
					ret= 1;
					break;
				case XGPenDrawPoly:
				case XGPenFillPoly:{
				  int i;
					pos->polyN_1= vals->polyN_1;
					if( pos->polyN_1 &&
						  /* Currently, always size the arrays to the actual number of points
						   \ (polyMax is redundant)
						   */
						((!pos->polyX || !pos->polyY) || pos->polyN_1!= pos->polyMax)
					){
						if( !(pos->polyX= (double*) XGrealloc( pos->polyX, pos->polyN_1* sizeof(double))) ){
							ascanf_emsg= (char*) serror();
							ascanf_arg_error= True;
						}
						if( !(pos->polyY= (double*) XGrealloc( pos->polyY, pos->polyN_1* sizeof(double))) ){
							ascanf_emsg= (char*) serror();
							ascanf_arg_error= True;
						}
						pos->polyMax= pos->polyN_1;
					}
					if( pos->polyX && pos->polyY ){
						if( PenPolyX->iarray ){
							pos->x= PenPolyX->iarray[0];
							pos->y= PenPolyY->iarray[0];
							for( i= 0; i< pos->polyN_1; i++ ){
								pos->polyX[i]= PenPolyX->iarray[i+1];
								pos->polyY[i]= PenPolyY->iarray[i+1];
							}
						}
						else{
							pos->x= PenPolyX->array[0];
							pos->y= PenPolyY->array[0];
							for( i= 0; i< pos->polyN_1; i++ ){
								pos->polyX[i]= PenPolyX->array[i+1];
								pos->polyY[i]= PenPolyY->array[i+1];
							}
						}
					}
					else{
						xfree( pos->polyX );
						xfree( pos->polyY );
						pos->polyN_1= pos->polyMax= 0;
						  /* Error!! No need to increase the penpointer in this case...	*/
						ret= -1;
					}
					pos->not_outlined= vals->not_outlined;
					break;
				}
			}
			if( ret!= -1 ){
				Pen->current_pos+= 1;
				if( Pen->current_pos< Pen->positions ){
				  /* Forward copy the settings of the current pen to the next.
				   \ This should take care of propagating the current attributes.
				   */
					pos[1].attr= pos->attr;
					  /* Some things must not be inherited.	*/
					if( !pos[1].pixelCName ){
						pos[1].colour= pos->colour;
					}
					if( !pos[1].flpixelCName ){
						pos[1].flcolour= pos->flcolour;
					}
					if( !pos[1].cfont_var ){
						pos[1].cfont_var= pos->cfont_var;
					}
/* 					pos[1].pixelCName= pos[1].flpixelCName= NULL;	*/
/* 					pos[1].text= NULL;	*/
/* 					pos[1].textSet= False;	*/
/* 					pos[1].polyX= pos[1].polyY= NULL;	*/
/* 					pos[1].polyN_1= pos[1].polyMax= 0;	*/
				}
			}
		}
		if( CheckMask( mask, XGP_noClip ) ){
			pos->attr.noclip= vals->attr.noclip;
			ret= 1;
		}
		if( CheckMask( mask, XGP_markFlag ) ){
			pos->attr.markFlag= vals->attr.markFlag;
			ret= 1;
		}
		if( CheckMask( mask, XGP_markType ) ){
			pos->attr.markType= vals->attr.markType;
			ret= 1;
		}
		if( CheckMask( mask, XGP_markSize ) ){
			pos->attr.markSize= vals->attr.markSize;
			ret= 1;
		}
		if( CheckMask( mask, XGP_text ) ){
			pos->textSet= True;
			if( !pos->text || !vals->text || strcmp( pos->text, vals->text ) ){
				xfree( pos->text );
				pos->text= XGstrdup( vals->text );
			}
			ret= 1;
		}
		if( CheckMask( mask, XGP_textJust ) ){
			pos->textJust= vals->textJust;
			ret= 1;
		}
		if( CheckMask( mask, XGP_textFntNr ) ){
			if( (pos->textFntNr= vals->textFntNr)< 0 ){
				pos->cfont_var= vals->cfont_var;
			}
			ret= 1;
		}
		if( CheckMask( mask, XGP_textOutside ) ){
			pos->attr.text_outside= vals->attr.text_outside;
			ret= 1;
		}
		if( CheckMask( mask, XGP_linestyle ) ){
			pos->attr.linestyle= vals->attr.linestyle;
			ret= 1;
		}
		if( CheckMask( mask, XGP_lineWidth ) ){
			pos->attr.lineWidth= vals->attr.lineWidth;
			ret= 1;
		}
		if( CheckMask( mask, XGP_colour ) ){
		  Pixel tp;
		  int nv= (vals->colour.pixvalue % MAXATTR );
			if( pos->colour.pixvalue== nv &&
				((pos->pixelCName== NULL && nv>=0) ||
				(pos->pixelCName && strncmp( pos->pixelCName, vals->pixelCName, strlen(vals->pixelCName))== 0))
			){
				ret= 0;
			}
			else if( (pos->colour.pixvalue= nv )< 0 ){
				if( vals->pixelCName && GetCMapColor( vals->pixelCName, &tp, Pen->wi->cmap ) ){
					if( pos->pixelCName ){
						FreeColor( &(pos->colour.pixelValue), &pos->pixelCName );
					}
					pos->colour.pixelValue= tp;
					StoreCName( pos->pixelCName );
					if( pragma_unlikely(ascanf_verbose) ){
						fprintf( StdErr, " (new pen #%d[%d] colour %s p%ld) ",
							Pen->pen_nr, Pen->current_pos, pos->pixelCName, pos->colour.pixelValue
						);
					}
					ret= 1;
				}
				else{
					ascanf_emsg= " (invalid colour specification)== ";
					ascanf_arg_error= True;
					pos->colour.pixvalue= 0;
					ret= 0;
				}
			}
			else if( pos->pixelCName ){
				FreeColor( &(pos->colour.pixelValue), &pos->pixelCName );
				ret= 1;
			}
		}
		if( CheckMask( mask, XGP_flcolour ) ){
		  Pixel tp;
		  int nv= (vals->flcolour.pixvalue % MAXATTR );
			if( pos->flcolour.pixvalue== nv &&
				((pos->flpixelCName== NULL && nv>=0) ||
				(pos->flpixelCName && strncmp( pos->flpixelCName, vals->flpixelCName, strlen(vals->flpixelCName))== 0))
			){
				ret= 0;
			}
			else if( (pos->flcolour.pixvalue= nv )< 0 ){
				if( GetCMapColor( vals->flpixelCName, &tp, Pen->wi->cmap ) ){
					if( pos->flpixelCName ){
						FreeColor( &(pos->flcolour.pixelValue), &pos->flpixelCName );
					}
					pos->flcolour.pixelValue= tp;
					StoreCName( pos->flpixelCName );
					if( pragma_unlikely(ascanf_verbose) ){
						fprintf( StdErr, " (new pen #%d[%d] fill colour %s p%ld) ",
							Pen->pen_nr, Pen->current_pos, pos->flpixelCName, pos->flcolour.pixelValue
						);
					}
					ret= 1;
				}
			}
			else if( pos->flpixelCName ){
				FreeColor( &(pos->flcolour.pixelValue), &pos->flpixelCName );
				ret= 1;
			}
		}
		if( (pos->command== XGPenTextBox && vals->command== XGPenTextBox) ||
			(pos->command== XGPenText && vals->command== XGPenText)
		){
			  /* Extreme care should be taken with the contents of the TextBox field! */
			if( !pos->TextBox ){
				pos->TextBox= (XGPenPosition*) calloc( 1, sizeof(XGPenPosition));
			}
			if( pos->TextBox ){
				*(pos->TextBox)= *pos;
				pos->TextBox->x= vals->x;
				pos->TextBox->y= vals->y;
				pos->TextBox->w= vals->TextBox->w;
				pos->TextBox->h= vals->TextBox->h;
				pos->TextBox->command= vals->TextBox->command;
				pos->TextBox->not_outlined= vals->TextBox->not_outlined;
				pos->TextBox->TextBox= NULL;
				memcpy( pos->TextBox->textJust2, vals->textJust2, sizeof(vals->textJust2) );
			}
		}
		else{
			xfree(pos->TextBox);
		}
		pos->operation |= mask;
		active= 0;
	}
	else if( active ){
		fprintf( StdErr, "AddPenPosition(#%s,\"%s\"): called with NULL Pen or vals or recursively.\n",
			(Pen)? d2str(Pen->pen_nr, 0,0) : "NULL",
			(vals)? PenCommandName[vals->command] : "NULL"
		);
	}
	return(ret);
}

#if defined(DEBUG) && defined(ASCANF_ALTERNATE)

int PenPosFrame( XGPen *Pen, int pos, ASCB_ARGLIST )
{ XGPenPosition *Pos;
	if( pos< 0 || pos>= Pen->positions ){
		pos= Pen->current_pos;
	}
	if( Pen->position && pos< Pen->positions ){
		Pos= &Pen->position[ pos ];
	}
	if( Pos ){
		if( ascanf_frame_has_expr && AH_EXPR ){
			xfree( Pos->expr );
			Pos->expr= strdup( AH_EXPR );
		}
		if( TBARprogress_header ){
			xfree( Pos->caller );
			Pos->caller= strdup( TBARprogress_header );
		}
		Pos->level= *(__ascb_frame->level);
	}
	return( pos );
}

#endif

int ascanf_NumPens ( ASCB_ARGLIST )
{ ASCB_FRAME_RESULT
	if( !ascanf_SyntaxCheck && ActiveWin && ActiveWin!= &StubWindow ){
		*result= ActiveWin->numPens;
	}
	else{
		*result= 0;
	}
	return(1);
}

int ascanf_SelectPen ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  int pen_nr;
	if( !args || ascanf_arguments< 1 ){
		if( !ascanf_SyntaxCheck && ActiveWin && ActiveWin!= &StubWindow && ActiveWin->current_pen ){
			*result= ActiveWin->current_pen->pen_nr;
		}
		else{
			*result= -1;
			ascanf_arg_error= True;
		}
		return(1);
	}
	if( args[0]< 0 || args[0]>= MAXINT ){
		ascanf_emsg= " <pen number out of range> ";
		ascanf_arg_error= True;
		return(1);
	}
	else{
		pen_nr= (int) args[0];
	}
	if( !ascanf_SyntaxCheck && ActiveWin && ActiveWin!= &StubWindow ){
		if( (ActiveWin->current_pen= CheckPens( ActiveWin, pen_nr)) ){
			*result= ActiveWin->current_pen->positions;
		}
		else{
			*result= -1;
			ascanf_emsg= " (could not get pen) ";
		}
	}
	return(1);
}

int ascanf_PenNumPoints ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  int posN;
	/* 	if( !args || ascanf_arguments< 1 ){	*/
	/* 		ascanf_arg_error= True;	*/
	/* 		return(1);	*/
	/* 	}	*/
	if( !ascanf_SyntaxCheck && ActiveWin && ActiveWin!= &StubWindow ){
		if( ActiveWin->current_pen ){
			if( args && ascanf_arguments>= 1 ){
				CLIP_EXPR_CAST( int, posN, double, args[0], 1, MAXINT );
				Alloc_PenPositions( ActiveWin->current_pen, posN );
				*result= ActiveWin->current_pen->positions;
			}
			else{
				*result= ActiveWin->current_pen->current_pos;
			}
		}
		else{
			*result= -1;
			ascanf_emsg= " (no current pen) ";
			ascanf_arg_error= True;
		}
	}
	return(1);
}

int ascanf_PenDrawn ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
	if( !ascanf_SyntaxCheck && ActiveWin && ActiveWin!= &StubWindow ){
		if( ActiveWin->current_pen ){
			if( !args || ascanf_arguments < 1 || ASCANF_FALSE(args[0]) ){
				*result = ActiveWin->current_pen->drawn;
			}
			else if( ASCANF_TRUE(args[0]) ){
				if( (!ActiveWin->no_pens && !PENSKIP(ActiveWin,ActiveWin->current_pen))
					|| PENLINKDRAWN(ActiveWin,ActiveWin->current_pen)
				){
					  /* Mark this pen as drawn.
					   \ This does not prevent it from being drawn another time if SkipOnce is not
					   \ also set! It only serves to include it in autoscaling if SkipOnce is set.
					   */
					*result = ActiveWin->current_pen->drawn = True;
				}
				else{
					if( pragma_unlikely(ascanf_verbose) ){
						fprintf( StdErr, " (pen not drawn because of NoPens, pen->set_link or pen->skip setting) " );
					}
					*result = ActiveWin->current_pen->drawn= False;
				}
			}
		}
		else{
			set_NaN(*result);
			ascanf_emsg= " (no current pen) ";
			ascanf_arg_error= True;
		}
	}
	return(1);
}

int PenReset( XGPen *pen, int dealloc )
{ int r= 0;
	if( pen ){
	  XGPenPosition *pos= pen->position;
	  int i;
		r= pen->current_pos;
		for( i= 0; i< pen->positions; i++ ){
			pos[i].textSet= 0;
			if( dealloc ){
				xfree( pen->position[i].text );
				xfree( pen->position[i].TextBox );
				if( pen->position[i].colour.pixvalue< 0 && pen->position[i].pixelCName ){
					FreeColor( &(pen->position[i].colour.pixelValue), &pen->position[i].pixelCName );
				}
				if( pen->position[i].flcolour.pixvalue< 0 && pen->position[i].flpixelCName ){
					FreeColor( &(pen->position[i].flcolour.pixelValue), &pen->position[i].flpixelCName );
/* 					XFreeColors( disp, cmap, &pen->position[i].flpixelValue, 1, 0);	*/
/* 					xfree( pen->position[i].flpixelCName );	*/
				}
				pen->position[i].operation = 0;
			}
			else{
				// reset all bits except for the colour bits:
				pen->position[i].operation &= (XGP_colour | XGP_flcolour);
			}
		}
		if( dealloc ){
			if( pen->hlpixelCName ){
				FreeColor( &pen->hlpixelValue, &pen->hlpixelCName );
			}
			xfree( pen->position );
			pen->positions= 0;
			pen->allocSize= PENALLOCSIZE;
			xfree( pen->penName );
			xfree( pen->pen_info );
		}
		pen->current_pos= 0;
		pen->skip= False;
		pen->drawn= False;
	}
	return(r);
}

int ascanf_PenReset ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  int pen_nr= -1, dealloc= False;
	if( args ){
		if( ascanf_arguments > 0 ){
			if( (args[0]< -MAXINT || args[0]>= ((ActiveWin && ActiveWin!= &StubWindow)? ActiveWin->numPens : MAXINT) ) ){
				ascanf_emsg= " <pen number out of range> ";
				ascanf_arg_error= True;
				return(1);
			}
			else{
				pen_nr= (int) args[0];
				if( ascanf_arguments> 1 && ASCANF_TRUE(args[1]) ){
					dealloc= True;
				}
			}
		}
	}
	if( !ascanf_SyntaxCheck && ActiveWin && ActiveWin!= &StubWindow ){
		if( pen_nr>= 0 ){
		  XGPen *pen= CheckPens( ActiveWin, pen_nr);
			if( pen ){
				*result= PenReset( pen, dealloc );
			}
			else{
				*result= -1;
				ascanf_emsg= " (can't get pen) ";
				ascanf_arg_error= True;
			}
		}
		else if( pen_nr< 0 && ascanf_arguments ){
		  XGPen *pen= ActiveWin->pen_list;
			*result= 0;
			while( pen ){
				*result+= PenReset( pen, dealloc );
				pen= pen->next;
			}
		}
		else if( ActiveWin->current_pen ){
			*result= PenReset( ActiveWin->current_pen, dealloc );
		}
		else{
			*result= -1;
			ascanf_emsg= " (no current pen) ";
			ascanf_arg_error= True;
		}
	}
	return(1);
}

int ascanf_PenDrawNow ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
	if( !ascanf_SyntaxCheck && ActiveWin && ActiveWin!= &StubWindow ){
		if( ActiveWin->current_pen ){
			if( (!ActiveWin->no_pens && !PENSKIP(ActiveWin,ActiveWin->current_pen))
				|| (ascanf_arguments && args && args[0] && PENLINKDRAWN(ActiveWin,ActiveWin->current_pen))
			){
				DrawPen( ActiveWin, ActiveWin->current_pen );
				  /* Mark this pen as drawn.
				   \ This does not prevent it from being drawn another time if SkipOnce is not
				   \ also set! It only serves to include it in autoscaling if SkipOnce is set.
				   */
				if( !ActiveWin->current_pen->drawn ){
					ActiveWin->current_pen->drawn= True;
				}
			}
			else if( pragma_unlikely(ascanf_verbose) ){
				fprintf( StdErr, " (pen not drawn because of NoPens, pen->set_link or pen->skip setting) " );
			}
			*result= ActiveWin->current_pen->pen_nr;
		}
		else{
			*result= -1;
			ascanf_emsg= " (no current pen) ";
			ascanf_arg_error= True;
		}
	}
	return(1);
}

int ascanf_PenOverwrite ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
	if( !args || ascanf_arguments< 1 ){
		if( !ascanf_SyntaxCheck && ActiveWin && ActiveWin!= &StubWindow && ActiveWin->current_pen ){
			*result= ActiveWin->current_pen->overwrite_pen;
		}
		else{
			*result= -1;
			ascanf_arg_error= True;
		}
		return(1);
	}
	if( !ascanf_SyntaxCheck && ActiveWin && ActiveWin!= &StubWindow ){
		if( ActiveWin->current_pen ){
			*result= ActiveWin->current_pen->overwrite_pen;
			ActiveWin->current_pen->overwrite_pen= ASCANF_TRUE(args[0])? True : False;
		}
		else{
			*result= -1;
			ascanf_emsg= " (no current pen) ";
			ascanf_arg_error= True;
		}
	}
	return(1);
}

int ascanf_PenBeforeSet ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
	if( !args || ascanf_arguments< 1 ){
		if( !ascanf_SyntaxCheck && ActiveWin && ActiveWin!= &StubWindow && ActiveWin->current_pen ){
			*result= ActiveWin->current_pen->before_set;
		}
		else{
			*result= -1;
			ascanf_arg_error= True;
		}
		return(1);
	}
	if( !ascanf_SyntaxCheck && ActiveWin && ActiveWin!= &StubWindow ){
		if( ActiveWin->current_pen ){
			*result= ActiveWin->current_pen->before_set;
			CLIP_EXPR_CAST( int, ActiveWin->current_pen->before_set, double, args[0], -1, setNumber );
		}
		else{
			*result= -1;
			ascanf_emsg= " (no current pen) ";
			ascanf_arg_error= True;
		}
	}
	return(1);
}

int ascanf_PenAfterSet ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
	if( !args || ascanf_arguments< 1 ){
		if( !ascanf_SyntaxCheck && ActiveWin && ActiveWin!= &StubWindow && ActiveWin->current_pen ){
			*result= ActiveWin->current_pen->after_set;
		}
		else{
			*result= -1;
			ascanf_arg_error= True;
		}
		return(1);
	}
	if( !ascanf_SyntaxCheck && ActiveWin && ActiveWin!= &StubWindow ){
		if( ActiveWin->current_pen ){
			*result= ActiveWin->current_pen->after_set;
			CLIP_EXPR_CAST( int, ActiveWin->current_pen->after_set, double, args[0], -1, setNumber );
		}
		else{
			*result= -1;
			ascanf_emsg= " (no current pen) ";
			ascanf_arg_error= True;
		}
	}
	return(1);
}

int ascanf_PenFloating ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
	if( !args || ascanf_arguments< 1 ){
		if( !ascanf_SyntaxCheck && ActiveWin && ActiveWin!= &StubWindow && ActiveWin->current_pen ){
			*result= ActiveWin->current_pen->floating;
		}
		else{
			*result= -1;
			ascanf_arg_error= True;
		}
		return(1);
	}
	if( !ascanf_SyntaxCheck && ActiveWin && ActiveWin!= &StubWindow ){
		if( ActiveWin->current_pen ){
			*result= ActiveWin->current_pen->floating;
			ActiveWin->current_pen->floating= ASCANF_TRUE(args[0])? True : False;
		}
		else{
			*result= -1;
			ascanf_emsg= " (no current pen) ";
			ascanf_arg_error= True;
		}
	}
	return(1);
}

int ascanf_PenHighlightColour ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  ascanf_Function *af;
  char *cname= NULL;
	if( !args || ascanf_arguments< 1 ){
		if( !ascanf_SyntaxCheck && ActiveWin && ActiveWin!= &StubWindow && ActiveWin->current_pen ){
			*result= ActiveWin->current_pen->highlight;
		}
		else{
			*result= -1;
			ascanf_arg_error= True;
		}
		return(1);
	}
	if( ascanf_arguments> 1 && (af= parse_ascanf_address(args[1], _ascanf_variable, "ascanf_PenHighlightColour", False, NULL)) ){
		if( af->usage ){
			cname= af->usage;
		}
		else{
			ascanf_emsg= " (illegal or void colourname (string)pointer) ";
			ascanf_arg_error= True;
		}
	}
	else if( args[1]>= 0 && args[1]< MAXINT ){
		cname= AllAttrs[ ((int) args[1]) % MAXATTR ].pixelCName;
	}
	if( !ascanf_SyntaxCheck && ActiveWin && ActiveWin!= &StubWindow ){
	  XGPen *pen= ActiveWin->current_pen;
		if( pen ){
			*result= pen->highlight;
			pen->highlight= ASCANF_TRUE(args[0])? True : False;
			if( args[0] && ascanf_arguments> 1 ){
				if( cname ){
					if( (!pen->hlpixelCName || strncmp( pen->hlpixelCName, cname, strlen(cname))) ){
					  Pixel tp;
						if( GetCMapColor( cname, &tp, pen->wi->cmap ) ){
							if( pen->hlpixelCName ){
								FreeColor( &(pen->hlpixelValue), &pen->hlpixelCName );
							}
							pen->hlpixelValue= tp;
							StoreCName( pen->hlpixelCName );
							if( pragma_unlikely(ascanf_verbose) ){
								fprintf( StdErr, " (new pen #%d highlight colour %s p%ld) ",
									pen->pen_nr, pen->hlpixelCName, pen->hlpixelValue
								);
							}
						}
					}
				}
				else{
					if( pen->hlpixelCName ){
						if( pragma_unlikely(ascanf_verbose) ){
							fprintf( StdErr, " (will use global highlight colour %s) ", highlightCName );
						}
						FreeColor( &(pen->hlpixelValue), &pen->hlpixelCName );
					}
				}
			}
		}
		else{
			*result= -1;
			ascanf_emsg= " (no current pen) ";
			ascanf_arg_error= True;
		}
	}
	return(1);
}

int ascanf_PenHighlightText ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
	if( !args || ascanf_arguments< 1 ){
		if( !ascanf_SyntaxCheck && ActiveWin && ActiveWin!= &StubWindow && ActiveWin->current_pen ){
			*result= ActiveWin->current_pen->highlight_text;
		}
		else{
			*result= -1;
			ascanf_arg_error= True;
		}
		return(1);
	}
	if( !ascanf_SyntaxCheck && ActiveWin && ActiveWin!= &StubWindow ){
		if( ActiveWin->current_pen ){
			*result= ActiveWin->current_pen->highlight_text;
			ActiveWin->current_pen->highlight_text= ASCANF_TRUE(args[0])? True : False;
		}
		else{
			*result= -1;
			ascanf_emsg= " (no current pen) ";
			ascanf_arg_error= True;
		}
	}
	return(1);
}

int ascanf_PenSkip ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
	if( !args || ascanf_arguments< 1 ){
		ascanf_arg_error= True;
		return(1);
	}
	if( !ascanf_SyntaxCheck && ActiveWin && ActiveWin!= &StubWindow ){
		if( ActiveWin->current_pen ){
			*result= ActiveWin->current_pen->skip;
			ActiveWin->current_pen->skip= ASCANF_TRUE(args[0])? True : False;
		}
		else{
			*result= -1;
			ascanf_emsg= " (no current pen) ";
			ascanf_arg_error= True;
		}
	}
	return(1);
}

int ascanf_PenSkip2 ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  int pen_nr= -1, skip= True;
	if( ascanf_arguments > 0 && args ){
		if( (args[0]< -MAXINT || args[0]>= ((ActiveWin && ActiveWin!= &StubWindow)? ActiveWin->numPens : MAXINT) ) ){
			ascanf_emsg= " <pen number out of range> ";
			ascanf_arg_error= True;
			return(1);
		}
		else{
			pen_nr= (int) args[0];
			if( ascanf_arguments> 1 ){
				skip= ASCANF_TRUE(args[1])? True : False;
			}
		}
	}
	else if( !ascanf_SyntaxCheck && ActiveWin && ActiveWin!= &StubWindow && ActiveWin->current_pen ){
		*result= ActiveWin->current_pen->skip;
	}
	if( !ascanf_SyntaxCheck && ActiveWin && ActiveWin!= &StubWindow ){
		if( pen_nr>= 0 ){
		  XGPen *pen= CheckPens( ActiveWin, pen_nr);
			if( pen ){
				*result= pen->skip;
				pen->skip= skip;
			}
			else{
				*result= -1;
				ascanf_emsg= " (can't get pen) ";
				ascanf_arg_error= True;
			}
		}
		else if( pen_nr< 0 && ascanf_arguments ){
		  XGPen *pen= ActiveWin->pen_list;
			set_NaN(*result);
			while( pen ){
				pen->skip= skip;
				pen= pen->next;
			}
		}
		else if( ActiveWin->current_pen ){
			*result= ActiveWin->current_pen->skip;
			ActiveWin->current_pen->skip= skip;
		}
		else{
			*result= -1;
			ascanf_emsg= " (no current pen) ";
			ascanf_arg_error= True;
		}
	}
	return(1);
}

int ascanf_PenSetLink ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( !ascanf_SyntaxCheck && ActiveWin && ActiveWin!= &StubWindow ){
		if( ActiveWin->current_pen ){
			*result= ActiveWin->current_pen->set_link;
			if( args && ascanf_arguments>= 1 ){
			  int idx= -1;
				if( args[0]< 0 || args[0]>= setNumber ){
					ascanf_emsg= " (invalid set number) ";
					ascanf_arg_error= True;
				}
				else{
					idx= (NaN(args[0]))? -1 : ((int) args[0]);
				}
				if( !ascanf_arg_error ){
					ActiveWin->current_pen->set_link= idx;
				}
			}
		}
		else{
			*result= -1;
			ascanf_emsg= " (no current pen) ";
			ascanf_arg_error= True;
		}
	}
	return(1);
}

int ascanf_PenInfo ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  ascanf_Function *af= NULL;
  int take_usage= 0;
  static ascanf_Function AF= {NULL};
  static char *AFname= "PenInfo-Static-StringPointer";
  int pen_nr= -1;
	if( !ascanf_SyntaxCheck && ActiveWin && ActiveWin!= &StubWindow ){
		if( ActiveWin->current_pen ){
			pen_nr= ActiveWin->current_pen->pen_nr;
			ascanf_arg_error= False;
		}
		else{
			ascanf_arg_error= True;
		}
		if( args && args[0] ){
			af= parse_ascanf_address( args[0], _ascanf_variable, "ascanf_PenInfo", (int) ascanf_verbose, &take_usage );
		}
		if( !af ){
			af= init_Static_StringPointer( &AF, AFname );
		}
		*result= 0;
		if( af && !ascanf_SyntaxCheck && !ascanf_arg_error ){
			if( !ascanf_SyntaxCheck ){
			  ascanf_Function *naf;
				if( ActiveWin->current_pen->pen_info ){
					xfree( af->usage );
					af->usage= strdup( ActiveWin->current_pen->pen_info);
					*result= take_ascanf_address( af );
				}
				if( ascanf_arguments> 1 &&
				    (naf= parse_ascanf_address( args[1], _ascanf_variable, "ascanf_PenInfo", (int) ascanf_verbose, &take_usage ))
				    && naf->usage
				){
					xfree( ActiveWin->current_pen->pen_info );
					ActiveWin->current_pen->pen_info= strdup( naf->usage );
				}
			}
			else{
				/* When compiling/ checking syntax, we *must* return a safe pointer of the correct type.
				\ Otherwise, printf[] might complain. NB: this means that the SyntaxCheck field must be
				\ set in this function's entry in the function table!
				*/
				*result= take_ascanf_address( af );
			}
		}
	}
	return( !ascanf_arg_error );
}

int ascanf_PenCurrentPos ( ASCB_ARGLIST )
{ ASCB_FRAME_RESULT
  ascanf_Function *af;
  static double array[3];
  static ascanf_Function AF= {NULL};
  static char *AFname= "CurrentPenPos-Static-Array";
	af= &AF;
	if( af->name ){
	  double oa= af->own_address;
		memset( af, 0, sizeof(ascanf_Function) );
		af->own_address= oa;
	}
	else{
		af->usage= NULL;
		af->type= _ascanf_array;
		af->N= sizeof(array)/sizeof(double);
		af->array= array;
		af->is_address= af->take_address= True;
		af->is_usage= af->take_usage= False;
		af->internal= True;
		af->name= AFname;
		take_ascanf_address(af);
	}
	af->name= AFname;
	af->type= _ascanf_array;
	af->N= sizeof(array)/sizeof(double);
	af->array= array;
	af->is_address= af->take_address= True;
	af->is_usage= af->take_usage= False;
	af->internal= True;

	*result= af->own_address;
	if( !ascanf_SyntaxCheck && ActiveWin && ActiveWin!= &StubWindow ){
		if( ActiveWin->current_pen ){
		  XGPenPosition *cpos;
			if( ActiveWin->current_pen->position && ActiveWin->current_pen->current_pos ){
				cpos= &(ActiveWin->current_pen->position[ActiveWin->current_pen->current_pos-1]);
				array[0]= cpos->x;
				array[1]= cpos->y;
				array[2]= cpos->orn;
				if( pragma_unlikely(ascanf_verbose) ){
					fprintf( StdErr, " (current pen %d, position %d {%s,%s,%s})== ",
						ActiveWin->current_pen->pen_nr, ActiveWin->current_pen->current_pos-1,
						ad2str( array[0], d3str_format, 0),
						ad2str( array[1], d3str_format, 0),
						ad2str( array[2], d3str_format, 0)
					);
				}
			}
			else{
				ascanf_emsg= " (no current position) ";
				*result= 0;
			}
		}
		else{
			ascanf_emsg= " (no current pen) ";
			ascanf_arg_error= True;
			*result= 0;
		}
	}
	return(1);
}

int __xgPenMoveTo( int argc, double arg0, double arg1, double arg2, double *result, ASCB_ARGLIST )
{
	if( !ascanf_SyntaxCheck && ActiveWin && ActiveWin!= &StubWindow ){
		if( ActiveWin->current_pen ){
		  XGPenPosition pos;
			pos.x= arg0;
			pos.y= arg1;
			pos.orn= (ascanf_arguments> 2)? arg2 : 0;
			pos.command= XGPenMoveTo;
			AddPenPosition( ActiveWin->current_pen, &pos, XGP_position );
#if defined(DEBUG) && defined(ASCANF_ALTERNATE)
			if( __ascb_frame ){
				PenPosFrame( ActiveWin->current_pen, ActiveWin->current_pen->current_pos- 1, __ascb_frame );
			}
#endif
			if( result ){
				*result= ActiveWin->current_pen->current_pos;
			}
		}
		else{
			if( result ){
				*result= -1;
			}
			ascanf_emsg= " (no current pen) ";
			ascanf_arg_error= True;
		}
	}
	return(1);
}

int xgPenMoveTo( int argc, double arg0, double arg1, double arg2, double *result )
{
	return __xgPenMoveTo( argc, arg0, arg1, arg2, result, NULL );
}

int ascanf_PenMoveTo ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
	if( !args || ascanf_arguments< 2 ){
		ascanf_arg_error= True;
		return(1);
	}
	return __xgPenMoveTo( ascanf_arguments, args[0], args[1], args[2], result, __ascb_frame );
}

int ascanf_PenLift ( ASCB_ARGLIST )
{ ASCB_FRAME_RESULT
  double nan;
	set_NaN(nan);
	return __xgPenMoveTo( 2, nan, nan, 0, result, __ascb_frame );
}

static int __xgPenLineTo( int argc, int checkArrays, double arg0, double arg1, double arg2, double *result, ASCB_ARGLIST )
{
	if( !ascanf_SyntaxCheck && ActiveWin && ActiveWin!= &StubWindow ){
		if( ActiveWin->current_pen ){
		  XGPenPosition pos, ppos, *cpos;
		  ascanf_Function *xpos= NULL, *ypos= NULL, sxpos, sypos;
		  int i, N= 1;
			if( checkArrays && (xpos= parse_ascanf_address(arg0, _ascanf_array, "ascanf_PenLineTo", False, NULL)) ){
				if( (ypos= parse_ascanf_address(arg1, _ascanf_array, "ascanf_PenLineTo", False, NULL)) ){
					N= MIN(xpos->N, ypos->N);
				}
			}
			if( !xpos && !ypos && arg0 >= 0 && arg0 <= setNumber && argc>= 3 ){
			  DataSet *this_set= &AllSets[ (int) arg0 ];
				if( arg1>= 0 && arg1< this_set->ncols &&
					arg2>= 0 && arg2< this_set->ncols
				){
					N= this_set->numPoints;
					sxpos.array= this_set->columns[(int)arg1];
					sypos.array= this_set->columns[(int)arg2];
					sxpos.iarray= NULL;
					sypos.iarray= NULL;
					xpos= &sxpos;
					ypos= &sypos;
				}
			}
			if( N > 1 ){
				CheckExpand_PenPositions( ActiveWin->current_pen, N );
			}
			// 20101029: initialise ppos with the pen's previous position.
			// no worky yet?!
			if( ActiveWin->current_pen->position && ActiveWin->current_pen->current_pos ){
				cpos = &(ActiveWin->current_pen->position[ActiveWin->current_pen->current_pos-1]);
				if( cpos->command == XGPenMoveTo || cpos->command == XGPenLineTo ){
					ppos.x= cpos->x;
					ppos.y= cpos->y;
#if DEBUG > 1
							fprintf( StdErr, "PenLineTo: continuing from (%s,%s)\n",
								d2str( ppos.x, 0,0), d2str( ppos.y, 0,0)
							);
#endif
				}
				else{
					cpos = NULL;
				}
			}
			else{
				cpos = NULL;
			}
			for( i= 0; i< N; i++ ){
				pos.x= (xpos)? ((xpos->iarray)? xpos->iarray[i] : xpos->array[i]) : arg0;
				pos.y= (ypos)? ((ypos->iarray)? ypos->iarray[i] : ypos->array[i]) : arg1;
				set_NaN(pos.orn);
				if( ( (i || cpos) && (NaN(ppos.x) || NaN(ppos.y)))
					|| (i== 0 && xpos==&sxpos)
				){
					// pen needs to be lifted and moved to the current co-ordinates.
					pos.command= XGPenMoveTo;
				}
				else{
					pos.command= XGPenLineTo;
				}
				AddPenPosition( ActiveWin->current_pen, &pos, XGP_position );
#if defined(DEBUG) && defined(ASCANF_ALTERNATE)
				if( __ascb_frame ){
					PenPosFrame( ActiveWin->current_pen, ActiveWin->current_pen->current_pos- 1, __ascb_frame );
				}
#endif
				ppos.x= pos.x;
				ppos.y= pos.y;
			}
			if( result ){
				*result= ActiveWin->current_pen->current_pos;
			}
		}
		else{
			if( result ){
				*result= -1;
			}
			ascanf_emsg= " (no current pen) ";
			ascanf_arg_error= True;
		}
	}
	return(1);
}

int xgPenLineTo( int argc, int checkArrays, double arg0, double arg1, double arg2, double *result )
{
	return __xgPenLineTo( argc, checkArrays, arg0, arg1, arg2, result, NULL );
}

int ascanf_PenLineTo ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
	if( !args || ascanf_arguments< 2 ){
		ascanf_arg_error= True;
		return(1);
	}
	return __xgPenLineTo( ascanf_arguments, 1, args[0], args[1], args[2], result, __ascb_frame );
}

int ascanf_PenEgoMoveTo ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
	if( !args || ascanf_arguments< 2 ){
		ascanf_arg_error= True;
		return(1);
	}
	if( ascanf_arguments== 2 || args[2]== 0.0 ){
		args[2]= M_2PI;
	}
	if( !ascanf_SyntaxCheck && ActiveWin && ActiveWin!= &StubWindow ){
		if( ActiveWin->current_pen ){
		  XGPenPosition pos, *cpos;
			if( ActiveWin->current_pen->position && ActiveWin->current_pen->current_pos ){
				cpos= &(ActiveWin->current_pen->position[ActiveWin->current_pen->current_pos-1]);
				pos.orn= (NaN(cpos->orn))? args[1] : args[1]+ cpos->orn;
			}
			else{
				cpos= NULL;
				pos.orn= args[1];
			}
			pos.x= args[0]* cos( M_2PI * pos.orn/ args[2] );
			pos.y= args[0]* sin( M_2PI * pos.orn/ args[2] );
			if( cpos ){
				pos.x+= cpos->x;
				pos.y+= cpos->y;
			}
			pos.command= XGPenMoveTo;
			AddPenPosition( ActiveWin->current_pen, &pos, XGP_position );
#if defined(DEBUG) && defined(ASCANF_ALTERNATE)
			PenPosFrame( ActiveWin->current_pen, ActiveWin->current_pen->current_pos- 1, __ascb_frame );
#endif
			*result= ActiveWin->current_pen->current_pos;
		}
		else{
			*result= -1;
			ascanf_emsg= " (no current pen) ";
			ascanf_arg_error= True;
		}
	}
	return(1);
}

int ascanf_PenEgoLineTo ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
	if( !args || ascanf_arguments< 2 ){
		ascanf_arg_error= True;
		return(1);
	}
	if( ascanf_arguments== 2 || args[2]== 0.0 ){
		args[2]= M_2PI;
	}
	if( !ascanf_SyntaxCheck && ActiveWin && ActiveWin!= &StubWindow ){
		if( ActiveWin->current_pen ){
		  XGPenPosition pos, *cpos;
		  double len= 0, l, ang= 0;
		  ascanf_Function *length, *angle;
		  int i= 0, N= 1, cum_length= False, world_angle= False;
			if( (length= parse_ascanf_address(args[0], _ascanf_array, "ascanf_PenEgoLineTo", False, NULL)) ){
				if( (angle= parse_ascanf_address(args[1], _ascanf_array, "ascanf_PenEgoLineTo", False, NULL)) ){
					N= MIN(length->N, angle->N);
				}
				else{
					ang= args[1];
				}
			}
			else{
				len= args[0];
			}
			if( ascanf_arguments> 3 ){
				cum_length= ASCANF_TRUE(args[3]);
			}
			if( ascanf_arguments> 4 ){
				world_angle= ASCANF_TRUE(args[4]);
			}
			if( ActiveWin->current_pen->position && ActiveWin->current_pen->current_pos ){
				cpos= &(ActiveWin->current_pen->position[ActiveWin->current_pen->current_pos-1]);
				if( !NaN(cpos->orn) ){
					pos.orn= cpos->orn;
				}
				pos.x= cpos->x;
				pos.y= cpos->y;
			}
			else{
				cpos= NULL;
				pos.orn= pos.x= pos.y= 0;
			}
			l= ASCANF_ARRAY_ELEM(length,0);
			if( N > 1 ){
				CheckExpand_PenPositions( ActiveWin->current_pen, N );
			}
			do{
				if( length ){
					if( cum_length ){
						len= ASCANF_ARRAY_ELEM(length,i) - l;
						l= ASCANF_ARRAY_ELEM(length,i);
					}
					else{
						len= ASCANF_ARRAY_ELEM(length,i);
					}
				}
				if( angle ){
				  double a= ASCANF_ARRAY_ELEM(angle, i);
					if( !NaN(a) ){
						ang= a;
					}
				}
				if( !NaN(len) ){
					if( world_angle ){
						pos.orn= ang;
					}
					else{
						pos.orn+= ang;
					}
					pos.x+= len* cos( M_2PI * pos.orn/ args[2] );
					pos.y+= len* sin( M_2PI * pos.orn/ args[2] );
					pos.command= XGPenLineTo;
					AddPenPosition( ActiveWin->current_pen, &pos, XGP_position );
#if defined(DEBUG) && defined(ASCANF_ALTERNATE)
					PenPosFrame( ActiveWin->current_pen, ActiveWin->current_pen->current_pos- 1, __ascb_frame );
#endif
				}
				i+= 1;
			} while( i< N );
			*result= ActiveWin->current_pen->current_pos;
		}
		else{
			*result= -1;
			ascanf_emsg= " (no current pen) ";
			ascanf_arg_error= True;
		}
	}
	return(1);
}

/* Quick and dirty lowlevel routine that adds a rectangle based on an argument array of
 \ doubles matching the definition of the user-callable PenRectangle[] routine. No
 \ checking is done.
 */
static void AddPenRectangle( double *args )
{ XGPenPosition pos;
	pos.x= args[0];
	pos.y= args[1];
	pos.w= args[2];
	pos.h= args[3];
	if( args[4] ){
		pos.command= (args[5])? XGPenFillCRect : XGPenDrawCRect;
	}
	else{
		pos.command= (args[5])? XGPenFillRect : XGPenDrawRect;
	}
	if( args[5] ){
		pos.not_outlined= (args[5]> 0)? False : True;
	}
	AddPenPosition( ActiveWin->current_pen, &pos, XGP_position );
}

int ascanf_PenRectangle ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
	if( !args || ascanf_arguments< 6 ){
		ascanf_arg_error= True;
		return(1);
	}
	if( !ascanf_SyntaxCheck && ActiveWin && ActiveWin!= &StubWindow ){
		if( ActiveWin->current_pen ){
			AddPenRectangle( args );
#if defined(DEBUG) && defined(ASCANF_ALTERNATE)
			PenPosFrame( ActiveWin->current_pen, ActiveWin->current_pen->current_pos- 1, __ascb_frame );
#endif
			*result= ActiveWin->current_pen->current_pos;
		}
		else{
			*result= -1;
			ascanf_emsg= " (no current pen) ";
			ascanf_arg_error= True;
		}
	}
	return(1);
}

int ascanf_PenEllipse ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
	if( !args || ascanf_arguments< 3 ){
		ascanf_arg_error= True;
		return(1);
	}
	if( !ascanf_SyntaxCheck && ActiveWin && ActiveWin!= &StubWindow ){
		if( ActiveWin->current_pen ){
		  XGPenPosition pos;
			pos.x= args[0];
			pos.y= args[1];
			pos.w= args[2];
			if( ascanf_arguments> 3 ){
				pos.h= args[3];
			}
			else{
				pos.h= args[2];
			}
			pos.command= XGPenDrawEllps;
			AddPenPosition( ActiveWin->current_pen, &pos, XGP_position );
#if defined(DEBUG) && defined(ASCANF_ALTERNATE)
			PenPosFrame( ActiveWin->current_pen, ActiveWin->current_pen->current_pos- 1, __ascb_frame );
#endif
			*result= ActiveWin->current_pen->current_pos;
		}
		else{
			*result= -1;
			ascanf_emsg= " (no current pen) ";
			ascanf_arg_error= True;
		}
	}
	return(1);
}

int ascanf_PenPolygon ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  int N;
	if( !args || ascanf_arguments< 2 ){
		ascanf_arg_error= True;
		return(1);
	}
	PenPolyX= parse_ascanf_address(args[0], _ascanf_array, "ascanf_PenPolygon", False, NULL);
	PenPolyY= parse_ascanf_address(args[1], _ascanf_array, "ascanf_PenPolygon", False, NULL);
	if( !PenPolyX || !PenPolyY || !PenPolyX->N || !PenPolyY->N ){
		ascanf_arg_error= True;
		ascanf_emsg= " (illegal argument for one or both of the X and Y arrays) ";
		return(1);
	}
	N= MIN( PenPolyX->N, PenPolyY->N );
	if( !ascanf_SyntaxCheck && ActiveWin && ActiveWin!= &StubWindow ){
		if( ActiveWin->current_pen ){
		  XGPenPosition pos;
			pos.polyN_1= N- 1;
			pos.polyX= NULL;
			pos.polyY= NULL;
			if( ascanf_arguments> 2 ){
				pos.command= (args[2])? XGPenFillPoly : XGPenDrawPoly;
			}
			if( args[2] ){
				pos.not_outlined= (args[2]> 0)? False : True;
			}
			AddPenPosition( ActiveWin->current_pen, &pos, XGP_position );
#if defined(DEBUG) && defined(ASCANF_ALTERNATE)
			PenPosFrame( ActiveWin->current_pen, ActiveWin->current_pen->current_pos- 1, __ascb_frame );
#endif
			*result= ActiveWin->current_pen->current_pos;
		}
		else{
			*result= -1;
			ascanf_emsg= " (no current pen) ";
			ascanf_arg_error= True;
		}
	}
	return(1);
}

int ascanf_PenLineStyleWidth ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
	if( !args || ascanf_arguments< 2 ){
		ascanf_arg_error= True;
		return(1);
	}
	if( !ascanf_SyntaxCheck && ActiveWin && ActiveWin!= &StubWindow ){
		if( ActiveWin->current_pen ){
		  XGPenPosition pos;
			CLIP_EXPR_CAST( short, pos.attr.linestyle, double, args[0], -MAXSHORT, MAXSHORT );
			pos.attr.lineWidth= args[1];
			*result= AddPenPosition( ActiveWin->current_pen, &pos, XGP_linestyle|XGP_lineWidth );
		}
		else{
			*result= -1;
			ascanf_emsg= " (no current pen) ";
			ascanf_arg_error= True;
		}
	}
	return(1);
}

int ascanf_PenClipping ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
	if( !ascanf_SyntaxCheck && ActiveWin && ActiveWin!= &StubWindow ){
		if( ActiveWin->current_pen ){
		  XGPenPosition pos;
		  short clipping;
			if( args && ascanf_arguments ){
				clipping= ASCANF_TRUE(args[0]);
			}
			else{
				clipping= True;
			}
			pos.attr.noclip= !clipping;
			*result= AddPenPosition( ActiveWin->current_pen, &pos, XGP_noClip );
		}
		else{
			*result= -1;
			ascanf_emsg= " (no current pen) ";
			ascanf_arg_error= True;
		}
	}
	return(1);
}

int ascanf_PenMark ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
	if( ascanf_arguments== 1 ){
		ascanf_emsg= " (pass either 0 or 2 arguments) ";
		ascanf_arg_error= True;
		return(1);
	}
	if( !ascanf_SyntaxCheck && ActiveWin && ActiveWin!= &StubWindow ){
		if( ActiveWin->current_pen ){
		  XGPenPosition pos;
			if( ascanf_arguments>= 2 ){
				CLIP_EXPR_CAST( short, pos.attr.markType, double, args[0], 1, MAXSHORT );
				pos.attr.markSize= args[1];
				pos.attr.markFlag= True;
				*result= AddPenPosition( ActiveWin->current_pen, &pos, XGP_markFlag|XGP_markType|XGP_markSize );
			}
			else{
				pos.attr.markFlag= False;
				*result= AddPenPosition( ActiveWin->current_pen, &pos, XGP_markFlag );
			}
		}
		else{
			*result= -1;
			ascanf_emsg= " (no current pen) ";
			ascanf_arg_error= True;
		}
	}
	return(1);
}

ascanf_Function *ascanf_ParseColour( double args, char **name_ret, int strict, char *caller )
{ ascanf_Function *af;
  int take_usage;
  char *c= NULL;
  static char buf[256];
	if( (af= parse_ascanf_address(args, 0, caller, False, &take_usage)) ){
		if( af->type== _ascanf_array && af->N== 3 ){
		  int i;
			if( af->iarray ){
				for( i= 0; i< 3; i++ ){
					CLIP( af->iarray[i], 0, 255 );
				}
				if( *AllowGammaCorrection ){
					sprintf( buf, "rgbi:%g/%g/%g", af->iarray[0]/ 255.0, af->iarray[1]/ 255.0, af->iarray[2]/ 255.0 );
				}
				else{
					sprintf( buf, "rgb:%04x/%04x/%04x",
						(int) (af->iarray[0]/ 255.0* 65535.0+ 0.5),
						(int) (af->iarray[1]/ 255.0* 65535.0+ 0.5),
						(int) (af->iarray[2]/ 255.0* 65535.0+ 0.5)
					);
				}
			}
			else{
				for( i= 0; i< 3; i++ ){
					CLIP( af->array[i], 0, 1 );
				}
				if( *AllowGammaCorrection ){
					sprintf( buf, "rgbi:%g/%g/%g", af->array[0], af->array[1], af->array[2] );
				}
				else{
					sprintf( buf, "rgb:%04x/%04x/%04x",
						(int) (af->array[0]* 65535+ 0.5),
						(int) (af->array[1]* 65535+ 0.5),
						(int) (af->array[2]* 65535+ 0.5)
					);
				}
			}
			StringCheck( buf, sizeof(buf)/sizeof(char), __FILE__, __LINE__ );
			c= buf;
		}
		else if( af->usage && (!strict || take_usage) ){
			c= af->usage;
		}
		else{
			ascanf_emsg= " (illegal or void colourname stringpointer, and not a 3-element array neither) ";
			ascanf_arg_error= True;
		}
		if( name_ret ){
			*name_ret= c;
		}
		if( pragma_unlikely(ascanf_verbose) && c ){
			fprintf( StdErr, " (found colour \"%s\") ", c );
			fflush( StdErr );
		}
	}
	return( af );
}

int ascanf_PenColour ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  ascanf_Function *af;
  char *cname= NULL;
	if( !args || ascanf_arguments< 1 ){
	  XGPenPosition *pos;
	  int i, positions;
		set_NaN(*result);
		if( !ascanf_SyntaxCheck && ActiveWin && ActiveWin!= &StubWindow && ActiveWin->current_pen ){
			pos = ActiveWin->current_pen->position;
			positions = ActiveWin->current_pen->current_pos;
			if( pos ){
				for( i = positions-1 ; i >= 0 ; i-- ){
					if( CheckMask(pos[i].operation, XGP_colour) ){
						if( pos[i].pixelCName ){
						  static ascanf_Function AF= {NULL};
						  static char *AFname= "PenColour-Static-StringPointer";
							init_Static_StringPointer( &AF, AFname );
							xfree( AF.usage );
							AF.usage= strdup( pos[i].pixelCName );
							*result= take_ascanf_address( &AF );
						}
						else if( pos[i].colour.pixvalue >= 0 ){
							*result = pos[i].colour.pixvalue;
						}
						return(1);
					}
				}
			}
		}
		return(0);
	}
	  /* Officially, we should expect and only accept a stringpointer as colourname. However, since
	   \ a pointer can be stored in a variable, or returned by a function, and a string pointer can not
	   \ yet, we also accept regular pointers. After all, we can get at its "string" value, and there is
	   \ no other relevant use for them.
	   */
	if( (af= ascanf_ParseColour( args[0], &cname, False, "ascanf_PenColour")) && !cname ){
		ascanf_emsg = " (invalid colourname argument) ";
		return(1);
	}
	if( !ascanf_SyntaxCheck && ActiveWin && ActiveWin!= &StubWindow ){
		if( ActiveWin->current_pen ){
		  XGPenPosition pos;
			if( cname ){
				pos.pixelCName= cname;
				pos.colour.pixvalue= -1;
			}
			else{
				CLIP_EXPR_CAST( int, pos.colour.pixvalue, double, args[0], 0, MAXINT );
			}
			if( (*result= AddPenPosition( ActiveWin->current_pen, &pos, XGP_colour ))== 1 ){
				if( af && af->type== _ascanf_array && af->N== 3 ){
					StoreCName( af->usage );
				}
			}
			XGStoreColours = 1;
		}
		else{
			*result= -1;
			ascanf_emsg= " (no current pen) ";
			ascanf_arg_error= True;
		}
	}
	return(1);
}

int ascanf_PenFillColour ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  ascanf_Function *af;
  char *cname= NULL;
	if( !args || ascanf_arguments< 1 ){
	  XGPenPosition *pos;
	  int i, positions;
		set_NaN(*result);
		if( !ascanf_SyntaxCheck && ActiveWin && ActiveWin!= &StubWindow && ActiveWin->current_pen ){
			pos = ActiveWin->current_pen->position;
			positions = ActiveWin->current_pen->current_pos;
			if( pos ){
				for( i = positions-1 ; i >= 0 ; i-- ){
					if( CheckMask(pos[i].operation, XGP_flcolour) ){
						if( pos[i].flpixelCName ){
						  static ascanf_Function AF= {NULL};
						  static char *AFname= "PenColour-Static-StringPointer";
							init_Static_StringPointer( &AF, AFname );
							xfree( AF.usage );
							AF.usage= strdup( pos[i].flpixelCName );
							*result= take_ascanf_address( &AF );
						}
						else if( pos[i].flcolour.pixvalue >= 0 ){
							*result = pos[i].flcolour.pixvalue;
						}
						return(1);
					}
				}
			}
		}
		return(0);
	}
	  /* Officially, we should expect and only accept a stringpointer as colourname. However, since
	   \ a pointer can be stored in a variable, or returned by a function, and a string pointer can not
	   \ yet, we also accept regular pointers. After all, we can get at its "string" value, and there is
	   \ no other relevant use for them.
	   */
	if( (af= ascanf_ParseColour( args[0], &cname, False, "ascanf_PenFillColour")) && !cname ){
		return(1);
	}
	if( !ascanf_SyntaxCheck && ActiveWin && ActiveWin!= &StubWindow ){
		if( ActiveWin->current_pen ){
		  XGPenPosition pos;
			if( cname ){
				pos.flpixelCName= cname;
				pos.flcolour.pixvalue= -1;
			}
			else{
				CLIP_EXPR_CAST( int, pos.flcolour.pixvalue, double, args[0], -MAXINT, MAXINT );
			}
			if( (*result= AddPenPosition( ActiveWin->current_pen, &pos, XGP_flcolour ))== 1 ){
				if( af && af->type== _ascanf_array && af->N== 3 ){
					StoreCName( af->usage );
				}
			}
			XGStoreColours = 1;
		}
		else{
			*result= -1;
			ascanf_emsg= " (no current pen) ";
			ascanf_arg_error= True;
		}
	}
	return(1);
}

/*  This bit of code should one day evolve into a CustomFont support for XGPens...
    20020317: which is implemented for some time yet?!
			if( ok && xfn && psfn && pspt ){
			  extern CustomFont *Init_CustomFont();
				if( !(cf= Init_CustomFont( xfn, axfn, psfn, pssize, psreencode )) ){
					fprintf( StdErr, "*ValCat_%c_Font*: can't initialise CustomFont..\n", an );
				}
				xfree( xfn);
				xfree( axfn);
				xfree( psfn );
			}
 */

void PenTextDimensions( LocalWin *wi, XGPenPosition *pos )
{ double maxFontWidth;
  double dx, dy, lh;
  XCharStruct bb;
  double fontScale;
  ascanf_Function *cfv= pos->cfont_var;
  int w, h, theight;
	if( pos->textFntNr<0 ){
		maxFontWidth= XGFontWidth_Lines( wi, pos->textFntNr, pos->text, '\n', &w, &theight, &h,
			(cfv)? cfv->cfont : NULL, &bb, &fontScale
		);
	}
	else{
		maxFontWidth= XGFontWidth_Lines( wi, pos->textFntNr, pos->text, '\n', &w, &theight, &h,
			NULL, &bb, &fontScale
		);
	}
	pos->textFntHeight= pos->TextBox->textFntHeight= h;
	if( theight> h ){
	  char *tbuf= XGstrdup(pos->text), *line, *c;
	  double lw;
		c= tbuf;
		lh= 0;
		while( c && xtb_getline( &c, &line) ){
			lw= (double) XGTextWidth( wi, line, pos->textFntNr, (pos->textFntNr< 0 && cfv)? cfv->cfont : NULL );
			lh= MAX( lh, lw );
		}
		h= theight;
		xfree( tbuf );
	}
	else{
		lh= (double) XGTextWidth( wi, pos->text, pos->textFntNr, (pos->textFntNr< 0 && cfv)? cfv->cfont : NULL );
	}
	if( pos->textJust== T_VERTICAL ){
		dx= h;
		dy= lh;
		if( PS_STATE(wi)->Printing== PS_PRINTING && wi->textrel.used_gsTextWidth ){
		  /* nothing	*/
		}
		else{
			dy*= w / maxFontWidth;
		}
	}
	else{
		dx= lh;
		if( PS_STATE(wi)->Printing== PS_PRINTING && wi->textrel.used_gsTextWidth ){
		  /* nothing	*/
		}
		else{
			dx*= w / maxFontWidth;
		}
		dy= h;
	}
	pos->TextBox->w= dx;
	pos->TextBox->h= dy;
	dx*= wi->XUnitsPerPixel/ wi->Xscale;
	dy*= wi->YUnitsPerPixel/ wi->Yscale;
	pos->w= dx;
	pos->h= dy;
}

static int _ascanf_PenText ( ASCB_ARGLIST , int boxed, char *caller )
{ ASCB_FRAME_SHORT
  ascanf_Function *taf, *dim= NULL, *cfont_var= NULL;
  int argc, take_usage= False, justX= 0, justY= 0, just, textFntNr= 0;
  char *text= NULL, *vtext= NULL;
  double X, Y, x, y, boxfill;
	if( !args || ascanf_arguments< 3 ){
		ascanf_arg_error= True;
		return(1);
	}
	X= args[0];
	Y= args[1];
	  /* Officially, we should expect and only accept a stringpointer as colourname. However, since
	   \ a pointer can be stored in a variable, or returned by a function, and a string pointer can not
	   \ yet, we also accept regular pointers. After all, we can get at its "string" value, and there is
	   \ no other relevant use for them.
	   */
	if( (taf= parse_ascanf_address(args[2], 0, caller, False, &take_usage)) ){
		if( take_usage ){
			if( taf->usage ){
				text= taf->usage;
			}
			else{
				ascanf_emsg= " (NULL stringpointer) ";
/* 				ascanf_arg_error= True;	*/
				return(1);
			}
		}
		else if( taf->type!= _ascanf_array ){
			ascanf_emsg= " (pointer must be to string or array) ";
			ascanf_arg_error= True;
			return(1);
		}
		  /* need one argument less.	*/
		argc= -1;
	}
	else{
		x= args[2];
		argc= 0;
	}
	if( ascanf_arguments> 3+argc ){
		y= args[3+argc];
		if( ascanf_arguments> 4+argc ){
			CLIP_EXPR_CAST( int, justX, double, args[4+argc], -2, 1 );
		}
		if( ascanf_arguments> 5+argc ){
			CLIP_EXPR_CAST( int, justY, double, args[5+argc], -1, 1 );
		}
		switch( justX ){
			case -2:
				just= T_VERTICAL;
				break;
			case -1:
				just= 2 << 2;
				break;
			case 0:
				just= 0;
				break;
			case 1:
				just= 1 << 2;
				break;
		}
		if( just!= T_VERTICAL ){
			switch( justY ){
				case -2:
					just= T_VERTICAL;
					break;
				case -1:
					just|= 2;
					break;
				case 0:
					just|= 0;
					break;
				case 1:
					just|= 1;
					break;
			}
			switch( just ){
				case 0:
					just= T_CENTER;
					break;
				case 4:
					just= T_LEFT;
					break;
				case 5:
					just= T_UPPERLEFT;
					break;
				case 1:
					just= T_TOP;
					break;
				case 9:
					just= T_UPPERRIGHT;
					break;
				case 8:
					just= T_RIGHT;
					break;
				case 10:
					just= T_LOWERRIGHT;
					break;
				case 2:
					just= T_BOTTOM;
					break;
				case 6:
					just= T_LOWERLEFT;
					break;
			}
		}
		if( ascanf_arguments> 6+argc ){
			dim= parse_ascanf_address(args[6+argc], 0, caller, False, NULL);
			if( !dim || (dim->type== _ascanf_variable && dim->cfont) ){
				if( !dim ){
					CLIP_EXPR_CAST( int, textFntNr, double, args[6+argc], 0, T_LEGEND );
				}
				else{
					textFntNr= -1;
					cfont_var= dim;
					dim= NULL;
				}
				argc+= 1;
				if( ascanf_arguments> 6+argc ){
					if( !(dim= parse_ascanf_address(args[6+argc], _ascanf_array, caller, False, NULL)) ){
						  /* this was not a dimension pointer, rewind the argument counter by 1: */
						argc-= 1;
					}
				}
			}
			else if( dim && dim->type!= _ascanf_array ){
				ascanf_emsg= " (dimension return variable must be an array of doubles) ";
				ascanf_arg_error= 1;
				dim= NULL;
			}
			if( dim && (dim->iarray || dim->name[0]== '%') ){
				ascanf_emsg= " (dimension return array must be of type double) ";
				ascanf_arg_error= 1;
				dim= NULL;
			}
		}
	}
	else{
		set_NaN(y);
	}
	if( boxed ){
		argc+= 1;
		if( ascanf_arguments> 6+ argc ){
			boxfill= args[6+argc];
		}
		else{
			fputs( " (textbox fill argument is missing)== ", StdErr );
			ascanf_emsg= " (textbox fill argument is missing)== ";
			ascanf_arg_error= 1;
			boxed= False;
		}
	}
	if( !ascanf_SyntaxCheck && ActiveWin && ActiveWin!= &StubWindow ){
		if( ActiveWin->current_pen ){
		  XGPenPosition pos, TextBox;
			pos.x= X;
			pos.y= Y;
			pos.command= XGPenText;
			if( text ){
				  /* !! 20030330: pos.text should be a copy since we'll xfree() it elsewhere!!! */
				pos.text= XGstrdup(text);
			}
			else if( taf && taf->type== _ascanf_array ){
			  int i;
			  char xs[127];
				WriteValue( ActiveWin, xs, (taf->iarray)? taf->iarray[0] : taf->array[0], 0, 0, 0, 0, X_axis, 0, 0, 127 );
				vtext= concat2( vtext, "(", xs, NULL);
				for( i= 1; i< taf->N; i++ ){
					WriteValue( ActiveWin, xs, (taf->iarray)? taf->iarray[i] : taf->array[i], 0, 0, 0, 0, Y_axis, 0, 0, 127 );
					vtext= concat2( vtext, ",", xs, NULL);
				}
				vtext= concat2( vtext, ")", NULL );
				pos.text= vtext;
			}
			else{
			  char xs[127], ys[127];
				WriteValue( ActiveWin, xs, x, 0, 0, 0, 0, X_axis, 0, 0, 127 );
				if( !NaN(y) ){
					WriteValue( ActiveWin, ys, y, 0, 0, 0, 0, Y_axis, 0, 0, 127 );
					vtext= concat2( vtext, "(", xs, ",", ys, ")", NULL );
				}
				else{
					vtext= concat2( vtext, xs, NULL );
				}
				pos.text= vtext;
			}
			pos.textJust= just;
			pos.textJust2[0]= justX;
			pos.textJust2[1]= justY;
			pos.textFntNr= textFntNr;
			pos.cfont_var= cfont_var;
			 /* Careful not to copy the template TextBox into the Pen->position array! */
			pos.TextBox= &TextBox;
			PenTextDimensions( ActiveWin, &pos );
			if( dim ){
			  double r= 2;
				Resize_ascanf_Array( dim, 2, &r );
				if( dim->N== 2 ){
					dim->array[0]= pos.w;
					dim->array[1]= pos.h;
				}
			}
			if( boxed ){
#if DEBUG == 2
			  double r[6], xlw, ylw, x_offset, y_offset;
#endif
				pos.command= XGPenTextBox;
				if( boxfill ){
					pos.TextBox->command= XGPenFillRect;
					pos.TextBox->not_outlined= (boxfill> 0)? False : True;
				}
				else{
					pos.TextBox->command= XGPenDrawRect;
				}
#if DEBUG == 2
				if( ActiveWin->current_pen->position && ActiveWin->current_pen->current_pos ){
					xlw= pLINEWIDTH(ActiveWin,ActiveWin->current_pen->position[ActiveWin->current_pen->current_pos-1].attr.lineWidth)*
						ActiveWin->dev_info.var_width_factor;
					ylw= xlw* ActiveWin->YUnitsPerPixel/ ActiveWin->Yscale;
					xlw*= ActiveWin->XUnitsPerPixel/ ActiveWin->Xscale;
				}
				else{
					ylw= xlw= 0;
				}
				r[0]= pos.x;
				r[1]= pos.y;
				r[2]= pos.w+ xlw;
				r[3]= pos.h+ ylw;
				r[4]= 0;
				r[5]= 0*boxfill;
				  /* offset values that give some margins around the string:	*/
				x_offset= ActiveWin->dev_info.bdr_pad/ 2.5* ActiveWin->win_geo._XUnitsPerPixel/ ActiveWin->Xscale;
				y_offset= ActiveWin->dev_info.bdr_pad/ 2.5* ActiveWin->win_geo._YUnitsPerPixel/ ActiveWin->Yscale;
				if( just== T_VERTICAL ){
					  /* Fine-tuning by hand... The vertical printing routine offsets the string by a certain amount,
					   \ that seems to be well estimated by 0.2* width (the used font's height!) horizontally, and
					   \ -1.5* width vertically. I hope these values are universal - and valid until I get a better
					   \ grasp on what the text rotation actually does!
					   \ Note that these shifts are slightly different from the one applied by DrawPen!!!
					   */
					r[0]-= 0.5* xlw- 0.2* pos.w+ x_offset;
					r[2]+= 2* x_offset;
					r[1]-= 0.5* ylw+ 1.5* pos.w+ y_offset;
					r[3]+= 2* y_offset;
				}
				else{
					switch( justX ){
					 /* Position fine-tuning: the string seems to be offset by -0.015*width (=to its left):
					  \ also shift the box over that (additional) amount.
					  */
						case -1:
							r[0]-= 1.015* pos.w+ 0.5* xlw+ x_offset;
							r[2]+= 2* x_offset;
							break;
						case 0:
							r[0]= r[0]- r[2]/2- x_offset;
							r[2]+= 2* x_offset;
							break;
						case 1:
							r[0]-= 0.5* xlw+ x_offset- 0.015* pos.w;
							r[2]+= 2* x_offset;
							break;
					}
					switch( justY ){
					  /* Vertically, the necessary shift seems to be well approximated by 0.05* pos.h	*/
						case -1:
							r[1]-= 1* y_offset+ 0.5* ylw+ 0.05* pos.h;
							r[3]+= 2* y_offset;
							break;
						case 0:
							r[1]= r[1]- r[3]/ 2- 0.75* y_offset;
							r[3]+= 2* y_offset;
							break;
						case 1:
							r[1]-= (1-0.05)* pos.h+ 0.5* ylw+ y_offset;
							r[3]+= 2* y_offset;
							break;
					}
				}
				AddPenRectangle( r );
#endif
			}
			else{
				pos.TextBox->command= 0;
			}
			*result= AddPenPosition( ActiveWin->current_pen, &pos, XGP_position|XGP_text|XGP_textJust|XGP_textFntNr );
#if defined(DEBUG) && defined(ASCANF_ALTERNATE)
			PenPosFrame( ActiveWin->current_pen, ActiveWin->current_pen->current_pos- 1, __ascb_frame );
#endif
			xfree( vtext );
		}
		else{
			*result= -1;
			ascanf_emsg= " (no current pen) ";
			ascanf_arg_error= True;
		}
	}
	return(1);
}

int ascanf_PenText ( ASCB_ARGLIST )
{
	return( _ascanf_PenText( ASCB_ARGUMENTS, 0, "ascanf_PenText" ) );
}

int ascanf_PenTextBox ( ASCB_ARGLIST )
{
	return( _ascanf_PenText( ASCB_ARGUMENTS, 1, "ascanf_PenTextBox" ) );
}

int ascanf_PenTextOutside ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
	if( !ascanf_SyntaxCheck && ActiveWin && ActiveWin!= &StubWindow ){
		if( ActiveWin->current_pen ){
		  XGPenPosition pos;
			if( ascanf_arguments>= 1 ){
				CLIP_EXPR_CAST( short, pos.attr.text_outside, double, args[0], -MAXSHORT, MAXSHORT );
				*result= AddPenPosition( ActiveWin->current_pen, &pos, XGP_textOutside );
			}
			else{
				pos.attr.text_outside= False;
				*result= AddPenPosition( ActiveWin->current_pen, &pos, XGP_textOutside );
			}
		}
		else{
			*result= -1;
			ascanf_emsg= " (no current pen) ";
			ascanf_arg_error= True;
		}
	}
	return(1);
}

int ascanf_PensShown ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  extern int no_pens;

	if( ActiveWin ){
		*result= ! ActiveWin->no_pens;
		if( ascanf_arguments ){
			ActiveWin->no_pens = ASCANF_FALSE(args[0]);
		}
	}
	else{
		*result= !no_pens;
		if( ascanf_arguments ){
			no_pens = ASCANF_FALSE(args[0]);
		}
	}
	return(1);
}

static int eval_now( int n, char *expr )
{ double *values;
  int ret = 0;
	if( !(values = (double*) calloc( n, sizeof(double) )) ){
		fprintf( StdErr, "eval_now() cannot allocation %d doubles (%s)\n", n, serror() );
	}
	else{
		if( debugFlag ){
			fprintf( StdErr, "eval_now(%d, \"%s\")\n", n, expr );
		}
		ret = fascanf_eval( &n, expr, values, NULL, NULL, False );
		xfree(values);
	}
	return ret;
}

static int __Dump_XGPens( LocalWin *source, LocalWin *target, FILE *fp, char *buf, size_t buflen, int DumpPositions, char *opcode )
{ int i=0, p=0;
  extern int XG_Stripped;
  int dumped = 0;
	if( debugFlag ){
		fprintf( StdErr, "__Dump_XGPens(source=%p, target=%p, fp=%p, buf=%p, buflen=%lu, DumpPositions=%d, opcode=\"%s\")\n",
			   source, target, fp, buf, buflen, DumpPositions, opcode
		);
	}
	if( !opcode ){
		opcode= "*STARTUP_EXPR**EVAL*";
	}
	if( source && source->pen_list ){
	  XGPen *Pen= source->pen_list;
	  XGPenCommand op;
	  LocalWin *AW= ActiveWin;
	  Window asw= ascanf_window;
	  extern int ascanf_check_int;
	  int aci= ascanf_check_int;
	  extern double *Allocate_Internal;
	  double AI= *Allocate_Internal;

		if( target ){
			ActiveWin= target;
			ascanf_window= 0;
			ascanf_check_int= 0;
		}

		while( Pen && (fp || ActiveWin != source) ){
		  XGPenPosition *pos;
		  int positions, nc;
		  char *head;
#define ADVANCE(h,i)	if((i)>0){ (h) += (i); } else { (h) = &h[strlen(h)-1]; }

			if( Pen->current_pos== 0 || (PENSKIP(source,Pen) && XG_Stripped) ){
				if( debugFlag ){
					fprintf( StdErr, "Skipping Pen #%d dump (pos=%d, skip=%d)\n", Pen->pen_nr, Pen->current_pos, PENSKIP(source,Pen) );
				}
				goto dump_next_pen;
			}
			pos = Pen->position;
			positions = (DumpPositions)? Pen->current_pos : 1;
			if( fp ){
				fprintf( fp, "*EXTRATEXT* Dumping Pen %d (using internal dictionnary to avoid namespace pollution)\n\n"
					"*SILENCE* 1\n",
					Pen->pen_nr
				);
				fprintf( fp, "%s $IDict[1],", opcode );
				if( DumpPositions ){
					fprintf( fp, " TBARprint[\"%s Pen #%d\"],", (target)? "Restoring" : "Reading", Pen->pen_nr );
				}
				else{
					fprintf( fp, " TBARprint[\"%s Pen #%d settings\"],", (target)? "Restoring" : "Reading", Pen->pen_nr );
				}
				  /* 20040422: Do a PenReset; otherwise, multiple windows (AllWin) restores won't work... */
				fprintf( fp, " SelectPen[%d], PenReset, PenNumPoints[%d] @\n",
					Pen->pen_nr, Pen->allocSize
				);
				if( Pen->pen_info ){
				  char *tbuf= NULL;
					sprint_string2( &tbuf, "", "", Pen->pen_info, True );
					fprintf( fp, "%s PenInfo[0,\"%s\"] @\n", opcode, tbuf );
					xfree( tbuf );
				}
				fprintf( fp, "%s PenOverwrite[%d], PenBeforeSet[%d], PenAfterSet[%d], PenFloating[%d] @\n", opcode,
					Pen->overwrite_pen, Pen->before_set, Pen->after_set, Pen->floating
				);
				fprintf( fp, "%s PenSkip[%d,%d], PenSetLink[%d], PenHighlightColour[%d", opcode,
					Pen->pen_nr, Pen->skip, Pen->set_link, Pen->highlight
				);
				if( Pen->hlpixelCName ){
					fprintf( fp, ",\"%s\"", Pen->hlpixelCName );
				}
				fprintf( fp, "], PenHighlightText[%d] @\n",
					Pen->highlight_text
				);
			}
			else{
				*Allocate_Internal= 1;

				if( DumpPositions ){
					sprintf( buf, " TBARprint[\"%s Pen #%d\"] @", (target)? "Restoring" : "Dumping", Pen->pen_nr );
				}
				else{
					sprintf( buf, " TBARprint[\"%s Pen #%d settings\"] @", (target)? "Restoring" : "Dumping", Pen->pen_nr );
				}
				StringCheck( buf, buflen, __FILE__, __LINE__ );
				eval_now( 1, buf );
// 				ASCB_call( ascanf_SelectPen, NULL, 0, NULL, 1, 1, (double) Pen->pen_nr );
				ActiveWin->current_pen= CheckPens( ActiveWin, Pen->pen_nr );
//  				ASCB_call( ascanf_PenReset, NULL, 0, NULL, 1, 0, 0.0 );
 				if( ActiveWin->current_pen ){
 					PenReset( ActiveWin->current_pen, False );
 				}
// 				ASCB_call( ascanf_PenNumPoints, NULL, 0, NULL, 1, 1, (double) Pen->allocSize );
				Alloc_PenPositions( ActiveWin->current_pen, Pen->allocSize );
				if( ActiveWin->current_pen ){
/* 					ASCB_call( ascanf_PenOverwrite, NULL, 0, NULL, 1, 1, (double) Pen->overwrite_pen );	*/
/* 					ASCB_call( ascanf_PenBeforeSet, NULL, 0, NULL, 1, 1, (double) Pen->before_set );	*/
/* 					ASCB_call( ascanf_PenAfterSet, NULL, 0, NULL, 1, 1, (double) Pen->after_set );	*/
/* 					ASCB_call( ascanf_PenFloating, NULL, 0, NULL, 1, 1, (double) Pen->floating );	*/
/* 					ASCB_call( ascanf_PenSkip2, NULL, 0, NULL, 2, 2, (double) Pen->pen_nr, (double) Pen->skip );	*/
					  /* Not only is it more efficient to handle the code above as below, the result is not always
					   \ the same (??!!). For the Pen->floating statement above, the test ASCANF_TRUE(args[0]) in
					   \ ascanf_PenFloating sometimes returned true even when args[0] was supposed to be (and printed as)
					   \ 0. On x86 and on an SGI O2, so there is a bug either my code or in the compiler.
					   */
					ActiveWin->current_pen->overwrite_pen= Pen->overwrite_pen;
					ActiveWin->current_pen->before_set= Pen->before_set;
					ActiveWin->current_pen->after_set= Pen->after_set;
					ActiveWin->current_pen->floating= Pen->floating;
					ActiveWin->current_pen->skip= Pen->skip;
					ActiveWin->current_pen->set_link= Pen->set_link;
					xfree(ActiveWin->current_pen->pen_info);
					if( Pen->pen_info ){
						ActiveWin->current_pen->pen_info= strdup(Pen->pen_info);
					}
				}
				head = buf;
				nc = sprintf( head, "PenHighlightColour[%d", Pen->highlight );
				ADVANCE(head,nc);
				if( Pen->hlpixelCName ){
					nc = sprintf( head, ",\"%s\"", Pen->hlpixelCName );
					ADVANCE(head,nc);
				}
				nc = sprintf( head, "], PenHighlightText[%d] @", Pen->highlight_text );
				StringCheck( buf, buflen, __FILE__, __LINE__ );
				eval_now( 2, buf );

				*Allocate_Internal= AI;
			}
			for( i= 0; i< positions; i++ ){
			  int n;
				if( i== 0 ){
					head = buf;
					nc = sprintf( head, "PenLineStyleWidth[%s,%s], PenColour[",
						ad2str( pos[i].attr.linestyle, d3str_format, 0), ad2str( pos[i].attr.lineWidth, d3str_format, 0)
					);
					ADVANCE(head,nc);
					if( pos[i].pixelCName ){
						nc = sprintf( head, "\"%s\"], PenFillColour[", pos[i].pixelCName );
					}
					else{
						nc = sprintf( head, "%s], PenFillColour[", ad2str( pos[i].colour.pixvalue, d3str_format,0) );
					}
					ADVANCE(head,nc);
					if( pos[i].flpixelCName ){
						nc = sprintf( head, "\"%s\"]", pos[i].flpixelCName );
					}
					else{
						nc = sprintf( head, "%s]", ad2str( pos[i].flcolour.pixvalue, d3str_format,0) );
					}
					ADVANCE(head,nc);
					if( pos[i].attr.markFlag ){
						nc = sprintf( head, ", PenMark[%s,%s]",
							ad2str( pos[i].attr.markType, d3str_format,0), ad2str( pos[i].attr.markSize, d3str_format, 0)
						);
						ADVANCE(head,nc);
					}
					else{
						strcat( buf, ", PenMark" );
						ADVANCE(head,9);
					}
					if( pos[i].attr.text_outside ){
						nc = sprintf( head, ", PenTextOutside[%s]",
							ad2str( pos[i].attr.text_outside, d3str_format,0)
						);
						ADVANCE(head,nc);
					}
					else{
						strcat( head, ", PenTextOutside[0]" );
						ADVANCE(head,19);
					}
					if( pos[i].attr.noclip ){
						nc = sprintf( head, ", PenClipping[%d]", !pos[i].attr.noclip );
					}
					else{
						strcat( head, ", PenClipping" );
						nc = 13;
					}
					strcat( buf, " @" );
					if( fp ){
						fprintf( fp, "%s %s\n", opcode, buf );
					}
					else{
						StringCheck( buf, buflen, __FILE__, __LINE__ );
						eval_now( 6, buf );
					}
				}
				else{
				  int attrchange= False, colchange= False, flcolchange= False;
					if( pos[i].attr.linestyle!= pos[i-1].attr.linestyle || pos[i].attr.lineWidth!= pos[i-1].attr.lineWidth ||
						pos[i].attr.markFlag!= pos[i-1].attr.markFlag || pos[i].attr.markType!= pos[i-1].attr.markType ||
						pos[i].attr.markSize!= pos[i-1].attr.markSize || pos[i].attr.text_outside!= pos[i-1].attr.text_outside
					){
						attrchange= True;
					}
					if( (pos[i].pixelCName && pos[i].pixelCName && strcmp(pos[i].pixelCName, pos[i-1].pixelCName)) ||
						pos[i].colour.pixvalue!= pos[i-1].colour.pixvalue
					){
						colchange= True;
					}
					if( (pos[i].flpixelCName && pos[i].flpixelCName && strcmp(pos[i].flpixelCName, pos[i-1].flpixelCName)) ||
						pos[i].flcolour.pixvalue!= pos[i-1].flcolour.pixvalue
					){
						flcolchange= True;
					}
					if( attrchange ){
						buf[0]= '\0', head = buf;
						if( pos[i].attr.linestyle!= pos[i-1].attr.linestyle || pos[i].attr.lineWidth!= pos[i-1].attr.lineWidth ){
							nc = sprintf( head, "PenLineStyleWidth[%s,%s]",
								ad2str( pos[i].attr.linestyle, d3str_format, 0), ad2str( pos[i].attr.lineWidth, d3str_format, 0)
							);
							ADVANCE(head,nc);
						}
						if( pos[i].attr.markFlag!= pos[i-1].attr.markFlag || pos[i].attr.markType!= pos[i-1].attr.markType ||
							pos[i].attr.markSize!= pos[i-1].attr.markSize
						){
							if( pos[i].attr.markFlag ){
								nc = sprintf( head, ", PenMark[%s,%s]",
									ad2str( pos[i].attr.markType, d3str_format,0), ad2str( pos[i].attr.markSize, d3str_format, 0)
								);
							}
							else{
								strcat( head, ", PenMark" );
								nc = 9;
							}
							ADVANCE(head,nc);
						}
						if( pos[i].attr.text_outside!= pos[i-1].attr.text_outside ){
							if( pos[i].attr.text_outside ){
								nc = sprintf( head, ", PenTextOutside[%s]",
									ad2str( pos[i].attr.text_outside, d3str_format,0)
								);
							}
							else{
								strcat( head, ", PenTextOutside[0]" );
								nc = 19;
							}
							ADVANCE(head,nc);
						}
						if( pos[i].attr.noclip!= pos[i-1].attr.noclip ){
							nc = sprintf( head, ", PenClipping[%d]", !pos[i].attr.noclip );
						}
						else{
							strcat( head, ", PenClipping" );
							nc = 13;
						}
						if( *buf ){
							strcat( buf, " @" );
							nc += 2;
						}
						ADVANCE(head,nc);
						if( *buf ){
							if( fp ){
								fprintf( fp, "%s %s\n", opcode, buf );
							}
							else{
								StringCheck( buf, buflen, __FILE__, __LINE__ );
								eval_now( 6, buf );
							}
						}
					}
					if( colchange || flcolchange ){
						buf[0]= '\0', head = buf;
						if( colchange && (pos[i].pixelCName || pos[i].colour.pixvalue>= 0) ){
							strcat( buf, "PenColour[" );
							if( pos[i].pixelCName ){
								nc = sprintf( head, "\"%s\"]", pos[i].pixelCName );
							}
							else{
								nc = sprintf( head, "%s]", ad2str( pos[i].colour.pixvalue, d3str_format,0) );
							}
							ADVANCE(head,nc);
						}
						if( flcolchange && (pos[i].flpixelCName || pos[i].flcolour.pixvalue>= 0) ){
							if( *buf ){
								strcat( head, ", " );
								ADVANCE(head,2);
							}
							strcat( head, "PenFillColour[" );
							ADVANCE(head,14);
							if( pos[i].flpixelCName ){
								nc = sprintf( head, "\"%s\"]", pos[i].flpixelCName );
							}
							else{
								nc = sprintf( head, "%s]", ad2str( pos[i].flcolour.pixvalue, d3str_format,0) );
							}
							ADVANCE(head,nc);
						}
						if( *buf ){
							strcat( buf, " @" );
							ADVANCE(head,2);
						}
						if( *buf ){
							if( fp ){
								fprintf( fp, "%s %s\n", opcode, buf );
							}
							else{
								StringCheck( buf, buflen, __FILE__, __LINE__ );
								eval_now( 3, buf );
							}
						}
					}
				}
				if( !DumpPositions ){
					goto next_position;
				}
				if( fp ){
					fprintf( fp, "%s ", opcode );
				}
				buf[0]= '\0', head = buf;
				op = pos[i].command;
				switch( op ){
					case XGPenMoveTo:
						if( fp ){
							if( NaN(pos[i].x) && NaN(pos[i].y) ){
								fprintf( fp, "PenLift" );
							}
							else{
								fprintf( fp, "PenMoveTo[%s,%s]",
									ad2str( pos[i].x, d3str_format, 0), ad2str( pos[i].y, d3str_format,0)
								);
							}
						}
						else{
							xgPenMoveTo( 2, pos[i].x, pos[i].y, 0, NULL );
// 							ASCB_call( ascanf_PenMoveTo, NULL, 0, NULL, 3, 2, pos[i].x, pos[i].y );
						}
						break;
					case XGPenLineTo:
						if( fp ){
							fprintf( fp, "PenLineTo[%s,%s]",
								ad2str( pos[i].x, d3str_format, 0), ad2str( pos[i].y, d3str_format,0)
							);
						}
						else{
// 						  int av = ascanf_verbose;
							xgPenLineTo( 2, 0, pos[i].x, pos[i].y, 0, NULL );
							  /* 20031112: calling ascanf_PenLineTo directly leads to weird behaviour in some cases. :( */
// 							ascanf_verbose = 1;
//  							ASCB_call( ascanf_PenLineTo, NULL, 0, NULL, 3, 2, pos[i].x, pos[i].y );
// 							ascanf_verbose = av;
// 							sprintf( buf, "PenLineTo[%s,%s]",
// 								ad2str( pos[i].x, d3str_format, 0), ad2str( pos[i].y, d3str_format,0)
// 							);
// 							eval_now( 1, buf );
/* 							n= 1;	*/
						}
						break;
					case XGPenDrawRect:
					case XGPenFillRect:
					case XGPenDrawCRect:
					case XGPenFillCRect:
						if( fp ){
							fprintf( fp, "PenRectangle[%s,%s,%s,%s,%d,%d]",
								ad2str( pos[i].x, d3str_format, 0), ad2str( pos[i].y, d3str_format,0),
								ad2str( pos[i].w, d3str_format, 0), ad2str( pos[i].h, d3str_format,0),
								(op== XGPenDrawCRect || op== XGPenFillCRect)? True : False,
								(op== XGPenFillRect || op== XGPenFillCRect)?
										((pos[i].not_outlined)? -1 : 1) : 0
							);
						}
						else{
							ASCB_call( ascanf_PenRectangle, NULL, 0, NULL, 6, 6,
								pos[i].x, pos[i].y, pos[i].w, pos[i].h,
								(op== XGPenDrawCRect || op== XGPenFillCRect)? 1.0 : 0.0,
								(op== XGPenFillRect || op== XGPenFillCRect)?
										((pos[i].not_outlined)? -1.0 : 1.0) : 0.0
							);
						}
						break;
					case XGPenDrawEllps:
						if( fp ){
							fprintf( fp, "PenEllipse[%s,%s,%s,%s]",
								ad2str( pos[i].x, d3str_format, 0), ad2str( pos[i].y, d3str_format,0),
								ad2str( pos[i].w, d3str_format, 0), ad2str( pos[i].h, d3str_format,0)
							);
						}
						else{
							ASCB_call( ascanf_PenEllipse, NULL, 0, NULL, 4, 4,
								pos[i].x, pos[i].y, pos[i].w, pos[i].h
							);
						}
						break;
					case XGPenDrawPoly:
					case XGPenFillPoly:
						head = buf;
						strcat( head, "PenPolygon[" );
						ADVANCE(head,11);
						nc = sprintf( head, "{%s,%s",
							ad2str( pos[i].x, d3str_format,0), ad2str( pos[i].polyX[0], d3str_format,0)
						);
						ADVANCE(head,nc);
						for( p= 1; p< pos[i].polyN_1; p++ ){
							nc = sprintf( head, ",%s", ad2str( pos[i].polyX[p], d3str_format,0) );
							ADVANCE(head,nc);
						}
						nc = sprintf( head, "},{%s,%s",
							ad2str( pos[i].y, d3str_format,0), ad2str( pos[i].polyY[0], d3str_format,0)
						);
						ADVANCE(head,nc);
						for( p= 1; p< pos[i].polyN_1; p++ ){
							nc = sprintf( head, ",%s", ad2str( pos[i].polyY[p], d3str_format,0) );
							ADVANCE(head,nc);
						}
						nc = sprintf( head, "},%d]", (op== XGPenFillPoly)?  ((pos[i].not_outlined)? -1 : 1) : 0);
						ADVANCE(head,nc);
						n= 1;
						break;
					case XGPenText:
					case XGPenTextBox:
						nc = sprintf( head, "PenText%s[%s,%s",
							( op== XGPenTextBox )? "Box" : "",
							ad2str( pos[i].x, d3str_format, 0), ad2str( pos[i].y, d3str_format,0)
						);
						ADVANCE(head,nc);
						{ char *tbuf= NULL;
							sprint_string2( &tbuf, ",\"", "\"", pos[i].text, True );
							strcat( buf, tbuf );
							nc = strlen(tbuf);
							ADVANCE(head,nc);
							xfree( tbuf );
						}
						nc = sprintf( head, ",%d,%d", pos[i].TextBox->textJust2[0], pos[i].TextBox->textJust2[1] );
						ADVANCE(head,nc);
						if( pos[i].textFntNr< 0 ){
							nc = sprintf( head, ",&%s", pos[i].cfont_var->name );
						}
						else{
							nc = sprintf( head, ",%d", pos[i].textFntNr );
						}
						ADVANCE(head,nc);
						if( op== XGPenTextBox ){
							nc = sprintf( head, ",%d",
								(pos[i].TextBox->command== XGPenFillRect)?
									((pos[i].TextBox->not_outlined)? -1 : 1) : 0
							);
							ADVANCE(head,nc);
						}
						strcat( buf, "]" );
						ADVANCE(head,1);
						n= 1;
						break;
				}
				if( fp ){
					if( *buf ){
						fputs( buf, fp );
					}
					fputs( " @\n", fp );
				}
				else if( *buf ){
					strcat( buf, " @" );
					StringCheck( buf, buflen, __FILE__, __LINE__ );
					eval_now( n, buf );
				}
				dumped += 1;
next_position:;
			}
			if( fp ){
				fprintf( fp, "%s $IDict[0] @\n", opcode );
				fputs( "\n", fp );
			}
			else{
				*Allocate_Internal= AI;
			}
dump_next_pen:;
			Pen= Pen->next;
		}
		ActiveWin= AW;
		ascanf_window= asw;
		ascanf_check_int= aci;
	}
	return dumped;
}

int Dump_XGPens( LocalWin *wi, FILE *fp, int DumpPositions, char *opcode )
{ int ret = 0;
	if( !opcode ){
		opcode= "*STARTUP_EXPR**EVAL*";
	}
	if( wi && wi->pen_list && fp ){
	  char *buf;

		buf = (char*) calloc( LMAXBUFSIZE, sizeof(char) );
		if( !buf ){
			fprintf( StdErr, "Dump_XGPens() couldn't allocate %d size buffer (%s)\n", LMAXBUFSIZE, serror() );
			return 0;
		}
		fprintf( fp,
			"*ECHO* verbose\n"
			"The next section contains %s expressions that will restore the state of the pens at\n"
			"the moment of the dump.\n", opcode
		);
		if( wi->DumpProcessed ){
			fprintf( fp, "Processed values were dumped: this may have caused a redraw, that may in its\n"
				"turn have altered the Pens state - something which is not always visible on the screen! If this happens,\n"
				"try to prevent the redraw, e.g. by making sure the window's state is completely refreshed.\n"
			);
		}
		if( !DumpPositions ){
			fprintf( fp, "Only the pens' configurations are dumped, not the actual positions!\n" );
		}
		fputc( '\n', fp );
		ret = __Dump_XGPens( wi, NULL, fp, buf, LMAXBUFSIZE, DumpPositions, opcode );
		fprintf( fp, "*SILENCE* 0\n\n"
			"*ECHO* verbose\n... Pens read ... (%d positions)\n\n", ret
		);
		xfree(buf);
	}
	return ret;
}

int Copy_XGPens( LocalWin *source, LocalWin *target )
{ int ret = 0;
	if( source && source->pen_list && target ){
	  char *buf;

		buf = (char*) calloc( LMAXBUFSIZE, sizeof(char) );
		if( !buf ){
			fprintf( StdErr, "Copy_XGPens() couldn't allocate %d size buffer (%s)\n", LMAXBUFSIZE, serror() );
			return 0;
		}

		ret = __Dump_XGPens( source, target, NULL, buf, LMAXBUFSIZE, True, "" );
		xfree( buf );
	}
	return ret;
}

