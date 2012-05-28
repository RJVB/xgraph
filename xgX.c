/*
 * Generic Output Driver for X
 * X version 11
 *
 * This is the primary output driver used by the new X graph
 * to display output to the X server.  It has been factored
 * out of the original xgraph to allow mulitple hardcopy
 * output devices to share xgraph's capabilities.  Note:
 * xgraph is still heavily X oriented.  This is not intended
 * for porting to other window systems.
 */

#include "config.h"
IDENTIFY( "X11 device code" );

#include <stdio.h>
#include <unistd.h>
#include <math.h>
#include "xgout.h"
#include "xgraph.h"
#include "new_ps.h"
#include "NaN.h"

/* Original values:
#define PADDING     2
#define SPACE       10
#define TICKLENGTH  7
 */
/* Values corresponding to PS widths, given my HP screen (1280 pix/357 mm):
	#define PS2XSCALE	25.4/0.27890625
 */

double Xdpi= 91;
#define PS2XSCALE	Xdpi
#define PADDING rd(PS_BDR_PAD*PS2XSCALE)
#define SPACE	rd(PS_AXIS_PAD*PS2XSCALE)
#define TICKLENGTH	rd(PS_TICK_LEN*PS2XSCALE)

#define MAXSEGS   	1000
#ifndef MIN
#	define MIN(a, b)	((a) < (b) ? (a) : (b))
#endif
#ifndef MAX
#	define MAX(a, b)	((a) > (b) ? (a) : (b))
#endif

typedef struct x_state {
	PS_Printing Printing;
	Window win;   		/* Primary window           */
	LocalWin *wi;
	xgOut *dev_info;
	/* Some information for if at any time support for multiple displays is included...    */
	Display *disp;
	GC text_gc, rectangle_gc, segment_gc, dot_gc;
	Boolean silent, ps_marks;
} x_state;

int RemoteConnection = 0;

static void clear_X(struct x_state *st, int x, int y, int w, int h, int use_colour, int colour);
static void text_X( char *user_state, int x, int y, char *text, int just, int style, CustomFont *cfont );
static void seg_X(char *user_state, int ns, XSegment *segs, double Width, int style, int lappr, int colour, Pixel pixval, void *set);
static void arc_X(char *user_state, int x, int y, int rx, int ry, double La, double Ha, double Width,
	int style, int lappr, int colour, Pixel pixval);
static void dot_X( char *user_state, int x, int y, int style, int type, int colour, Pixel pixval, int setnr, void *set );
static void clear_rect_X( char *user_state, int x, int y, int w, int h, int use_colour, int colour);
static void polyg_X(char *user_state, XPoint *specs, int N, double lWidth, int style, int lappr, int colour, Pixel pixval,
	int fill, int fill_colour, Pixel fill_pixval, void *set
);
static void rect_X(char *user_state, XRectangle *specs, double lWidth, int style, int lappr, int colour, Pixel pixval,
	int fill, int fill_colour, Pixel fill_pixval, void *set);

extern int debugFlag;
extern FILE *StdErr;

extern char *index();

#include "xfree.h"

#include "ascanf.h"

#include "fdecl.h"
#include "copyright.h"

extern int rd(double x);

XRectangle *rect_xywh( int x, int y, int width, int height )
{  static XRectangle rect;
	if( width< 0 ){
		width*= -1;
		x-= width;
	}
	if( height< 0 ){
		height*= -1;
		y-= height;
	}
	rect.x= (short) x;
	rect.y= (short) y;
	rect.width= (unsigned short) abs( width);
	rect.height= (unsigned short) abs( height);
	return( &rect );
}

XRectangle *rect_diag2xywh( int x1, int y1, int x2, int y2 )
{  static XRectangle rect;
	if( x2> x1){
		rect.x= (short) x1;
		rect.width= (unsigned short) abs( x2- x1+ 1);
	}
	else{
		rect.x= (short) x2;
		rect.width= (unsigned short) abs( x1- x2+ 1);
	}
	if( y2> y1){
		rect.y= (short) y1;
		rect.height= (unsigned short) abs( y2- y1+ 1);
	}
	else{
		rect.y= (short) y2;
		rect.height= (unsigned short) abs( y1- y2+ 1);
	}
	return( &rect );
}

XRectangle *rect_xsegs2xywh( int ns, XSegment *segs )
{ int i= 0;
  int lx, ly, hx, hy;
	if( ns> 0 && segs ){
	  lx= MIN( segs[i].x1, segs[i].x2 );
	  ly= MIN( segs[i].y1, segs[i].y2 );
	  hx= MAX( segs[i].x1, segs[i].x2 );
	  hy= MAX( segs[i].y1, segs[i].y2 );
	  for( i= 1; i< ns; i++ ){
		  lx= MIN( lx, MIN( segs[i].x1, segs[i].x2 ) );
		  ly= MIN( ly, MIN( segs[i].y1, segs[i].y2 ) );
		  hx= MAX( hx, MAX( segs[i].x1, segs[i].x2 ) );
		  hy= MAX( hy, MAX( segs[i].y1, segs[i].y2 ) );
	  }
	  return( rect_diag2xywh( lx, ly, hx, hy ) );
	}
	return( NULL );
}

/* X11 method that switches off output. Side-effects of the output (GC settings
 \ and the like) are *not* disabled.
 */
int silence_X( char *user_state, int silent)
{ struct x_state *st = (struct x_state *) user_state;
   int ret;
	if( !st ){
		return(0);
	}
	ret= st->silent;
	if( debugFlag ){
		fprintf( StdErr, "%s\n", (silent)? "X11 output off" : "X11 output on" );
	}
	st->silent= silent;
	return( ret );
}

int X_silenced( LocalWin *wi )
{   struct x_state *st= (wi)? (struct x_state *) wi->dev_info.user_state : NULL;
	if( !st ){
		return(1);
	}
	if( st->Printing== X_DISPLAY ){
		return( (int) st->silent );
	}
	else{
	  int sil= wi->dev_info.xg_silent( wi->dev_info.user_state, 1);
		wi->dev_info.xg_silent( wi->dev_info.user_state, sil);
		return(sil);
	}
}

int X_ps_Marks( char *user_state, int ps_marks )
{ struct x_state *st = (struct x_state *) user_state;
   int ret= st->ps_marks;
	st->ps_marks= (ps_marks< 0)? !ret : ps_marks;
	return( ret );
}

static double mark_size_factor= 1;

int CustomFont_width_X( CustomFont *cf )
{ static void *lcf= NULL;
  static int w;
	if( cf ){
		if( cf!= lcf ){
			w= XFontWidth(cf->XFont.font);
		}
		lcf= cf;
	}
	else{
		lcf= NULL;
		w= 0;
	}
	return(w);
}

int CustomFont_height_X( CustomFont *cf )
{ static void *lcf= NULL;
  static int h;
	if( cf ){
		if( cf!= lcf ){
			h= cf->XFont.font->max_bounds.ascent + cf->XFont.font->max_bounds.descent;
		}
		lcf= cf;
	}
	else{
		lcf= NULL;
		h= 0;
	}
	return( h );
}

void Set_X( LocalWin *new_info, xgOut *out_info )
/*
 * Sets some of the common parameters for the X output device.
 */
{
	struct x_state *new_state;
	Window new_win= new_info->window;			/* Newly created window */
	int axis_height;

	out_info->dev_flags = ((depth > 3) ? D_COLOR : 0);
	out_info->bdr_pad = PADDING;
	out_info->axis_pad = SPACE;
/*     out_info->legend_pad = 0;    */
	out_info->legend_pad = rd( PS_LEG_PAD * PS2XSCALE );
	out_info->tick_len = TICKLENGTH;
	out_info->errortick_len = rd(TICKLENGTH/ 1.414);

	out_info->axis_width = XFontWidth(axisFont.font);
	out_info->axis_height =
	 axisFont.font->max_bounds.ascent + axisFont.font->max_bounds.descent;

	out_info->legend_width = XFontWidth(legendFont.font);
	out_info->legend_height =
	 legendFont.font->max_bounds.ascent + legendFont.font->max_bounds.descent;
	out_info->label_width = XFontWidth(labelFont.font);
	out_info->label_height =
	 labelFont.font->max_bounds.ascent + labelFont.font->max_bounds.descent;

	out_info->title_width = XFontWidth(titleFont.font);
	out_info->title_height =
	 titleFont.font->max_bounds.ascent + titleFont.font->max_bounds.descent;
	out_info->max_segs = MAXSEGS;

	axis_height= out_info->axis_height;
	if( new_info->ValCat_XFont && new_info->ValCat_X && new_info->ValCat_X_axis ){
		axis_height= MAX( axis_height, CustomFont_height_X( new_info->ValCat_XFont ) );
	}
	out_info->xname_vshift = (int)(out_info->bdr_pad* 2 + axis_height + out_info->label_height+ 0.5);
	if( new_info->ValCat_X_axis && new_info->ValCat_X && new_info->axisFlag ){
		out_info->xname_vshift+= (abs(new_info->ValCat_X_levels)- 1)* axis_height;
	}

	out_info->polar_width_factor= 1;
	out_info->axis_width_factor= 1;
	out_info->zero_width_factor= 1;
	out_info->var_width_factor= 1;
	  /* Determine the scaling factor for PS markers. The floating point factor is determined empirically as the quotient
	   \ of a 32cm wide PS dump (15118 devs wide) and the screenwidth of a should-be-as-wide X11 window (1241 pixels). The
	   \ other factors come from the determination of the PS width of markers of size 1. I don't think this scaling factor
	   \ ought to be screen-dependent...
	   */
	out_info->mark_size_factor= mark_size_factor= PS_MARK * rd(VDPI/ POINTS_PER_INCH* BASE_WIDTH)* 0.0820875777219209;

	out_info->xg_text = text_X;
/*     out_info->xg_clear = clear_X;    */
	out_info->xg_clear = clear_rect_X;
	out_info->xg_seg = seg_X;
	out_info->xg_rect = rect_X;
	out_info->xg_polyg= polyg_X;
	out_info->xg_arc = arc_X;
	out_info->xg_dot = dot_X;
	out_info->xg_end = (void (*)()) 0;
	out_info->xg_silent= (int (*)()) silence_X;
	out_info->xg_CustomFont_width= CustomFont_width_X;
	out_info->xg_CustomFont_height= CustomFont_height_X;
	CustomFont_width_X( NULL );
	CustomFont_height_X( NULL );

	if( !out_info->user_state ){
		if( !(new_state = (struct x_state *) calloc(1, sizeof(struct x_state))) ){
			fprintf( StdErr, "Set_X(): can't allocate state buffer (%s)\n",
				serror()
			);
			DelWindow( new_win, new_info );
		}
		out_info->user_state = (char *) new_state;
		out_info->user_ssize= sizeof(struct x_state);

		out_info->area_w = out_info->area_h = 0; /* Set later */
		new_state->silent= False;
	}
	else{
		new_state= (struct x_state*) out_info->user_state;
	}
	if( new_info->XDBE_buffer ){
		new_state->win= new_info->XDBE_buffer;
	}
	else{
		new_state->win = new_win;
	}
	new_state->disp= disp;
	new_state->wi= new_info;
	new_state->dev_info= out_info;
	if( new_state->Printing!= XG_DUMPING ){
		new_state->Printing= X_DISPLAY;
	}
}


int DisplayWidth_MM= -1, DisplayHeight_MM= -1;

int XG_DisplayWidth( Display *disp, int screen, LocalWin *wi )
{
	if( wi ){
		return( ux11_multihead_DisplayWidth(disp, screen,
			(wi->dev_info.area_x + wi->dev_info.area_w/2),
			(wi->dev_info.area_y + wi->dev_info.area_h/2), NULL, NULL, NULL) );
	}
	else{
		return( DisplayWidth(disp, screen) );
	}
}

int XG_DisplayHeight( Display *disp, int screen, LocalWin *wi )
{
	if( wi ){
		return( ux11_multihead_DisplayHeight(disp, screen,
			(wi->dev_info.area_x + wi->dev_info.area_w/2),
			(wi->dev_info.area_y + wi->dev_info.area_h/2), NULL, NULL, NULL) );
	}
	else{
		return( DisplayHeight(disp, screen) );
	}
}

int XG_DisplayWidthMM( Display *disp, int screen )
{
	if( DisplayWidth_MM> 0 ){
		return( DisplayWidth_MM );
	}
	else{
		return( DisplayWidthMM(disp, screen) );
	}
}

int XG_DisplayHeightMM( Display *disp, int screen )
{
	if( DisplayHeight_MM> 0 ){
		return( DisplayHeight_MM );
	}
	else{
// 		return( DisplayWidthMM(disp, screen) );
		// 20120528!!!
		return( DisplayHeightMM(disp, screen) );
	}
}

int DisplayXRes= -1, DisplayYRes= -1;

int XG_DisplayXRes( Display *disp, int screen )
{
	if( DisplayXRes> 0 ){
		return( DisplayXRes );
	}
	else{
		return( (int) (XG_DisplayWidth(disp,screen,NULL)/ (XG_DisplayWidthMM(disp, screen)/10.0) * 2.54+ 0.5) );
	}
}

int XG_DisplayYRes( Display *disp, int screen )
{
	if( DisplayYRes> 0 ){
		return( DisplayYRes );
	}
	else{
		return( (int) (XG_DisplayHeight(disp,screen,NULL)/ (XG_DisplayHeightMM(disp, screen)/10.0) * 2.54+ 0.5) );
	}
}

/* An inch squared, in millimeters:	*/
#define INCHSQUARE_MM	645.16

/*ARGSUSED*/
void init_X(char *user_state)
{
	struct x_state *st = (struct x_state *) user_state;
	/* Body left empty on purpose */
	st->disp= disp;
	Xdpi= sqrt( (XG_DisplayWidth(disp,screen,NULL) * XG_DisplayHeight(disp,screen,NULL)) /
			((XG_DisplayWidthMM(disp,screen) * XG_DisplayHeightMM(disp,screen)) / INCHSQUARE_MM)
		);
}

XRectangle XGClipRegion[16];
int ClipCounter= 0;

void SetXClip( LocalWin *wi, int x, int y, int w, int h)
{
	wi->ClipCounter= 1;
	wi->XGClipRegion[0].x= (short) x;
	wi->XGClipRegion[0].y= (short) y;
	wi->XGClipRegion[0].width= (unsigned short) w;
	wi->XGClipRegion[0].height= (unsigned short) h;
	if( debugFlag ){
		fprintf( StdErr, "SetXClip(%d,%d,%u,%u)\n",
			(int) wi->XGClipRegion[0].x, (int) wi->XGClipRegion[0].y,
			(unsigned int) wi->XGClipRegion[0].width, (unsigned int) wi->XGClipRegion[0].height
		);
		fflush( StdErr );
	}
}

void AddXClip( LocalWin *wi, int x, int y, int w, int h)
{
	wi->XGClipRegion[wi->ClipCounter].x= (short) x;
	wi->XGClipRegion[wi->ClipCounter].y= (short) y;
	wi->XGClipRegion[wi->ClipCounter].width= (unsigned short) w;
	wi->XGClipRegion[wi->ClipCounter].height= (unsigned short) h;
	if( debugFlag ){
		fprintf( StdErr, "AddXClip(%d,%d,%u,%u)\n",
			(int) wi->XGClipRegion[wi->ClipCounter].x, (int) wi->XGClipRegion[wi->ClipCounter].y,
			(unsigned int) wi->XGClipRegion[wi->ClipCounter].width, (unsigned int) wi->XGClipRegion[wi->ClipCounter].height
		);
		fflush( StdErr );
	}
	if( wi->ClipCounter< sizeof(wi->XGClipRegion)/sizeof(XRectangle) - 1 ){
		wi->ClipCounter+= 1;
	}
}

static int SetClip( LocalWin *wi, Display *disp, GC gc )
{ int ret;
	ret= XSetClipRectangles( disp, gc, 0, 0, &wi->XGClipRegion[0], wi->ClipCounter, Unsorted );
	return( ret );
}

extern int GCError;
extern LocalWin *ChangeWindow;

void close_X(char *user_state)
{ x_state *st= (x_state*) user_state;
	if( st->text_gc ){
		XFreeGC( st->disp, st->text_gc );
		st->text_gc= (GC) 0;
	}
	if( st->rectangle_gc ){
		XFreeGC( st->disp, st->rectangle_gc );
		st->rectangle_gc= (GC) 0;
	}
	if( st->segment_gc ){
		XFreeGC( st->disp, st->segment_gc );
		st->segment_gc= (GC) 0;
	}
	if( st->dot_gc ){
		XFreeGC( st->disp, st->dot_gc );
		st->dot_gc= (GC) 0;
	}
}

static void CheckGC(x_state *st)
{
	if( GCError ){
	 /* Oops! This only (hopefully) happens when something's seriously wrong.
       \ Just too hell... renew all the window's GCs, and mark the window for
       \ "reincarnation" just to get rid of it!
       */
		close_X( (char*) st );
		GCError= 0;
		ChangeWindow= st->wi;
	}
}

GC X_CreateGC(Display *disp, Window win, unsigned long gcmask, XGCValues *gcvals, char *fnc, char *__file__, int __line__)
{ GC ret= XCreateGC( disp, win, gcmask, gcvals );
	if( debugFlag ){
		fprintf( StdErr, "Created GC (%s, file %s, line %d)\n", fnc, __file__, __line__ );
		fflush( StdErr );
	}
	return( ret );
}

#define _XCreateGC(disp,win,gcmask,gcvals,fnc)	X_CreateGC(disp, win, gcmask, gcvals, fnc, __FILE__, __LINE__)

static GC textGC(st, t_font)
x_state *st;
XFontStruct *t_font;		/* Text font            */
/*
 * Sets the fields above in a global graphics context.  If
 * the graphics context does not exist,  it is created.
 */
{
	XGCValues gcvals;
	unsigned long gcmask;
	extern Pixel textPixel;
	extern int use_textPixel;

	gcvals.plane_mask= AllPlanes;
	gcvals.font = t_font->fid;
	gcvals.foreground = (use_textPixel)? textPixel : normPixel;
	gcvals.background= bgPixel;
	gcmask = GCPlaneMask | GCFont|GCForeground|GCBackground;
	CheckGC(st);
	if (st->text_gc == (GC) 0 ){
		st->text_gc = _XCreateGC(st->disp, st->win, gcmask, &gcvals, "textGC" );
	} else {
		XChangeGC(st->disp, st->text_gc, gcmask, &gcvals);
	}
	SetClip( st->wi, st->disp, st->text_gc );
	return st->text_gc;
}

static GC rectGC( x_state *st, Pixel r_fg, int r_style, int r_width, char *r_chars, int r_len)
/*
 * Sets the fields above in a global graphics context.  If the
 * graphics context does not exist, it is created.
 */
{
	XGCValues gcvals;
	unsigned long gcmask;

	gcvals.plane_mask= AllPlanes;
	gcvals.foreground = r_fg;
	gcvals.background= bgPixel;
	gcvals.line_style = r_style;
	gcvals.line_width = r_width;
	gcvals.fill_style = FillSolid;
	gcvals.join_style= JoinRound;
	gcvals.cap_style= CapButt;
	gcmask = GCPlaneMask | GCForeground | GCBackground| GCLineStyle | GCLineWidth | GCFillStyle|GCJoinStyle|GCCapStyle;
	CheckGC(st);
	if( st->rectangle_gc == (GC) 0 ){
		st->rectangle_gc = _XCreateGC(st->disp, st->win, gcmask, &gcvals, "rectGC" );
	} else {
		XChangeGC(st->disp, st->rectangle_gc, gcmask, &gcvals);
	}
	if (r_len > 0) {
		XSetDashes(st->disp, st->rectangle_gc, 0, r_chars, r_len);
	}
	SetClip( st->wi, st->disp, st->rectangle_gc );
	return st->rectangle_gc;
}

static GC segGC( x_state *st, Pixel l_fg, int l_style, int l_width, char *l_chars, int l_len,
	Pixel l_bg
)
/*
 * Sets the fields above in a global graphics context.  If the
 * graphics context does not exist, it is created.
 */
{ XGCValues gcvals;
  unsigned long gcmask;

	gcvals.foreground = l_fg;
	gcvals.background= l_bg;
	gcvals.plane_mask= AllPlanes;
	gcvals.clip_mask = None;
/*
    gcvals.clip_mask = FillSolid;
    gcvals.line_style = l_style;
    gcvals.line_width = l_width;
    gcmask = GCPlaneMask | GCForeground | GCLineStyle | GCLineWidth | GCClipMask;
 */
	gcmask = GCForeground|GCBackground;
	CheckGC(st);
	if( st->segment_gc == (GC) 0 ){
		gcvals.fill_style= FillSolid;
		gcvals.join_style= JoinRound;
		gcvals.cap_style= CapButt;
		gcmask|= GCFillStyle|GCJoinStyle|GCCapStyle;
		st->segment_gc = _XCreateGC(st->disp, st->win, gcmask, &gcvals, "segGC" );
	} else {
		XChangeGC(st->disp, st->segment_gc, gcmask, &gcvals);
	}
	if( l_width<= 1 ){
		l_width= 0;
	}
	if (l_len > 0) {
	  /* 990420: joinstyle was JoinMiter and SetDashes 2nd call	*/
		XSetDashes(st->disp, st->segment_gc, l_width, l_chars, l_len);
		XSetLineAttributes( st->disp, st->segment_gc, l_width, LineOnOffDash, CapButt, JoinRound );
	}
	else{
		XSetLineAttributes( st->disp, st->segment_gc, l_width, LineSolid, CapButt, JoinRound );
	}
	SetClip( st->wi, st->disp, st->segment_gc );
	return st->segment_gc;
}

static GC dotGC(st, d_fg, d_clipmask, d_xorg, d_yorg)
x_state *st;
Pixel d_fg;   		/* Foreground colour */
Pixmap d_clipmask;		/* Clipmask         */
int d_xorg, d_yorg;   	/* Clipmask origin  */
/*
 * Sets the fields above in a global graphics context.  If the
 * graphics context does not exist, it is created.
 */
{
	XGCValues gcvals;
	unsigned long gcmask;

	gcvals.foreground = d_fg;
	gcvals.background = bgPixel;
	gcvals.plane_mask= AllPlanes;
	gcvals.line_style = LineSolid;
	gcvals.line_width = 0;
	if( (gcvals.clip_mask = d_clipmask)!= None ){
		gcvals.fill_style = FillOpaqueStippled;
		gcvals.clip_x_origin = d_xorg;
		gcvals.clip_y_origin = d_yorg;
		gcmask = GCPlaneMask | GCForeground | GCBackground | GCClipMask | GCClipXOrigin | GCClipYOrigin | GCLineStyle | GCLineWidth | GCFillStyle;
	}
	else{
		gcvals.fill_style = FillSolid;
		gcmask = GCPlaneMask | GCForeground | GCBackground | GCClipMask | GCLineStyle | GCLineWidth | GCFillStyle;
	}
	CheckGC(st);
	if( st->dot_gc == (GC) 0 ){
		st->dot_gc = _XCreateGC(st->disp, RootWindow( st->disp, screen), gcmask, &gcvals, "dotGC");
	}
	else{
		XChangeGC(st->disp, st->dot_gc, gcmask, &gcvals);
	}
	if( d_clipmask== None ){
		SetClip( st->wi, st->disp, st->dot_gc );
	}
	return st->dot_gc;
}


static XFontStruct *lfont;

XGFontStruct *XGFont( long which, int *best_crit )
{ XGFontStruct *Fnt= NULL;
	switch( which ){
		case (long) 'TITL':
			Fnt= &titleFont;
			if( best_crit ){
				*best_crit= 4500;
			}
			break;
		case (long) 'MARK':
			Fnt= &markFont;
			break;
		case (long) 'LABL':
			Fnt= &labelFont;
			if( best_crit ){
				*best_crit= 5000;
			}
			break;
		case (long) 'LEGN':
			Fnt= &legendFont;
			if( best_crit ){
				*best_crit= 4500;
			}
			break;
		case (long) 'AXIS':
			Fnt= &axisFont;
			if( best_crit ){
				*best_crit= 4000;
			}
			break;
		case (long) 'DIAL':
			Fnt= &dialogFont;
			if( best_crit ){
				*best_crit= 4175;
			}
			break;
		case (long) 'FDBK':
			Fnt= &fbFont;
			break;
	}
	return( Fnt );
}

int New_XGFont( long which, char *font_name )
{ int best_size;
  XGFontStruct *Fnt= XGFont( which, &best_size );
  XFontStruct *tempFont= NULL;
  char use[32];
	if( !Fnt ){
		return(0);
	}
	if( Fnt->use[0] ){
		strncpy( use, Fnt->use, sizeof(use)/sizeof(char) );
	}
	else{
	  char *c= (char*) &which;
		strncpy( use, c, 4 );
		use[4]= '\0';
		strncpy( Fnt->use, use, sizeof(Fnt->use)/sizeof(char) );
	}
	if( !Fnt->font || (Fnt->name && strcasecmp( font_name, Fnt->name )) ){
		if( strcasecmp( font_name, "best")== 0 ){
		  char Rtn_name[128]= "", *rtn_name= Rtn_name;
			if( !ux11_size_font(disp, DefaultScreen(disp), best_size, &tempFont, &rtn_name, (best_size>=4500)? True : False )){
				tempFont= NULL;
			}
			else{
				font_name= rtn_name;
			}
		}
		else{
		  XGFontStruct lFnt;
/* 			tempFont = XLoadQueryFont(disp, font_name);	*/
			lFnt.font= 0;
			strncpy( lFnt.use, FontName( Fnt ), sizeof(lFnt.use)/sizeof(char) );
			GetFont( &lFnt, NULL, font_name, 0, 0, 0 );
			if( (tempFont= lFnt.font) && strcmp( lFnt.name, font_name) ){
				fprintf( StdErr, "Request for \"%s\" font '%s', yielded '%s'\n",
					lFnt.use, font_name, lFnt.name
				);
			}
		}
		if( !tempFont){
			fprintf( StdErr, "can't get %s '%s'\n", use, font_name);
			return(0);
		}
		else if( Fnt->font!= tempFont ){
			if( Fnt->font ){
				XFreeFont( disp, Fnt->font );
				Fnt->font= NULL;
			}
			Fnt->font = tempFont;
			strncpy( Fnt->name, font_name, 127);
			if( debugFlag){
				fprintf( StdErr, "%s : %s\n", use, font_name);
			}
			Update_greekFonts( which );
		}
	}
	return(1);
}

int XGFontWidth( LocalWin *wi, int FontNR, char *text, int *width, int *height, CustomFont *cfont, XCharStruct *bb, double *scale )
{ XFontStruct *font1, *font2;
  int maxWidth;
  int has_greek;

	if( cfont ){
		font1= cfont->XFont.font;
		font2= cfont->X_greekFont.font;
		if( wi && width ){
			*width= (*wi->dev_info.xg_CustomFont_width)( cfont );
		}
		if( wi && height ){
			*height= (*wi->dev_info.xg_CustomFont_height)( cfont );
		}
	}
	else switch( FontNR ){
		case T_TITLE:
			font1= titleFont.font;
			font2= title_greekFont.font;
			if( wi && width ){
				*width= wi->dev_info.title_width;
			}
			if( wi && height ){
				*height= wi->dev_info.title_height;
			}
			break;
		case T_MARK:
			font1= markFont.font;
			font2= NULL;
			if( wi && width ){
				*width= -1;
			}
			if( wi && height ){
				*height= -1;
			}
			break;
		case T_LABEL:
			font1= labelFont.font;
			font2= label_greekFont.font;
			if( wi && width ){
				*width= wi->dev_info.label_width;
			}
			if( wi && height ){
				*height= wi->dev_info.label_height;
			}
			break;
		case T_LEGEND:
			font1= legendFont.font;
			font2= legend_greekFont.font;
			if( wi && width ){
				*width= wi->dev_info.legend_width;
			}
			if( wi && height ){
				*height= wi->dev_info.legend_height;
			}
			break;
		case T_AXIS:
			font1= axisFont.font;
			font2= axis_greekFont.font;
			if( wi && width ){
				*width= wi->dev_info.axis_width;
			}
			if( wi && height ){
				*height= wi->dev_info.axis_height;
			}
			break;
	}
	if( text ){
		has_greek= (int) xtb_has_greek(text);
		if( bb ){
		  int dir, ascent, descent;
			xtb_TextExtents( font1, font2, text, strlen(text), &dir, &ascent, &descent, bb, False );
		}
		if( wi && scale && PS_STATE(wi)->Printing== PS_PRINTING ){
			if( cfont ){
				*scale= CustomFont_psWidth(cfont) / MAX( XFontWidth(cfont->XFont.font), XFontWidth(cfont->X_greekFont.font) );
			}
			else{
				*scale= (*width) / MAX( XFontWidth(font1), XFontWidth(font2) );
			}
		}
		else{
			*scale= 1;
		}
	}
	else{
		has_greek= False;
	}
	maxWidth= XFontWidth( font1 );
	if( has_greek ){
		maxWidth= MAX( maxWidth, XFontWidth( font2 ) );
	}
	return( maxWidth );
}

/* Same as XGFontWidth, but also determines the total height for strings containing multiple lines. This is a
 \ generalised version of increment_height()...
 */
int XGFontWidth_Lines( LocalWin *wi, int FontNR, char *text, char letter, int *width, int *theight, int *height, CustomFont *cfont, XCharStruct *bb, double *scale )
{ int maxWidth= XGFontWidth( wi, FontNR, text, width, height, cfont, bb, scale );
	if( theight && height && text ){
	  char *c= text;
	  int n= 0;
		*theight= *height;
		while( c && *c ){
			if( *c== letter && c[1] ){
			  /* The last '\n' is redundant: it does not have an effect	*/
				(*theight)+= *height;
				n+= 1;
			}
			c++;
		}
		if( debugFlag ){
			fprintf( StdErr, "XGFontWidth_Lines(): string \"%s\" is %d high (%d lines)\n",
				text, *theight, n
			);
		}
	}
	return( maxWidth );
}

extern int use_gsTextWidth, _use_gsTextWidth, used_gsTextWidth;
/* static unsigned long NgsTW= 0;	*/
/* static double SgsTW= 0;	*/

int XGTextWidth( LocalWin *wi, char *text, int style, CustomFont *cfont)
{ int w= -1;
  XFontStruct *font1, *font2;

	if( wi->textrel.used_gsTextWidth< 0 ){
		wi->textrel.NgsTW= 0;
		wi->textrel.SgsTW= 0;
	}
	if( PS_STATE(wi)->Printing== PS_PRINTING && use_gsTextWidth ){
		_use_gsTextWidth= use_gsTextWidth;
		w= gsTextWidth( wi, text, style, cfont )+ 0.5;
	}
	if( w< 0 || (!w && *text) ){
		_use_gsTextWidth= False;
		if( cfont ){
			font1= cfont->XFont.font;
			font2= cfont->X_greekFont.font;
		}
		else switch( style ){
			case T_TITLE:
				font1= titleFont.font;
				font2= title_greekFont.font;
				break;
			case T_MARK:
				font1= markFont.font;
				font2= NULL;
				break;
			case T_LABEL:
				font1= labelFont.font;
				font2= label_greekFont.font;
				break;
			case T_LEGEND:
				font1= legendFont.font;
				font2= legend_greekFont.font;
				break;
			case T_AXIS:
				font1= axisFont.font;
				font2= axis_greekFont.font;
				break;
		}
		lfont= font1;
		w= xtb_TextWidth( text, font1, font2);
	}
	wi->textrel.NgsTW+= 1;
	wi->textrel.SgsTW+= _use_gsTextWidth;
	wi->textrel.used_gsTextWidth= (int) (wi->textrel.SgsTW/wi->textrel.NgsTW+ 0.5);
	return(w);
}

Boolean xg_text_highlight= False;
int xg_text_highlight_colour= 0;

static void text_X( char *user_state, int x, int y, char *text, int just, int style, CustomFont *cfont )
/*
 * This routine should draw text at the indicated position using
 * the indicated justification and style.  The justification refers
 * to the location of the point in reference to the text.  For example,
 * if just is T_LOWERLEFT,  (x,y) should be located at the lower left
 * edge of the text string.
 */
{
	struct x_state *st = (struct x_state *) user_state;
	XCharStruct bb;
	int rx, ry, gry, vrx, vry, len, height, gheight, width, dir;
	int max_ascent, ascent, descent;
	char *Text, *greek_rest;
	XFontStruct *font1, *font2;
	GC tgc;

	if( !text ){
		return;
	}

	if( cfont ){
		font1= cfont->XFont.font;
		font2= cfont->X_greekFont.font;
	}
	else switch( style ){
		case T_TITLE:
			font1= titleFont.font;
			font2= title_greekFont.font;
			break;
		case T_MARK:
			font1= markFont.font;
			font2= NULL;
			break;
		case T_LABEL:
			font1= labelFont.font;
			font2= label_greekFont.font;
			break;
		case T_LEGEND:
			font1= legendFont.font;
			font2= legend_greekFont.font;
			break;
		case T_AXIS:
			font1= axisFont.font;
			font2= axis_greekFont.font;
			break;
	}
	lfont= font1;
	xtb_TextExtents( font1, font2, text, strlen(text), &dir, &ascent, &descent, &bb, False );
	max_ascent= bb.ascent;
	width= bb.rbearing - bb.lbearing;
	gheight= height= bb.ascent + bb.descent;

	switch (just) {
			case T_CENTER:
				rx = x - (width/2);
				ry = y - (height/2);
				gry = y - (gheight/2);
				break;
			case T_LEFT:
				rx = x;
				ry = y - (height/2);
				gry = y - (gheight/2);
				break;
			case T_VERTICAL:
			case T_UPPERLEFT:
				rx = x;
				gry= ry = y;
				break;
			case T_TOP:
				rx = x - (width/2);
				gry= ry = y;
				break;
			case T_UPPERRIGHT:
				rx = x - width;
				gry= ry = y;
				break;
			case T_RIGHT:
				rx = x - width;
				ry = y - (height/2);
				gry = y - (gheight/2);
				break;
			case T_LOWERRIGHT:
				rx = x - width;
				ry = y - height;
				gry = y - gheight;
				break;
			case T_BOTTOM:
				rx = x - (width/2);
				ry = y - height;
				gry = y - gheight;
				break;
			case T_LOWERLEFT:
				rx = x;
				ry = y - height;
				gry = y - gheight;
				break;
	}

	lfont= font1;
	Text= text;
	vrx= rx;
/* 	if( Text[0]== '\\' && Text[1]!= '\\' && style!= T_MARK )	*/
	if( style!= T_MARK && xtb_toggle_greek(Text,text) )
	{
		vry= gry;
	}
	else{
		vry= ry;
	}
	while( Text && *Text ){
	 int y;
	 char *_Text;
		if( (greek_rest= xtb_has_greek( &Text[1] )) && style!= T_MARK ){
			*greek_rest= '\0';
		}
/* 		if( Text[0]== '\\' && Text[1]!= '\\' && style!= T_MARK )	*/
		if( style!= T_MARK && xtb_toggle_greek(Text, text) )
		{
			Text++;
			if( lfont== font1){
				lfont= font2;
				y= gry;
			}
			else{
				lfont= font1;
				y= ry;
			}
		}
		else{
			y= ry;
		}
		_Text= xtb_textfilter( Text, lfont, False );
		len = strlen(_Text);
		XTextExtents( lfont, _Text, len, &dir, &ascent, &descent, &bb);
		tgc= textGC(st, lfont);
		if( !st->silent ){
		 int width;
			width= bb.rbearing - bb.lbearing;

			if( xg_text_highlight ){
				clear_rect_X( user_state, rx, y+ max_ascent- height,
					width, height, 1, xg_text_highlight_colour
				);
			}
			if( just== T_VERTICAL ){
			  XRectangle rect= *rect_xywh( vrx, vry+ max_ascent- height, height, -width );
				rect_X( user_state, &rect, 0, L_VAR, 0, 0, 0, 0, 0, 0, NULL );
			}
			XDrawString(st->disp, st->win, tgc, rx, y + max_ascent, _Text, len);
		}
		rx+= bb.rbearing - bb.lbearing;
		vry-= bb.rbearing - bb.lbearing;
		if( greek_rest && style!= T_MARK ){
			Text= greek_rest;
			*Text= '\\';
		}
		else{
			Text= NULL;
		}
	}
/* 	xtb_textfilter_free();	*/
	xg_text_highlight= False;
	ps_old_font_offsets= False;
}


static void clear_X(struct x_state *st, int x, int y, int w, int h, int use_colour, int colour)
{ int i;
	if( !RemoteConnection ){
		XFlush( st->disp);
	}
	xtb_XSync( st->disp, False);
	if( st->silent ){
		return;
	}
	if( use_colour ){
	 /* Just draw a rectangle with the specified coordinates and colour    */
	}
	else{
		ClipCounter= st->wi->ClipCounter;
		memcpy( XGClipRegion, st->wi->XGClipRegion, sizeof(XGClipRegion) );
		for( i= 0; i< ClipCounter; i++ ){
			XClearArea( st->disp, st->win,
				(int) XGClipRegion[i].x, (int) XGClipRegion[i].y, (int) XGClipRegion[i].width, (int) XGClipRegion[i].height,
				False
			);
			if( debugFlag ){
				fprintf( StdErr, "\tclear_X(%d,%d,%d,%d) #%d\n",
					(int) XGClipRegion[i].x, (int) XGClipRegion[i].y, (int) XGClipRegion[i].width, (int) XGClipRegion[i].height,
					i
				);
				fflush( StdErr );
			}
		}
		if( !RemoteConnection ){
			XFlush( st->disp);
		}
		xtb_XSync( st->disp, False);
	}
}

static void clear_rect_X( char *user_state, int x, int y, int w, int h, int use_colour, int colour)
{ struct x_state *st = (struct x_state *) user_state;
  GC gc;

	gc = (use_colour)? rectGC(st, AllAttrs[colour].pixelValue, LineSolid, 0, (char *) 0, 0) :
		rectGC(st, bgPixel, LineSolid, 0, (char *) 0, 0);
	if( st->silent ){
		return;
	}
	if( st->win!= st->wi->XDBE_buffer ){
		  /* This generates errors when we pass it an XdbeBackBuffer?? */
		XSetWindowBackground( st->disp, st->win, bgPixel );
	}
/* 	XFlush( st->disp );	*/
	  /* 990715: xsync made "false" 990716: restored	*/
	  /* 20000427: made false again	*/
	  /* 20020513: actually, I don't see the need to do all that flushing here! */
/* 	xtb_XSync( st->disp, False );	*/
	if( w< 0 ){
		w*= -1;
		x-= w;
	}
	if( h< 0 ){
		h*= -1;
		y-= h;
	}
	XFillRectangle(st->disp, st->win, gc, x, y, w, h);
/* 	XFlush( st->disp );	*/
	if( debugFlag ){
		fprintf( StdErr, "\tclear_rect_X(%d,%d,%d,%d)\n", x, y, w, h);
		fflush( StdErr );
	}
}

#ifndef MAX
#	define MAX(a,b)	((a) > (b) ? (a) : (b))
#endif
#ifndef MIN
#	define MIN(a,b)	((a) < (b) ? (a) : (b))
#endif
#define MINMAX(min,max,val) if((val)<(min)){(min)=(val);}else if((val)>(max)){(max)=(val);}

static void Find_Segs_Bounds( XSegment *segs, int ns, DataSet *set, char *caller, Boolean use_both )
{ int i, s;
#ifdef DEBUG_SEGS_BOUNDS
   extern int setNumber;
	if( set->set_nr< 0 || set->set_nr>= setNumber ){
		fprintf( StdErr, "%s called with invalid set (#%d)\n", caller, set->set_nr );
		fflush( StdErr);
	}
#endif
	if( set->s_bounds_set<= 0 ){
		if( use_both ){
			set->sx_min= MIN( segs->x1, segs->x2);
			set->sy_min= MIN( segs->y1, segs->y2);
			set->sx_max= MAX( segs->x1, segs->x2);
			set->sy_max= MAX( segs->y1, segs->y2);
		}
		else{
			set->sx_min= set->sx_max= segs->x1;
			set->sy_min= set->sy_max= segs->y1;
		}
		set->s_bounds_set= 1;
		s= 1;
	}
	else{
		s= 0;
	}
	for( i= s; i< ns; i++, segs++ ){
		MINMAX( set->sx_min, set->sx_max, segs->x1);
		MINMAX( set->sy_min, set->sy_max, segs->y1);
		if( use_both ){
			MINMAX( set->sx_min, set->sx_max, segs->x2);
			MINMAX( set->sy_min, set->sy_max, segs->y2);
		}
	}
}

static void Find_XPts_Bounds( XPoint *pts, int ns, DataSet *set, char *caller )
{ int i, s;
#ifdef DEBUG_SEGS_BOUNDS
   extern int setNumber;
	if( set->set_nr< 0 || set->set_nr>= setNumber ){
		fprintf( StdErr, "%s called with invalid set (#%d)\n", caller, set->set_nr );
		fflush( StdErr);
	}
#endif
	if( set->s_bounds_set<= 0 ){
		set->sx_min= set->sx_max= pts->x;
		set->sy_min= set->sy_max= pts->y;
		set->s_bounds_set= 1;
		s= 1;
	}
	else{
		s= 0;
	}
	for( i= s; i< ns; i++, pts++ ){
		MINMAX( set->sx_min, set->sx_max, pts->x);
		MINMAX( set->sy_min, set->sy_max, pts->y);
	}
}

static void seg_X(char *user_state, int ns, XSegment *segs, double Width, int style, int lappr, int colour, Pixel pixval, void *set)
/*
 * This routine draws a number of line segments at the points
 * given in `seglist'.  Note that contiguous segments need not share
 * endpoints but often do.  All segments should be `width' devcoords wide
 * and drawn in style `style'.  If `style' is L_VAR,  the parameters
 * `colour' and `lappr' should be used to draw the line.  Both
 * parameters vary from 0 to 7.  If the device is capable of
 * colour,  `colour' varies faster than `style'.  If the device
 * has no colour,  `style' will vary faster than `colour' and
 * `colour' can be safely ignored.  However,  if the
 * the device has more than 8 line appearences,  the two can
 * be combined to specify 64 line style variations.
 * Xgraph promises not to send more than the `max_segs' in the
 * xgOut structure passed back from xg_init().
 */
{ struct x_state *st = (struct x_state *) user_state;
  GC gc;
  int width;

	CLIP( Width, 0, MAXINT-1 );
	width= (int) (Width+ 0.5);
	lappr= lappr % MAXATTR;
	colour= colour % MAXATTR;
	switch( style ){
		case L_AXIS:
			gc = segGC(st, (colour< 0)? pixval : normPixel, LineSolid, /* axisWidth */ width, (char *) 0, 0, bgPixel );
			break;
		case L_ZERO:
			/* Set the colour and line style */
			if( zeroLSLen> 0 ){
				gc = segGC(st, zeroPixel, LineSolid, /* axisWidth+zeroWidth */ width, zeroLS, zeroLSLen, bgPixel  );
			}
			else{
				gc = segGC(st, zeroPixel, LineSolid, /* axisWidth+zeroWidth */ width, (char *) 0, 0, bgPixel );
			}
			break;
		case L_POLAR:
			width= 0;
			 /* fall-through to default (L_VAR):   */
		default:
			/* Colour and line style vary */
			if (lappr == 0) {
				gc = segGC(st, (colour< 0)? pixval : AllAttrs[colour].pixelValue, LineSolid,
					   width, (char *) 0, 0, bgPixel );
			} else {
				gc = segGC(st, (colour< 0)? pixval : AllAttrs[colour].pixelValue, LineOnOffDash,
					   width, AllAttrs[lappr].lineStyle, AllAttrs[lappr].lineStyleLen, bgPixel );
			}
			break;
	}
	if( set ){
		Find_Segs_Bounds( segs, ns, set, "xg_seg", True );
	}
	  /* 980527	moved following test to after Find_Segs_Bounds	*/
	if( st->silent ){
		return;
	}

#ifdef DEBUG
	if( set ){
	  int i;
		for( i= 0; i< ns; i++ ){
			if( segs[i].x1<= 0 && segs[i].y1<= 0 && segs[i].x2<= 0 && segs[i].y2<= 0 ){
				fprintf( StdErr, "seg_X(set=%d,segs=%d): seg %d is (0,0)\n",
					((DataSet*)set)->set_nr, ns, i
				);
			}
		}
	}
#endif

	  /* 980212: for some reason I used XDrawLines until now. I don't remember whether there was
	   \ an important reason to do so, but anyway it evidently doesn't draw multiple, unconnected line-segments..
			XDrawLines( st->disp, st->win, gc, (XPoint*) segs, 2* ns, CoordModeOrigin );
	   */
	if( st->wi->data_sync_draw && Synchro_State ){
	  /* 20061130 */
	  int i;
		for( i= 0; i< ns; i++ ){
			XDrawLine(st->disp, st->win, gc, segs[i].x1, segs[i].y1, segs[i].x2, segs[i].y2 );
		}
	}
	else{
	  int i;
#if 0
	  int ss;
		if( ns> 512 && !Synchro_State ){
			ss= True;
			XSynchronize( disp, True );
		}
		else{
			ss= False;
		}
#endif
		for( i= 0; ns> 2*MAXSEGS; i+= 2*MAXSEGS ){
			XDrawSegments(st->disp, st->win, gc, &segs[i], 2*MAXSEGS );
			ns-= 2*MAXSEGS;
		}
		XDrawSegments(st->disp, st->win, gc, &segs[i], ns );
#if 0
		if( ss ){
			XSynchronize( disp, False );
		}
#endif
	}
}

static GC SetShapeGC( x_state *st, int style, int colour, Pixel ol, Pixel fl, int fill, int lwidth, int lappr )
{ GC gc;
	switch( style ){
		case L_AXIS:
			gc = segGC(st, (fill)? fl : normPixel, LineSolid, lwidth, (char *) 0, 0, bgPixel );
			break;
		case L_ZERO:
			/* Set the colour and line style */
			if( zeroLSLen> 0 ){
				gc = segGC(st, (fill)? fl : zeroPixel, LineSolid, lwidth, zeroLS, zeroLSLen, bgPixel  );
			}
			else{
				gc = segGC(st, (fill)? fl : zeroPixel, LineSolid, lwidth, (char *) 0, 0, bgPixel );
			}
			break;
		case L_POLAR:
			lwidth= 0;
			 /* fall-through to default (L_VAR):   */
		default:
			/* Colour and line style vary */
			if (lappr == 0) {
				gc = segGC(st, (fill)? fl : (colour< 0)? ol : AllAttrs[colour].pixelValue, LineSolid,
					   lwidth, (char *) 0, 0, bgPixel );
			} else {
				gc = segGC(st, (fill)? fl : (colour< 0)? ol : AllAttrs[colour].pixelValue, LineOnOffDash,
					   lwidth, AllAttrs[lappr].lineStyle, AllAttrs[lappr].lineStyleLen, bgPixel );
			}
			break;
	}
	return( gc );
}

static void rect_X(char *user_state, XRectangle *specs, double lWidth, int style, int lappr, int colour, Pixel pixval,
	int fill, int fill_colour, Pixel fill_pixval, void *set
)
/*
 * This routine draws a rectangle at (x,y)-(x+w,y+h)
 * All segments should be `lwidth' devcoords wide
 * and drawn in style `style'.  If `style' is L_VAR,  the parameters
 * `colour' and `lappr' should be used to draw the line.  Both
 * parameters vary from 0 to MAXATTR.
 \ If fill is set, draw rectangles filled with fill_colour or fill_pixval. Note that X doesn't
 \ directly support this kind of shapes, filled with one colour, and traced with another.
 \ Thus, we need to first draw the fill with the fg colour set to the desired background,
 \ and then call ourselves (rect_X) again, with the fill argument to False, in order
 \ to draw the outline.
 */
{ struct x_state *st = (struct x_state *) user_state;
  GC gc;
  Pixel bg;
  int lwidth;

	CLIP( lWidth, 0, MAXINT-1 );
	lwidth= (int) (lWidth+ 0.5);

	lappr= lappr % MAXATTR;
	colour= colour % MAXATTR;
	fill_colour= fill_colour % MAXATTR;
	if( fill_colour< 0 ){
		bg= fill_pixval;
	}
	else{
		bg= AllAttrs[fill_colour].pixelValue;
	}
	gc= SetShapeGC( st, style, colour, pixval, bg, fill, lwidth, lappr );
	if( set && !fill ){
	  XSegment seg;
		seg.x1= specs->x;
		seg.x2= specs->x+specs->width;
		seg.y1= specs->y;
		seg.y2= specs->y+specs->height;
		Find_Segs_Bounds( &seg, 1, set, "xg_rect", True );
	}
	  /* 980527	moved following test to after Find_Segs_Bounds	*/
	if( st->silent ){
		return;
	}
	if( fill ){
		XFillRectangle( st->disp, st->win, gc, specs->x, specs->y, specs->width, specs->height );
		  /* 991208: no need to call again if framecolour is to be the same...	*/
		if( colour!= fill_colour || pixval!= fill_pixval || lWidth> 1 ){
			rect_X(user_state, specs, lwidth, style, lappr, colour, pixval,
				0, fill_colour, fill_pixval, set
			);
		}
	}
	else{
		XDrawRectangle( st->disp, st->win, gc, specs->x, specs->y, specs->width, specs->height );
	}
}

static void polyg_X(char *user_state, XPoint *specs, int N, double lWidth, int style, int lappr, int colour, Pixel pixval,
	int fill, int fill_colour, Pixel fill_pixval, void *set
)
/*
 * This routine draws a polygon with the points listed in specs.
 \ Behaviour as described for rect_X().
 */
{ struct x_state *st = (struct x_state *) user_state;
  GC gc;
  Pixel bg;
  int lwidth;

	CLIP( lWidth, 0, MAXINT-1 );
	lwidth= (int) (lWidth+ 0.5);

	lappr= lappr % MAXATTR;
	colour= colour % MAXATTR;
	fill_colour= fill_colour % MAXATTR;
	if( fill_colour< 0 ){
		bg= fill_pixval;
	}
	else{
		bg= AllAttrs[fill_colour].pixelValue;
	}
	gc= SetShapeGC( st, style, colour, pixval, bg, fill, lwidth, lappr );
	if( set ){
		Find_XPts_Bounds( specs, N, set, "xg_polyg" );
	}
	if( st->silent ){
		return;
	}
	if( fill ){
		XFillPolygon( st->disp, st->win, gc, specs, N, Complex, CoordModeOrigin );
		  /* 991208: no need to call again if framecolour is to be the same...	*/
		if( colour!= fill_colour || pixval!= fill_pixval || lWidth> 1 ){
			polyg_X(user_state, specs, N, lwidth, style, lappr, colour, pixval,
				0, fill_colour, fill_pixval, set
			);
		}
	}
	else{
		XDrawLines( st->disp, st->win, gc, specs, N, CoordModeOrigin );
		if( memcmp( &specs[N-1], &specs[0], sizeof(XPoint)) ){
		  XPoint close[2];
			close[0]= specs[N-1];
			close[1]= specs[0];
			XDrawLines( st->disp, st->win, gc, close, 2, CoordModeOrigin );
		}
	}
}

void _arc_X( Display *disp, Window win, GC gc, int x, int y, int rx, int ry, double la, double ha )
{
	XDrawArc(disp, win, gc, x- rx, y- ry, 2* rx, 2*ry, (int) (64* la+ 0.5), (int) (64* ha+ 0.5) );
}

static void arc_X(char *user_state, int x, int y, int rx, int ry, double La, double Ha, double Width,
	int style, int lappr, int colour, Pixel pixval)
{
	struct x_state *st = (struct x_state *) user_state;
	GC gc;
	extern double fabs();
	double la, ha;
	int width;

	CLIP( Width, 0, MAXINT-1 );
	width= (int) (Width+ 0.5);

	lappr= lappr % MAXATTR;
	colour= colour % MAXATTR;
	switch( style ){
		case L_AXIS:
			gc = segGC(st, normPixel, LineSolid, /* axisWidth */ width, (char *) 0, 0, bgPixel );
			break;
		case L_ZERO:
			/* Set the colour and line style */
			gc = segGC(st, zeroPixel, LineSolid, /* axisWidth+zeroWidth */ width, (char *) 0, 0, bgPixel );
			break;
		case L_POLAR:
			width= 0;
			 /* fall-through to default (L_VAR):   */
		default:
			/* Colour and line style vary */
			if (lappr == 0) {
				gc = segGC(st, (colour<0)? pixval : AllAttrs[colour].pixelValue, LineSolid,
					   width, (char *) 0, 0, bgPixel );
			} else {
				gc = segGC(st, (colour<0)? pixval : AllAttrs[colour].pixelValue, LineOnOffDash,
					   width, AllAttrs[lappr].lineStyle, AllAttrs[lappr].lineStyleLen, bgPixel );
			}
			break;
	}
	if( st->silent ){
		return;
	}
	la=   (La+ st->wi->radix_offset)* 360.0/ st->wi->radix;
	ha= fabs((Ha+ st->wi->radix_offset)* 360.0/ st->wi->radix- la);
	if( debugFlag ){
		fprintf( StdErr, "arc_X(..,%g,%g,..) -> XDrawArc(..,%d,%d,%d,%d,%d*64,%d*64)\n",
			La, Ha,
			x- rx, y- ry, 2* rx, 2*ry, (int) (la+ 0.5), (int) (ha+ 0.5)
		);
		fflush( StdErr );
	}
	 /* XDrawArc works with a [-180,180] coordinate system, always drawing towards the higher angle    */
	_arc_X( st->disp, st->win, gc, x, y, rx, ry, la, ha);
}

#define LAST_CHECK

extern LocalWin *ActiveWin;

#ifndef MININT
#	define MININT (1 << (8*sizeof(int)-1))
#endif

static double _SCREENXDIM(LocalWin *ws, double userX)
{  double screenX;
	if( !ws ){
		return(0);
	}
	if( ws->win_geo._XUnitsPerPixel ){
		CLIP_EXPR( screenX, (userX)/ws->win_geo._XUnitsPerPixel, MININT, MAXINT);
	}
	else{
		screenX= MAXINT;
	}
	return( screenX );
}

static double _SCREENYDIM(LocalWin *ws, double userY)
{  double screenY;
	if( !ws ){
		return(0);
	}
	if( ws->win_geo._YUnitsPerPixel ){
		CLIP_EXPR( screenY, (userY)/ws->win_geo._YUnitsPerPixel, MININT, MAXINT );
	}
	else{
		screenY= MAXINT;
	}
	return( screenY );
}

double _X_ps_MarkSize_X( double setnr)
{ extern int internal_psMarkers;
   extern double psm_base, psm_incr;
#ifndef OLD_PSMARK_ESTIMATION
   int i= (int) setnr;
   extern int setNumber;
   double s;
	  /* A should-be-better estimation of the real size of PS markers, translated into X co-ordinates.
	   \ Approximation is better for larger markers.
	   */
	if( i>= 0 && i< setNumber && !NaNorInf(AllSets[i].markSize) ){
		if( AllSets[i].markSize< 0 ){
			s= _SCREENXDIM( ActiveWin, -AllSets[i].markSize );
		}
		else{
			s=( ( mark_size_factor* AllSets[i].markSize ) );
		}
	}
	else{
		s=( ( mark_size_factor*( (int)(setnr/internal_psMarkers)* psm_incr+ psm_base) ) );
	}
	return(s);
#else
	return( ( PS_MARK * BASE_WIDTH *
			( (int)(setnr/internal_psMarkers)* psm_incr+ psm_base)* (Xdpi/POINTS_PER_INCH) *
			 /* screen factor - may vary per screen. Actually, this factor should be
               \ in other scaling places (inversed) - without it, the marks have the right
               \ size on my screen... I.o.w., it is merely a trick to adapt the marker size
               \ to other, not quite correct scaling. But this is easier..
               */
			0.9432624113475178
		)
	);
#endif
}

int X_ps_MarkSize_X( double setnr)
{
	return( rd( _X_ps_MarkSize_X(setnr) ) );
}

double _X_ps_MarkSize_Y( double setnr)
{ extern int internal_psMarkers;
   extern double psm_base, psm_incr;
   int i= (int) setnr;
   extern int setNumber;
   double s;
	  /* A should-be-better estimation of the real size of PS markers, translated into X co-ordinates.
	   \ Approximation is better for larger markers.
	   */
	if( i>= 0 && i< setNumber && !NaNorInf(AllSets[i].markSize) ){
		if( AllSets[i].markSize< 0 ){
			s= _SCREENYDIM( ActiveWin, -AllSets[i].markSize );
		}
		else{
			s=( ( mark_size_factor* AllSets[i].markSize ) );
		}
	}
	else{
		s=( ( mark_size_factor*( (int)(setnr/internal_psMarkers)* psm_incr+ psm_base) ) );
	}
	return(s);
}

int X_ps_MarkSize_Y( double setnr)
{
	return( rd( _X_ps_MarkSize_Y(setnr) ) );
}

static void mark_X( x_state *st, int x, int y, int type, int colour, Pixel pixval, double setnr, double scX, double scY )
{ extern int internal_psMarkers;
   double DsizeX= _X_ps_MarkSize_X( setnr)* scX, DsizeY= _X_ps_MarkSize_Y(setnr)* scY;
   int diamX= rd(2*DsizeX), diamY= rd(2*DsizeY),
/* 	   sizeX= diamX/2, sizeY= diamY/2,	*/
	   xsizeX= rd(x+DsizeX), ysizeY= rd(y+DsizeY),
	   x_sizeX= rd(x-DsizeX), y_sizeY= rd(y-DsizeY);
   GC gc;
   Pixel fgPixel= (colour< 0)? pixval : AllAttrs[colour].pixelValue;
   XPoint poly[5];
	if( st->silent ){
		return;
	}
	type%= internal_psMarkers;
	switch( type ){
		case 0:
		 /* M0: a blank, filled rectangle  */
			gc = segGC(st, bgPixel, LineSolid, 0, (char *) 0, 0, bgPixel );
			XFillRectangle(st->disp, st->win, gc,
				x_sizeX, y_sizeY,
				diamX, diamY
			);
			gc = segGC(st, fgPixel, LineSolid, 0, (char *) 0, 0, bgPixel );
			XDrawRectangle(st->disp, st->win, gc,
				x_sizeX, y_sizeY,
				diamX, diamY
			);
			break;
		case 1:
		 /* M1: a black, filled rectangle  */
			gc = segGC(st, fgPixel, LineSolid, 0, (char *) 0, 0, bgPixel );
			XFillRectangle(st->disp, st->win, gc,
				x_sizeX, y_sizeY,
				diamX, diamY
			);
			break;
		case 2:
		 /* M2: a blank, filled circle */
			gc = segGC(st, bgPixel, LineSolid, 0, (char *) 0, 0, bgPixel );
			XFillArc( st->disp, st->win,
				gc,
				x_sizeX, y_sizeY,
				diamX, diamY, 0, 360*64
			);
			gc = segGC(st, fgPixel, LineSolid, 0, (char *) 0, 0, bgPixel );
			XDrawArc( st->disp, st->win,
				gc,
				x_sizeX, y_sizeY,
				diamX, diamY, 0, 360*64
			);
			break;
		case 3:
		 /* M3: a black, filled circle */
			gc = segGC(st, fgPixel, LineSolid, 0, (char *) 0, 0, bgPixel );
			XFillArc( st->disp, st->win,
				gc,
				x_sizeX, y_sizeY,
				diamX, diamY, 0, 360*64
			);
			break;
		case 4:
		 /* M4: a blank, filled diamond    */
		case 5:
		 /* M5: a black, filled diamond    */
			poly[0].x= x_sizeX; poly[0].y= y;
			poly[1].x= x; poly[1].y= ysizeY;
			poly[2].x= xsizeX; poly[2].y= y;
			poly[3].x= x; poly[3].y= y_sizeY;
			poly[4].x= x_sizeX; poly[4].y= y;
			if( type== 4 ){
				gc = segGC(st, bgPixel, LineSolid, 0, (char *) 0, 0, bgPixel );
				XFillPolygon(st->disp, st->win, gc,
					poly, 5, Complex, CoordModeOrigin
				);
				gc = segGC(st, fgPixel, LineSolid, 0, (char *) 0, 0, bgPixel );
				XDrawLines(st->disp, st->win, gc,
					poly, 5, CoordModeOrigin
				);
			}
			else{
				gc = segGC(st, fgPixel, LineSolid, 0, (char *) 0, 0, bgPixel );
				XFillPolygon(st->disp, st->win, gc,
					poly, 5, Complex, CoordModeOrigin
				);
			}
			break;
		case 6:
		 /* M6: a blank, filled upwards triangle   */
		case 7:{
		 /* M7: a black, filled upwards triangle   */
			poly[0].x= x; poly[0].y= y_sizeY;
			poly[1].x= xsizeX; poly[1].y= ysizeY;
			poly[2].x= x_sizeX; poly[2].y= ysizeY;
			poly[3].x= x; poly[3].y= y_sizeY;
			if( type== 6 ){
				gc = segGC(st, bgPixel, LineSolid, 0, (char *) 0, 0, bgPixel );
				XFillPolygon(st->disp, st->win, gc,
					poly, 4, Complex, CoordModeOrigin
				);
				gc = segGC(st, fgPixel, LineSolid, 0, (char *) 0, 0, bgPixel );
				XDrawLines(st->disp, st->win, gc,
					poly, 4, CoordModeOrigin
				);
			}
			else{
				gc = segGC(st, fgPixel, LineSolid, 0, (char *) 0, 0, bgPixel );
				XFillPolygon(st->disp, st->win, gc,
					poly, 4, Complex, CoordModeOrigin
				);
			}
			break;
		}
		case 8:
		 /* M8: a blank, filled downwards triangle */
		case 9:{
		 /* M9: a black, filled downwards triangle */
			poly[0].x= x; poly[0].y= ysizeY;
			poly[1].x= xsizeX; poly[1].y= y_sizeY;
			poly[2].x= x_sizeX; poly[2].y= y_sizeY;
			poly[3].x= x; poly[3].y= ysizeY;
			if( type== 8 ){
				gc = segGC(st, bgPixel, LineSolid, 0, (char *) 0, 0, bgPixel );
				XFillPolygon(st->disp, st->win, gc,
					poly, 4, Complex, CoordModeOrigin
				);
				gc = segGC(st, fgPixel, LineSolid, 0, (char *) 0, 0, bgPixel );
				XDrawLines(st->disp, st->win, gc,
					poly, 4, CoordModeOrigin
				);
			}
			else{
				gc = segGC(st, fgPixel, LineSolid, 0, (char *) 0, 0, bgPixel );
				XFillPolygon(st->disp, st->win, gc,
					poly, 4, Complex, CoordModeOrigin
				);
			}
			break;
		}
		case 10:
		 /* M10: a blank, filled diabolo   */
		case 11:
		 /* M11: a black, filled diabolo   */
			poly[0].x= x; poly[0].y= y;
			poly[1].x= x_sizeX; poly[1].y= y_sizeY;
			poly[2].x= xsizeX; poly[2].y= y_sizeY;
			poly[3].x= x; poly[3].y= y;
			if( type== 10 ){
				gc = segGC(st, bgPixel, LineSolid, 0, (char *) 0, 0, bgPixel );
				XFillPolygon(st->disp, st->win, gc,
					poly, 4, Complex, CoordModeOrigin
				);
			}
			else{
				gc = segGC(st, fgPixel, LineSolid, 0, (char *) 0, 0, bgPixel );
				XFillPolygon(st->disp, st->win, gc,
					poly, 4, Complex, CoordModeOrigin
				);
			}
			poly[0].x= x; poly[0].y= y;
			poly[1].x= x_sizeX; poly[1].y= ysizeY;
			poly[2].x= xsizeX; poly[2].y= ysizeY;
			poly[3].x= x; poly[3].y= y;
			if( type== 10 ){
				gc = segGC(st, bgPixel, LineSolid, 0, (char *) 0, 0, bgPixel );
				XFillPolygon(st->disp, st->win, gc,
					poly, 4, Complex, CoordModeOrigin
				);
				poly[0].x= x_sizeX; poly[0].y= y_sizeY;
				poly[1].x= xsizeX; poly[1].y= y_sizeY;
				poly[2].x= x_sizeX; poly[2].y= ysizeY;
				poly[3].x= xsizeX; poly[3].y= ysizeY;
				poly[4].x= x_sizeX; poly[4].y= y_sizeY;
				gc = segGC(st, fgPixel, LineSolid, 0, (char *) 0, 0, bgPixel );
				XDrawLines(st->disp, st->win, gc,
					poly, 5, CoordModeOrigin
				);
			}
			else{
				gc = segGC(st, fgPixel, LineSolid, 0, (char *) 0, 0, bgPixel );
				XFillPolygon(st->disp, st->win, gc,
					poly, 4, Complex, CoordModeOrigin
				);
			}
			break;
		case 12:
		 /* M12: a diagonal cross   */
			poly[0].x= x_sizeX; poly[0].y= y_sizeY;
			poly[1].x= xsizeX; poly[1].y= ysizeY;
			poly[2].x= x_sizeX; poly[2].y= ysizeY;
			poly[3].x= xsizeX; poly[3].y= y_sizeY;
			gc = segGC(st, fgPixel, LineSolid, 0, (char *) 0, 0, bgPixel );
			XDrawLines(st->disp, st->win, gc,
				poly, 2, CoordModeOrigin
			);
			XDrawLines(st->disp, st->win, gc,
				&poly[2], 2, CoordModeOrigin
			);
			break;
		case 13:
		 /* M13: a cross   */
			poly[0].x= x_sizeX; poly[0].y= y;
			poly[1].x= xsizeX; poly[1].y= y;
			poly[2].x= x; poly[2].y= ysizeY;
			poly[3].x= x; poly[3].y= y_sizeY;
			gc = segGC(st, fgPixel, LineSolid, 0, (char *) 0, 0, bgPixel );
			XDrawLines(st->disp, st->win, gc,
				poly, 2, CoordModeOrigin
			);
			XDrawLines(st->disp, st->win, gc,
				&poly[2], 2, CoordModeOrigin
			);
			break;
		case 14:
		 /* M14: a blank filled rectangle with a cross */
			mark_X( st, x, y, 0, colour, pixval, setnr, 1, 1 );
			mark_X( st, x, y, 12, colour, pixval, setnr, 1, 1 );
			break;
		case 15:{
/* 		 int sizex, sizey;	*/
		 /* M15: a rectangle/diamond ("star")  */
			mark_X( st, x, y, 0, colour, pixval, setnr, 1/1.3, 1/1.3 );
/* 			sizex= rd( sizeX/ 1.2 );	*/
/* 			sizey= rd( sizeY* 1.2 );	*/
			poly[0].x= x_sizeX; poly[0].y= y;
			poly[1].x= x; poly[1].y= ysizeY;
			poly[2].x= xsizeX; poly[2].y= y;
			poly[3].x= x; poly[3].y= y_sizeY;
			poly[4].x= x_sizeX; poly[4].y= y;
			gc = segGC(st, bgPixel, LineSolid, 0, (char *) 0, 0, bgPixel );
			XFillPolygon(st->disp, st->win, gc,
				poly, 5, Complex, CoordModeOrigin
			);
			gc = segGC(st, fgPixel, LineSolid, 0, (char *) 0, 0, bgPixel );
			XDrawLines(st->disp, st->win, gc,
				poly, 5, CoordModeOrigin
			);
			break;
		}
		case 16:
			mark_X( st, x, y, 2, colour, pixval, setnr, 1, 1 );
			mark_X( st, x, y, 12, colour, pixval, setnr, 1, 1 );
			break;
	}
}

static void dot_X( char *user_state, int x, int y, int style, int type, int colour, Pixel pixval, int setnr, void *set )
/*
 * This routine should draw a marker at location `x,y'.  If the
 * style is P_PIXEL,  the dot should be a single pixel.  If
 * the style is P_DOT,  the dot should be a reasonably large
 * dot.  If the style is P_MARK,  it should be a distinguished
 * mark which is specified by `type' (0-7).  If the output
 * device is capable of colour,  the marker should be drawn in
 * `colour' (0-7) which corresponds with the colour for xg_line.
 */
{
	struct x_state *st = (struct x_state *) user_state;
	extern int use_markFont;
	GC gc;

	colour= colour % MAXATTR;
	if( set ){
	  XSegment seg;
		seg.x1= x;
		seg.y1= y;
		Find_Segs_Bounds( &seg, 1, set, "xg_dot", False );
	}
	switch (style) {
		case P_PIXEL:
/* 			gc= dotGC(st, (colour< 0)? pixval : AllAttrs[colour].pixelValue, None, 0, 0);	*/
			gc = segGC(st, (colour<0)? pixval : AllAttrs[colour].pixelValue, LineSolid, 0, (char *) 0, 0, bgPixel );
			if( !st->silent ){
				XDrawPoint(st->disp, st->win, gc, x, y);
			}
			break;
		case P_DOT:
			if( VendorRelease(st->disp)== 4 && !strcmp( ServerVendor(st->disp), "Hewlett-Packard Company") ){
/* 				gc= dotGC(st, (colour< 0)? pixval : AllAttrs[colour].pixelValue, None,	*/
/* 					(int) (x - (dot_w >> 1)),	*/
/* 					(int) (y - (dot_h >> 1))	*/
/* 				);	*/
				gc = segGC(st, (colour<0)? pixval : AllAttrs[colour].pixelValue, LineSolid, 0, (char *) 0, 0, bgPixel );
				if( !st->silent ){
					XFillArc( st->disp, st->win,
						gc,
						(int) (x - (dot_w >> 1)), (int) (y - (dot_h >> 1)),
						dot_w, dot_h, 0, 360*64
					);
				}
			}
			else{
				gc= dotGC(st, (colour< 0)? pixval : AllAttrs[colour].pixelValue, dotMap,
					(int) (x - (dot_w >> 1)),
					(int) (y - (dot_h >> 1))
				);
				if( !st->silent ){
					XFillRectangle(st->disp, st->win, gc,
						(int) (x - (dot_w >> 1)), (int) (y - (dot_h >> 1)),
						dot_w, dot_h
					);
				}
			}
			break;
		case P_MARK:
			if( st->ps_marks ){
				mark_X( st, x, y, type, colour, pixval, (double) setnr, 1, 1 );
			}
			else if( use_markFont ){
			 char Mark[16];
				if( markFont.font== legendFont.font ){
					sprintf( Mark, "%d", type );
					text_X(user_state, x, y, Mark, T_CENTER, T_LABEL, NULL);
				}
				else{
					type= (type % 255) + 1;
					sprintf( Mark, "%c", type );
					text_X(user_state, x, y, Mark, T_CENTER, T_MARK, NULL);
				}
			}
			else{
				type%= MAXATTR;
				gc= dotGC(st, (colour< 0)? pixval : AllAttrs[colour].pixelValue,
					AllAttrs[type].markStyle,
					(int) (x - mark_cx),
					(int) (y - mark_cy)
				);
				if( !st->silent ){
					XFillRectangle(st->disp, st->win,
						gc,
						(int) (x - mark_cx), (int) (y - mark_cy),
						mark_w, mark_h
					);
				}
			}
			break;
	}
}

char *ParsedColourName= NULL;
#define StoreCName(name)	xfree(name);name=XGstrdup(ParsedColourName)

double IntensityRGBValues[7]= { 0,0,0, 0, -1,-1,-1 };

XColor GetThisColor, GotThisColor, ExactColour;
int GetCMapColor( char *Name, Pixel *pix, Colormap colourmap )
/*
 * Given a standard color name,  this routine fetches the associated
 * pixel value from the global `cmap'.  The name may be specified
 * using the standard specification format parsed by XParseColor.
 * Some sort of parsing error causes the program to exit.
 */
{ XColor def;
  char *c, emsg[1024], ok= 0;
  char name[256];
  static char cname[256];
  int ret= 0, store_org_name= False;

	if( !Name || !pix ){
		return(0);
	}

	emsg[0]= '\0';

	GotThisColor.pixel= -1;
	ExactColour.pixel= -1;
#ifdef DEBUG
	if( !RemoteConnection ){
		XFlush( disp );
	}
#endif

	if( Name== GETCOLOR_USETHIS ){
		def= GetThisColor;
		ExactColour= def;
		if( *AllowGammaCorrection ){
			sprintf( cname, "rgbi:%g/%g/%g", def.red/65535.0, def.green/ 65535.0, def.blue/ 65535.0 );
		}
		else{
			sprintf( cname, "rgb:%04x/%04x/%04x", def.red, def.green, def.blue );
		}
		strcpy( name, cname);
		StringCheck( name, sizeof(name)/ sizeof(char), __FILE__, __LINE__ );
		ok= 1;
	}
	else{
	  char *xg_rgbi= NULL, *xg_rgb= NULL;
		if( ! *Name ){
			return(0);
		}
		if( strncasecmp( Name, "ForeGround", 10)== 0 ){
			strncpy( name, normCName, 255 );
			store_org_name= True;
		}
		else if( strncasecmp( Name, "BackGround", 10)== 0 ){
			strncpy( name, bgCName, 255 );
			store_org_name= True;
		}
		else{
			strncpy( name, Name, 255 );
			store_org_name= False;
		}
		name[255]= '\0';
		if( *(c= &name[strlen(name)-1])== '\n' ){
			*c= '\0';
		}
		if( index( name, ',') ){
		  double rgb[3];
		  int n= 3;
			if( fascanf( &n, name, rgb, NULL, NULL, NULL, NULL)== 3 ){
				for( n= 0; n< 3; n++ ){
					CLIP( rgb[n], 0, 255 );
				}
				if( *AllowGammaCorrection ){
					sprintf( cname, "rgbi:%g/%g/%g", rgb[0]/255.0, rgb[1]/255.0, rgb[2]/255.0 );
				}
				else{
					sprintf( cname, "rgb:%04x/%04x/%04x", (int) rgb[0], (int) rgb[1], (int) rgb[2] );
				}
				strcpy( name, cname);
				StringCheck( name, sizeof(name)/ sizeof(char), __FILE__, __LINE__ );
			}
		}
		if( (xg_rgbi= strstr( name, " (rgbi:")) ){
		  /* A colour saved by xgraph, having an intensity-spec. associated with it
		   \ "name (rgbi:ir/ig/ib)"
		   */
			*xg_rgbi++ = '\0';
			if( store_org_name ){
			  char *c;
				  /* Prevent infinite repetitions:	*/
				if( (c= strstr( Name, " (rgbi:")) || (c= strstr( Name, " (rgb:")) ){
					*c++= '\0';
				}
			}
		}
		if( (xg_rgb= strstr( name, " (rgb:")) ){
		  /* A colour saved by xgraph, having an absolute (hex) colour-specification associated with it
		   \ "name (rgb:r/g/b)"
		   */
			*xg_rgb++ = '\0';
			if( store_org_name ){
			  char *c;
				if( (c= strstr( Name, " (rgbi:")) || (c= strstr( Name, " (rgb:")) ){
					*c++= '\0';
				}
			}
		}
		if( XLookupColor(disp, colourmap, name, &def, &GotThisColor ) ){
			ok= 1;
			ExactColour= def;
			if( debugFlag ){
				fprintf( StdErr, "GetColor(%s): exact (%g,%g,%g), screen (%g,%g,%g)\n",
					name,
					def.red/ 65535.0, def.green/ 65535.0, def.blue/ 65535.0,
					GotThisColor.red/ 65535.0, GotThisColor.green/ 65535.0, GotThisColor.blue/ 65535.0
				);
			}
		}
		else{
		  char *c;
			ok= 0;
			if( xg_rgbi && (c= index( xg_rgbi, ')')) && !( *AllowGammaCorrection==0 && xg_rgb) ){
			  double Intensity[3];
			  int ni;
				*c= '\0';
				sprintf( emsg, "GetColor(): cannot parse colour specification: \"%s\", trying \"",
					name
				);
				if( !*AllowGammaCorrection ){
					ni= sscanf( &xg_rgbi[1], "rgbi:%lf/%lf/%lf", &Intensity[0], &Intensity[1], &Intensity[2] );
				}
				if( *AllowGammaCorrection || ni!= 3 ){
					strcpy( name, &xg_rgbi[1] );
				}
				else{
					sprintf( name, "rgb:%04x/%04x/%04x",
						(int) (Intensity[0]* 65535+ 0.5),
						(int) (Intensity[1]* 65535+ 0.5),
						(int) (Intensity[2]* 65535+ 0.5)
					);
				}
				StringCheck( name, sizeof(name)/ sizeof(char), __FILE__, __LINE__ );
				sprintf( emsg, "%s%s\"\n", emsg, name );
				StringCheck( emsg, sizeof(emsg)/ sizeof(char), __FILE__,__LINE__ );
				if( XLookupColor(disp, colourmap, name, &def, &GotThisColor ) ){
					ok= 1;
					ExactColour= def;
				}
			}
			if( !ok && (xg_rgb && (c= index( xg_rgb, ')')) /* && !( *AllowGammaCorrection && xg_rgbi) */ ) ){
			  int rgb[3];
			  int ni;
				*c= '\0';
				sprintf( emsg, "GetColor(): cannot parse colour specification: \"%s\", trying \"",
					name
				);
				if( *AllowGammaCorrection ){
					ni= sscanf( &xg_rgb[1], "rgb:%x/%x/%x", &rgb[0], &rgb[1], &rgb[2] );
				}
				if( !*AllowGammaCorrection || ni!= 3 ){
					strcpy( name, &xg_rgb[1] );
				}
				else{
					sprintf( name, "rgbi:%g/%g/%g", rgb[0]/ 65535.0, rgb[1]/ 65535.0, rgb[2]/ 65535.0);
				}
				StringCheck( name, sizeof(name)/ sizeof(char), __FILE__, __LINE__ );
				sprintf( emsg, "%s%s\"\n", emsg, name );
				StringCheck( emsg, sizeof(emsg)/ sizeof(char), __FILE__,__LINE__ );
				if( XLookupColor(disp, colourmap, name, &def, &GotThisColor ) ){
					ok= 1;
					ExactColour= def;
				}
			}
		}
	}
	if( ok ){
	  extern int TrueGray, reverseFlag;
		if( TrueGray ){
		  int g= (int)(xtb_PsychoMetric_Gray( &def )+ 0.5);
			def.red= g;
			def.green= g;
			def.blue= g;
			ExactColour= def;
			GotThisColor.pixel= -1;
		}
		if( reverseFlag ){
			def.red= 65535- def.red;
			def.green= 65535- def.green;
			def.blue= 65535- def.blue;
			ExactColour= def;
			GotThisColor.pixel= -1;
		}
		ret= 0;
		if( install_flag== 2 ){
			if( XStoreColor( disp, colourmap, &def ) ){
				ret= 1;
				if( GotThisColor.pixel== -1 ){
					GotThisColor= def;
				}
			}
		}
		else{
			if( XAllocColor(disp, colourmap, &def)) {
				ret= 1;
				if( GotThisColor.pixel== -1 ){
					GotThisColor= def;
/*
					GotThisColor.pixel= def.pixel;
					XQueryColor( disp, colourmap, &GotThisColor );
 */
				}
			}
		}
		if( ret ){
			*pix= def.pixel;
			GetThisColor= def;
			if( *AllowGammaCorrection ){
				sprintf( cname, "%s (rgbi:%g/%g/%g)",
					(store_org_name)? Name : name, def.red/ 65535.0, def.green/65535.0, def.blue/65535.0
				);
			}
			else{
				sprintf( cname, "%s (rgb:%04x/%04x/%04x)",
					(store_org_name)? Name : name, def.red, def.green, def.blue
				);
			}
			StringCheck( cname, sizeof(cname)/ sizeof(char), __FILE__, __LINE__ );
			ParsedColourName= cname;
				IntensityRGBValues[0]= GotThisColor.red/ 65535.0;
				IntensityRGBValues[1]= GotThisColor.green/ 65535.0;
				IntensityRGBValues[2]= GotThisColor.blue/ 65535.0;
				IntensityRGBValues[3]= GotThisColor.pixel;
				IntensityRGBValues[4]= ExactColour.red/ 65535.0;
				IntensityRGBValues[5]= ExactColour.green/ 65535.0;
				IntensityRGBValues[6]= ExactColour.blue/ 65535.0;
		}
		else{
			sprintf( emsg, "GetColor(): could not store/allocate color: '%s'\n", name);
			StringCheck( emsg, sizeof(emsg)/ sizeof(char), __FILE__,__LINE__ );
		}
#ifdef DEBUG
if( normCName ){
  double r, g, b;
  char *c= strstr( normCName, "(rgbi:" );
		def.pixel= normPixel;
		XQueryColor( disp, colourmap, &def );
		if( c && sscanf( c, "(rgbi:%lf/%lf/%lf)", &r, &g, &b )== 3 ){
			if( def.red/ 65535.0!= r || def.green/ 65535.0!= g || def.blue/ 65535.0!= b ){
				fprintf( StdErr, "normPixel \"%s\" now rgbi:%g/%g/%g; allocing \"%s\"\n",
					normCName,
					def.red/ 65535.0, def.green/65535.0, def.blue/65535.0,
					cname
				);
			}
		}
		else{
			fprintf( StdErr, "normPixel \"%s\" now rgbi:%g/%g/%g; allocing \"%s\"\n",
				normCName,
				def.red/ 65535.0, def.green/65535.0, def.blue/65535.0,
				cname
			);
		}
}
#endif
    } else {
		sprintf( emsg, "%sGetColor(): cannot parse color specification: '%s'\n", emsg, name);
		StringCheck( emsg, sizeof(emsg)/ sizeof(char), __FILE__,__LINE__ );
    }
	if( emsg[0] ){
		if( ActiveWin ){
			xtb_error_box( ActiveWin->window, emsg, "Warning" );
		}
		fprintf( StdErr, "xgraph::%s", emsg );
	}
	return ret;
}

int GetColor( char *Name, Pixel *pix )
{
	return( GetCMapColor(Name, pix, cmap) );
}

void FreeCMapColor( Pixel *pix, char **cname, Colormap colourmap )
{
	if( pix ){
		XFreeColors( disp, colourmap, pix, 1, 0 );
		*pix= -1;
	}
	if( cname ){
		xfree( *cname );
	}
}

void FreeColor( Pixel *pix, char **cname )
{
	FreeCMapColor( pix, cname, cmap );
}

void RecolourCursors()
{ XColor cc_color, fg_color, bg_color;
  extern Cursor noCursor, theCursor, zoomCursor, labelCursor, cutCursor, filterCursor;
	fg_color.pixel = normPixel;
	XQueryColor(disp, cmap, &fg_color);
	bg_color.pixel = bgPixel;
	XQueryColor(disp, cmap, &bg_color);
	cc_color.pixel= zeroPixel;
	XQueryColor(disp, cmap, &cc_color);
	XRecolorCursor(disp, noCursor, &cc_color, &cc_color);
	XRecolorCursor(disp, theCursor, &fg_color, &bg_color);
	XRecolorCursor(disp, zoomCursor, &fg_color, &bg_color);
	XRecolorCursor(disp, labelCursor, &fg_color, &bg_color);
	XRecolorCursor(disp, cutCursor, &fg_color, &bg_color);
	XRecolorCursor(disp, filterCursor, &fg_color, &bg_color);
}

static void Free_Intensity_Colours();

static Colormap prev_cmap= (Colormap) NULL;

int ReallocColours(Boolean do_redraw)
{ Pixel tp;
  int set;
  LocalWindows *WL= WindowList;
/*   XColor fg_color, bg_color;	*/
  extern int Intensity_Colours();
  static char active= 0;
  XGPen *pen_list;
  Colormap new_cmap= cmap;

	if( active ){
		return(0);
	}
	active= 1;

	if( prev_cmap ){
		cmap= prev_cmap;
	}

	  /* First deallocate all...
	   \ We don't use FreeColor() since that would throw away the colour's name
	   \ too - and we need that again to reallocate it!
	   */
	XFreeColors( disp, cmap, &black_pixel, 1, 0);
	XFreeColors( disp, cmap, &white_pixel, 1, 0);
	XFreeColors( disp, cmap, &bgPixel, 1, 0);
	XFreeColors( disp, cmap, &normPixel, 1, 0);
	XFreeColors( disp, cmap, &bdrPixel, 1, 0);
	XFreeColors( disp, cmap, &zeroPixel, 1, 0);
	XFreeColors( disp, cmap, &axisPixel, 1, 0);
	XFreeColors( disp, cmap, &gridPixel, 1, 0);
	XFreeColors( disp, cmap, &highlightPixel, 1, 0);
	for( set= 0;  set< MAXATTR; set++ ){
		XFreeColors( disp, cmap, &AllAttrs[set].pixelValue, 1, 0);
	}
	for( set= 0;  set< setNumber; set++ ){
		if( AllSets[set].pixvalue< 0 ){
			XFreeColors( disp, cmap, &AllSets[set].pixelValue, 1, 0);
		}
	}
	while( WL ){
	  UserLabel *ul;
	  LocalWin *wi;
		wi= WL->wi;
		ul= wi->ulabel;
		while( ul ){
			if( ul->pixvalue< 0 && !ul->pixlinked ){
				XFreeColors( disp, wi->cmap, &ul->pixelValue, 1, 0);
			}
			ul= ul->next;
		}
		for( set= 0;  set< setNumber; set++ ){
			if( wi->legend_line[set].pixvalue< 0 ){
				XFreeColors( disp, wi->cmap, &wi->legend_line[set].pixelValue, 1, 0);
			}
		}
		if( (pen_list= wi->pen_list) ){
		  XGPenPosition *pos;
			while( pen_list ){
				if( pen_list->hlpixelCName ){
					XFreeColors( disp, wi->cmap, &pen_list->hlpixelValue, 1, 0);
				}
				pos= pen_list->position;
				for( set= 0; set< pen_list->current_pos && pos; set++ ){
					if( pos[set].colour.pixvalue< 0 ){
						if( pos[set].pixelCName ){
							XFreeColors( disp, wi->cmap, &(pos[set].colour.pixelValue), 1, 0);
						}
					}
					if( pos[set].flcolour.pixvalue< 0 ){
						if( pos[set].flpixelCName ){
							XFreeColors( disp, wi->cmap, &(pos[set].flcolour.pixelValue), 1, 0);
						}
					}
				}
				pen_list= pen_list->next;
			}
		}
		WL= WL->next;
	}
	Free_Intensity_Colours();

	cmap= new_cmap;
	prev_cmap= (Colormap) NULL;

	if( GetColor( blackCName, &tp) ){
		black_pixel= tp;
	}
	if( GetColor( whiteCName, &tp) ){
		white_pixel= tp;
	}
	if( GetColor( bgCName, &tp) ){
		bgPixel= tp;
	}
	if( GetColor( normCName, &tp) ){
		normPixel= tp;
	}
	if( GetColor( bdrCName, &tp) ){
		bdrPixel= tp;
	}
	if( GetColor( zeroCName, &tp) ){
		zeroPixel= tp;
	}
	if( GetColor( axisCName, &tp) ){
		axisPixel= tp;
	}
	if( GetColor( gridCName, &tp) ){
		gridPixel= tp;
	}
	if( GetColor( highlightCName, &tp) ){
		highlightPixel= tp;
	}
	for( set= 0;  set< MAXATTR; set++ ){
		if( GetColor( AllAttrs[set].pixelCName, &tp ) ){
			AllAttrs[set].pixelValue= tp;
		}
	}
	for( set= 0;  set< setNumber; set++ ){
		if( AllSets[set].pixvalue< 0 ){
			if( GetColor( AllSets[set].pixelCName, &tp ) ){
				AllSets[set].pixelValue= tp;
			}
		}
	}

	  /* Do window-specific things, and redraw them when so requested.
	   \ The intensity colour list should have been reallocated at this point.
	   \ But since we don't know whether it is actually going to be in use,
	   \ we will leave that to the first time a window actually needs it.
	   */
	WL= WindowList;
	while( WL ){
	  UserLabel *ul;
	  LocalWin *wi;
		wi= WL->wi;
		ul= wi->ulabel;
		while( ul ){
			if( ul->pixvalue< 0 && !ul->pixlinked ){
				if( GetCMapColor( ul->pixelCName, &tp, wi->cmap ) ){
					ul->pixelValue= tp;
				}
			}
			ul= ul->next;
		}
		for( set= 0;  set< setNumber; set++ ){
			if( wi->legend_line[set].pixvalue< 0 ){
				if( GetCMapColor( wi->legend_line[set].pixelCName, &tp, wi->cmap ) ){
					wi->legend_line[set].pixelValue= tp;
				}
			}
		}

		if( (pen_list= wi->pen_list) ){
		  XGPenPosition *pos;
		  Boolean ok= False, olok= False;
		  Pixel oltp;
			while( pen_list ){
				if( pen_list->hlpixelCName ){
					if( GetCMapColor( pen_list->hlpixelCName, &tp, wi->cmap ) ){
						pen_list->hlpixelValue= tp;
					}
				}
				pos= pen_list->position;
				for( set= 0; set< pen_list->current_pos && pos; set++ ){
					if( pos[set].colour.pixvalue< 0 ){
						if( pos[set].pixelCName ){
							if( GetCMapColor( pos[set].pixelCName, &tp, wi->cmap ) ){
								pos[set].colour.pixelValue= tp;
								ok= True;
							}
							else{
								ok= False;
							}
						}
						else if( ok ){
							pos[set].colour.pixelValue= tp;
						}
					}
					if( pos[set].flcolour.pixvalue< 0 ){
						if( pos[set].flpixelCName ){
							if( GetCMapColor( pos[set].flpixelCName, &oltp, wi->cmap ) ){
								pos[set].flcolour.pixelValue= oltp;
								olok= True;
							}
							else{
								olok= False;
							}
						}
						else if( olok ){
							pos[set].flcolour.pixelValue= oltp;
						}
					}
				}
				pen_list= pen_list->next;
			}
		}

		if( do_redraw ){
			RedrawNow( wi );
		}
		else{
			wi->redraw= 1;
		}
		WL= WL->next;
	}

	xtb_init(disp, screen, black_pixel, white_pixel, dialogFont.font, dialog_greekFont.font, True );
	xtb_init(disp, screen, normPixel, bgPixel, dialogFont.font, dialog_greekFont.font, False );

	RecolourCursors();
	active= 0;
	return(0);
}

extern double *ascanf_IntensityColours;

static void Free_Intensity_Colours()
{ int i;
	if( IntensityColourFunction.NColours && IntensityColourFunction.XColours ){
#if 0
	  ALLOCA( pixx, unsigned long, 8192, pixx_len);
		  /* RJB990809: For some reason, on my O2, freeing 65535 colours on
		   \ a 24bit DirectColor visual hangs the process: it mews something
		   \ about an invalid request, and then ends up in a WaitRequest. In
		   \ any case, this happens when doing the CleanUp at program exit. So
		   \ I implemented this workaround.. we free the colours in chunks of
		   \ 8192 at a time.
		   */
		if( pixx )
#else
	  unsigned long pixx[8192];
#endif
		{
		  int j, NC= IntensityColourFunction.NColours;
			TitleMessage( ActiveWin, "Collecting old intensity colourmap entries...\n");
			for( j= 0, i= 0; i< IntensityColourFunction.NColours; i++, j++ ){
				if( j== 8192 ){
					TitleMessage( ActiveWin, "Freeing old intensity colourmap entries...\n");
					XFreeColors( disp, cmap, pixx, j, 0);
					TitleMessage( ActiveWin, NULL );
					j= 0;
					NC-= 8192;
				}
				pixx[j]= IntensityColourFunction.XColours[i].pixel;
			}
			if( NC ){
				TitleMessage( ActiveWin, "Freeing old intensity colourmap entries...\n");
				XFreeColors( disp, cmap, pixx, NC, 0);
			}
			GCA();
		}
#if 0
		else{
			TitleMessage( ActiveWin, "Freeing old intensity colourmap entries...\n");
			for( i= 0; i< IntensityColourFunction.NColours; i++ ){
				XFreeColors( disp, cmap, &IntensityColourFunction.XColours[i].pixel, 1, 0);
			}
			xtb_XSync( disp, False );
		}
#endif
		  /* RJB990809: Just set NColours to 0, don't de-allocate XColours (leave that to realloc.. might
		   \ be more efficient). Caveat: call this routine only when changing the intensity
		   \ colourmap, not completely removing it!
		   */
		*ascanf_IntensityColours= IntensityColourFunction.NColours= 0;
		  /* RJB990809: removed call to ReallocColours, as we are sure only to be called
		   \ by a call that will call ReallocColours..
		   */
	}
}

int Intensity_Colours( char *exp )
{ static char active= 0;
  struct Compiled_Form *C_exp= NULL;
  int ok;
  extern double *param_scratch;
  extern int ascanf_propagate_events, ascanf_arg_error;
  extern Window ascanf_window;
  extern char *TBARprogress_header;
	if( active ){
		return(0);
	}
	if( (!exp || !*exp) && IntensityColourFunction.expression && !IntensityColourFunction.use_table ){
		exp= IntensityColourFunction.expression;
	}
	if( exp && !IntensityColourFunction.use_table ){
	  int n= 4, i;
		active= 1;
		clean_param_scratch();
		if( fascanf( &n, exp, param_scratch, NULL, NULL, NULL, &C_exp ) && n== 4 ){
		  int ape= ascanf_propagate_events;
		  Window aw= ascanf_window;
			n= 1;
			param_scratch[0]= 0;
			TBARprogress_header= "*INTENSITY_COLOURS*";
			ascanf_propagate_events= 0;
			if( compiled_fascanf( &n, exp, param_scratch, NULL, NULL, NULL, &C_exp) && n== 1 &&
					param_scratch[0]>= 0
			){
				if( exp!= IntensityColourFunction.expression ){
					xfree( IntensityColourFunction.expression );
				}
				Destroy_Form( &IntensityColourFunction.C_expression );
				Free_Intensity_Colours();
				if( !ascanf_window && ActiveWin ){
					ascanf_window= ActiveWin->window;
				}
				n= (int) param_scratch[0];
				if( n> (1 << depth) ){
				  char emsg[128];
					n= 1 << depth;
					sprintf( emsg, "Intensity_Colours(): N=%g >= %d maxcolours on this screen: corrected.\n",
						param_scratch[0], n
					);
					if( WindowList && WindowList->wi ){
						xtb_error_box( WindowList->wi->window, emsg, "Warning" );
					}
					else{
						fputs( emsg, StdErr );
					}
				}
				add_process_hist( exp );
				IntensityColourFunction.exactRGB= XGrealloc( IntensityColourFunction.exactRGB, n* sizeof(XColor));
				if( (IntensityColourFunction.XColours= XGrealloc( IntensityColourFunction.XColours, n* sizeof(XColor))) ){
				  double data[ASCANF_DATA_COLUMNS];
				  int column[ASCANF_DATA_COLUMNS]= {0,1,2,3};
				  int NC= n, f= 0;
				  unsigned long *pmr= NULL, *pixels= NULL;

					ReallocColours(False);

					*ascanf_IntensityColours= NC;
					IntensityColourFunction.NColours= 0;
					if( exp!= IntensityColourFunction.expression ){
						IntensityColourFunction.expression= strdup(exp);
					}
					IntensityColourFunction.C_expression= C_exp;
					param_scratch[1]= param_scratch[2]= param_scratch[3]= 0;
					if( debugFlag ){
						fprintf( StdErr, "IntensityColours{%s}:", exp );
						fflush( StdErr );
					}
					  /* Don't try to do smart with r/w colourtables -- too difficult */
/* 					pmr= XGrealloc( pmr, NC* sizeof(unsigned long)), pixels= XGrealloc( pixels, NC* sizeof(unsigned long));	*/
					if( pmr && pixels ){
						for( i= 0; i< NC; i++ ){
							pixels[i]= (unsigned long) i;
						}
						if( !XAllocColorCells( disp, cmap, False, pmr, 0, pixels, NC ) ){
							xfree( pmr );
							xfree( pixels );
						}
						else if( debugFlag ){
							fprintf( StdErr, "[Great - got a r/w colourtable!]\n" );
						}
					}
					else{
						xfree( pmr );
						xfree( pixels );
					}
					for( i= 0; i< NC; IntensityColourFunction.NColours++, i++ ){
						param_scratch[0]= i;
						data[0]= data[1]= data[2]= data[3]= i;
						n= 4;
						compiled_fascanf( &n, exp, param_scratch, NULL, data, column, &C_exp );
						if( !ascanf_arg_error && n ){
						  Pixel pix;
						  double r, g, b;
							CLIP_EXPR( r, 65535* param_scratch[1], 0, 65535 );
							CLIP_EXPR( g, 65535* param_scratch[2], 0, 65535 );
							CLIP_EXPR( b, 65535* param_scratch[3], 0, 65535 );
							GetThisColor.red= (unsigned short) r;
							GetThisColor.green= (unsigned short) g;
							GetThisColor.blue= (unsigned short) b;
							GetThisColor.flags= DoRed|DoGreen|DoBlue;
							if( IntensityColourFunction.exactRGB ){
								IntensityColourFunction.exactRGB[i].red= GetThisColor.red;
								IntensityColourFunction.exactRGB[i].green= GetThisColor.green;
								IntensityColourFunction.exactRGB[i].blue= GetThisColor.blue;
							}
							if( pixels ){
								GetThisColor.pixel= pixels[i];
								IntensityColourFunction.XColours[i]= GetThisColor;
							}
							else{
								GetThisColor.pixel= i;
								if( debugFlag ){
									fprintf( StdErr, " (%g,%g,%g)", param_scratch[1], param_scratch[2], param_scratch[3] );
									if( debugLevel ){
										fprintf( StdErr, "=(%hu,%hu,%hu)",
											GetThisColor.red, GetThisColor.green, GetThisColor.blue
										);
									}
									else{
										fflush( StdErr );
									}
								}
								if( GetColor( GETCOLOR_USETHIS, &pix) ){
									IntensityColourFunction.XColours[i]= GetThisColor;
									if( debugFlag && debugLevel ){
										fprintf( StdErr, ">(%hu,%hu,%hu)",
											GetThisColor.red, GetThisColor.green, GetThisColor.blue
										);
										fflush( StdErr );
									}
								}
								else if( debugFlag ){
									fprintf( StdErr, "[can't]");
									fflush( StdErr );
									f+= 1;
								}
							}
						}
					}
					if( pixels ){
						XStoreColors( disp, cmap, IntensityColourFunction.XColours, IntensityColourFunction.NColours );
						xfree( pmr );
						xfree( pixels );
					}
					else if( debugFlag ){
						fprintf( StdErr, " -- %d colours, %d failures\n", i- f, f );
					}
				}
			}
			TBARprogress_header= NULL;
			ascanf_propagate_events= ape;
			ascanf_window= aw;
			ok= 1;
		}
		else{
			Destroy_Form( &C_exp );
			ok= 0;
		}
		active= 0;
		return(ok);
	}
	else if( IntensityColourFunction.name_table && IntensityColourFunction.use_table ){
	  XGStringList *tab= IntensityColourFunction.name_table;
	  int i, NC= 0;
		while( tab ){
			NC+= 1;
			tab= tab->next;
		}
		if( NC> (1 << depth) ){
		  char emsg[128];
			NC= 1 << depth;
			sprintf( emsg, "Intensity_Colours(): N=%g >= %d maxcolours on this screen: corrected.\n",
				param_scratch[0], NC
			);
			if( WindowList && WindowList->wi ){
				xtb_error_box( WindowList->wi->window, emsg, "Warning" );
			}
			else{
				fputs( emsg, StdErr );
			}
		}
		IntensityColourFunction.exactRGB= XGrealloc( IntensityColourFunction.exactRGB, NC* sizeof(XColor));
		if( NC> 0 && (IntensityColourFunction.XColours= XGrealloc( IntensityColourFunction.XColours, NC* sizeof(XColor))) ){
		  int f= 0;

			ReallocColours(False);

			*ascanf_IntensityColours= NC;
			IntensityColourFunction.NColours= 0;
			if( debugFlag ){
				fprintf( StdErr, "IntensityColours{}:" );
			}
			tab= IntensityColourFunction.name_table;
			for( i= 0; i< NC; IntensityColourFunction.NColours++, i++ ){
			  Pixel pix;
				if( debugFlag ){
					fprintf( StdErr, " \"%s\"", tab->text );
				}
				if( GetColor( tab->text, &pix) ){
					IntensityColourFunction.XColours[i]= GetThisColor;
					StoreCName(tab->text);
					  /* RJVB 20030829: in this case, we only know the ExactColour co-ordinates *after*
					   \ having done a lookup. Since tab->text can be anything.
					   */
					if( IntensityColourFunction.exactRGB ){
						IntensityColourFunction.exactRGB[i].red= ExactColour.red;
						IntensityColourFunction.exactRGB[i].green= ExactColour.green;
						IntensityColourFunction.exactRGB[i].blue= ExactColour.blue;
					}
				}
				else if( debugFlag ){
					fprintf( StdErr, "[can't]");
					f+= 1;
				}
				tab= tab->next;
			}
			if( debugFlag ){
				fprintf( StdErr, " -- %d colours, %d failures\n", i- f, f );
			}
		}
		return(1);
	}
	else if( IntensityColourFunction.NColours && IntensityColourFunction.XColours ){
		XG_error_box( &ActiveWin, "Warning",
			"No colourdefinitions, or no valid indication of which definition to use\nFreeing existing colours\n", NULL
		);
		Free_Intensity_Colours();
		if( IntensityColourFunction.XColours ){
			xfree( IntensityColourFunction.XColours );
			ReallocColours(False);
		}
		xfree( IntensityColourFunction.exactRGB );
	}
	return(0);
}

void Default_Intensity_Colours()
{
  char defCTab[]= "IDict[DCL[NN,64]], div[$DATA{0},sub[NN,1]],div[$DATA{1},sub[NN,1]],div[$DATA{2},sub[NN,1]] @";
	if( IntensityColourFunction.use_table ){
	  int i, g;
	  char cbuf[64];
		for( i= 0; i< 64; i++ ){
			g= (int) (i* 65535.0/63.0);
			sprintf( cbuf, "#%04x%04x%04x", g, g, g);
			IntensityColourFunction.name_table=
				XGStringList_AddItem( IntensityColourFunction.name_table, cbuf );
		}
		Intensity_Colours( NULL );
		XG_error_box( &ActiveWin, "Warning: default INTENSITY colourtable has been set corresponding to:", defCTab, NULL );
	}
	else{
		Intensity_Colours( defCTab );
		XG_error_box( &ActiveWin, "Warning: default INTENSITY colours have been set:", defCTab, NULL );
	}
}

int ascanf_ClearWindow ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
	if( !ascanf_SyntaxCheck && ActiveWin ){
		if( ascanf_arguments>= 4 ){
		  XRectangle *rect= rect_diag2xywh(
				SCREENX( ActiveWin, args[0] ), SCREENY( ActiveWin, args[1] ),
				SCREENX( ActiveWin, args[2] ), SCREENY( ActiveWin, args[3] ) );
			ActiveWin->dev_info.xg_clear(ActiveWin->dev_info.user_state,
				rect->x, rect->y,
				rect->width, rect->height, 0, 0
			);
		}
		else{
			ActiveWin->dev_info.xg_clear(ActiveWin->dev_info.user_state, 0, 0,
				ActiveWin->dev_info.area_w, ActiveWin->dev_info.area_h, 0, 0
			);
		}
		ActiveWin->redraw= True;
		*result= 1;
	}
	else{
		*result= 0;
	}
	return(1);
}

int ascanf_GetRGB ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  extern int ascanf_arguments, ascanf_arg_error;
  extern char *ascanf_emsg;
  XColor rgb;
	if( ascanf_arguments< 1 || !args ){
		ascanf_arg_error= 1;
		return(0);
	}
	if( args[0]< 0 || args[0]> 65535 ){
		ascanf_emsg= "(colourindex range error)";
		ascanf_arg_error= 1;
		return(0);
	}
	rgb.pixel= (int) args[0];
	XQueryColor( disp, cmap, &rgb );
	IntensityRGBValues[0]= rgb.red/ 65535.0;
	IntensityRGBValues[1]= rgb.green/ 65535.0;
	IntensityRGBValues[2]= rgb.blue/ 65535.0;
	IntensityRGBValues[3]= rgb.pixel;
	IntensityRGBValues[4]= ExactColour.red/ 65535.0;
	IntensityRGBValues[5]= ExactColour.green/ 65535.0;
	IntensityRGBValues[6]= ExactColour.blue/ 65535.0;
	*result= xtb_PsychoMetric_Gray( &rgb);
	return(1);
}

int ascanf_GetIntensityRGB ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  extern int ascanf_arguments, ascanf_arg_error;
  extern char *ascanf_emsg;
  XColor rgb;
	if( ascanf_arguments< 1 || !args ){
		ascanf_arg_error= 1;
		return(0);
	}
	if( args[0]< 0 || args[0]>= IntensityColourFunction.NColours ){
		ascanf_emsg= "(intensity range error)";
		ascanf_arg_error= 1;
		return(0);
	}
	rgb= IntensityColourFunction.XColours[(int) args[0]];
	XQueryColor( disp, cmap, &rgb );
	IntensityRGBValues[0]= rgb.red/ 65535.0;
	IntensityRGBValues[1]= rgb.green/ 65535.0;
	IntensityRGBValues[2]= rgb.blue/ 65535.0;
	IntensityRGBValues[3]= rgb.pixel;
	if( pragma_unlikely(ascanf_verbose) ){
		fprintf( StdErr, " INTENSITY[%d]=(%uh,%uh,%uh) => RGB (%uh,%uh,%uh)",
			(int) args[0], IntensityColourFunction.XColours[(int) args[0]].red,
			IntensityColourFunction.XColours[(int) args[0]].green,
			IntensityColourFunction.XColours[(int) args[0]].blue,
			rgb.red, rgb.green, rgb.blue
		);
	}
	if( IntensityColourFunction.exactRGB ){
		IntensityRGBValues[4]= IntensityColourFunction.exactRGB[(int)args[0]].red/ 65535.0;
		IntensityRGBValues[5]= IntensityColourFunction.exactRGB[(int)args[0]].green/ 65535.0;
		IntensityRGBValues[6]= IntensityColourFunction.exactRGB[(int)args[0]].blue/ 65535.0;
		if( pragma_unlikely(ascanf_verbose) ){
			fprintf( StdErr, ", exact (%uh,%uh,%uh) ",
				IntensityColourFunction.exactRGB[(int)args[0]].red,
				IntensityColourFunction.exactRGB[(int)args[0]].green,
				IntensityColourFunction.exactRGB[(int)args[0]].blue
			);
		}
	}
	else{
		IntensityRGBValues[4]= -1;
		IntensityRGBValues[5]= -1;
		IntensityRGBValues[6]= -1;
	}
	IntensityColourFunction.last_read= (int) args[0];
	*result= xtb_PsychoMetric_Gray( &rgb);
	return(1);
}

char *VisualClass[]= { "StaticGray", "GrayScale", "StaticColor", "PseudoColor", "TrueColor", "DirectColor" };
int VisualClasses= 6;

void XG_choose_visual()
{ int vm;
  char vmsg[256];
	if( !prev_cmap ){
		prev_cmap= cmap;
	}
	vm= ux11_std_vismap(disp, &vis, &cmap, &screen, &depth, True /* search_vislist */ );
	if( vm== UX11_DEFAULT && ux11_vis_class>= 0 && ux11_min_depth> DefaultDepth(disp,screen) ){
	  int uvc= ux11_vis_class;
		ux11_vis_class= -1;
		fprintf( StdErr, "[*** Disregarding visual type request ***]\n" );
		vm= ux11_std_vismap(disp, &vis, &cmap, &screen, &depth, True /* search_vislist */ );
		ux11_vis_class= uvc;
	}
	if( vm== UX11_ALTERNATE || vm== UX11_ALTERNATE_RW ){
		if( ux11_vis_class>= 0 && vis->class!= ux11_vis_class ){
			sprintf( vmsg, "[** Warning: requested visual type %s with depth>=%d not found on %s **]\n",
				VisualClass[ux11_vis_class], ux11_min_depth, DisplayString(disp)
			);
			add_comment(vmsg);
			fputs( vmsg, StdErr );
		}
		sprintf( vmsg, "[Opened %d planes deep %s visual 0x%lx%s]\n",
			depth, VisualClass[vis->class], vis->visualid,
			(ux11_useDBE)? "; using DoubleBuffer extension" : ""
		);
		add_comment( vmsg );
		fputs( vmsg, StdErr );
		install_flag = (vm== UX11_ALTERNATE_RW)? 2 : 1;
	} else {
		sprintf( vmsg, "[Opened %d planes deep %s visual 0x%lx%s]\n",
			depth, VisualClass[vis->class], vis->visualid,
			(ux11_useDBE)? "; using DoubleBuffer extension" : ""
		);
		add_comment( vmsg );
		install_flag = 0;
	}
}

int XErrorHandled= 0, GCError= 0;

/*ARGSUSED*/
int XErrHandler( Display *disp, XErrorEvent *evt )
/*
 * Displays a nicely formatted message.
 */
{  static XErrorEvent last_error;
   char *eb= NULL;
	last_error.serial= evt->serial;
	if( debugFlag ){
		fprintf(StdErr, "xgraph$%d::XErrHandler: X Error: %s", getpid(), (eb= ux11_error(evt)) );
	}
	else{
		if( memcmp( &last_error, evt, sizeof(XErrorEvent)) ){
			fprintf(StdErr, "xgraph$%d::XErrHandler: X Error: %s", getpid(), (eb= ux11_error(evt)) );
			memcpy( &last_error, evt, sizeof(XErrorEvent));
		}
	}
	if( eb ){
		if( strstr( eb, "BadGC" ) ){
			GCError+= 1;
		}
		xfree( eb );
	}
	XErrorHandled+= 1;
    return(0);
}

int XFatalHandler( Display *disp )
{
	fprintf( StdErr, "xgraph$%d::XFatalHandler: Fatal X Error on display '%s': %s\n", getpid(),
		DisplayString(disp), serror()
	);
	abort();
	return(1);
}

int HardSynchro= False, Synchro_State= 0;

void* X_Synchro(LocalWin *wi)
{
	if( HardSynchro ){
		Synchro_State= True;
	}
	else{
		Synchro_State= !Synchro_State;
	}
	if( wi && wi->SD_Dialog ){
		set_changeables(2,False);
	}
	return( (void*) XSynchronize( disp, Synchro_State ) );
}

extern char RememberedPrintFont[256];
extern double RememberedPrintSize;
extern XFontStruct *def_font;

char *ParseFontName( char *Name )
{ char *name, xres[32], yres[32];
	sprintf( xres, "-%d-", XG_DisplayXRes( disp, screen ) );
	StringCheck( xres, sizeof(xres)/ sizeof(char), __FILE__, __LINE__ );
	sprintf( yres, "-%d-", XG_DisplayYRes( disp, screen ) );
	StringCheck( yres, sizeof(yres)/ sizeof(char), __FILE__, __LINE__ );
	name= SubstituteOpcodes( Name, "-XDPI-", xres, "-YDPI-", yres, NULL );
	return( name );
}

char *GetFont( XGFontStruct *Font, char *resource_name, char *default_font, long size, int bold, int use_remembered )
{  char *font_name, *rfont_name= NULL;
   static char called= 0;
	if( !Font->font){
	  ALLOCA( DEFFont, char, (default_font)? strlen(default_font)+ 1 : 1, df_len);
	  char *DefFont= DEFFont;
		if( default_font ){
			strcpy( DefFont, default_font );
			DefFont= ParseFontName( DefFont );
			font_name= DefFont;
		}
		else{
			font_name= NULL;
		}
		RememberedPrintFont[0]= '\0';
		RememberedPrintSize= -1;
		  /* 20010530: added check for resource_name non-NULL: */
		if( resource_name && rd_font(resource_name, &font_name )) {
			Font->font = def_font;
			strncpy( Font->name, font_name, 127);
			  /* I'd say we don't want/need to remember a font
			   \ specified by the resources.
			RememberFont( disp, Font, font_name);
			   */
		}
		else if( !( use_remembered && RememberedFont( disp, Font, &rfont_name) ) ){
			if( font_name){
			  int again= 0;
				do{
					Font->font= XLoadQueryFont( disp, font_name );
					again= 0;
					if( !Font->font ){
						fprintf(StdErr, "Can't find %s font %s @%ld\n",
							FontName(Font), font_name, size
						);
						if( font_name[0]== '-' && font_name[1]!= '*' ){
						  /* 20001216: a foundry specification that is not a wildcard. Try it with a
						   \ wildcard.
						   */
						  char *fn= &font_name[1];
							while( *fn && *fn!= '-' && *fn!= '*' ){
								fn++;
							}
							switch( *fn ){
								case '*':
									strcpy( &font_name[1], fn );
									again= 1;
									break;
								case '-':
									font_name[1]= '*';
									strcpy( &font_name[2], fn );
									again= 1;
									break;
							}
						}
					}
				} while( again );
				if( !Font->font && use_remembered ){
					fprintf(StdErr, "Can't find %s font %s - trying RememberedFont or suitable default (sized %ld mu)\n",
						FontName(Font), font_name, size
					);
					fflush( StdErr );
					if( !RememberedFont( disp, Font, &rfont_name) ){
						goto load_default;
					}
				}
				if( Font->font ){
					strncpy( Font->name, font_name, 127);
					if( use_remembered ){
						RememberFont( disp, Font, font_name);
					}
				}
			}
			else{
				printf( "GetFont(%s): finding suitable default %ld mu high",
					FontName(Font), size
				);
				if( !(debugFlag && debugLevel== -3) ){
					fflush( stdout );
				}
				else{
					fputc( '\n', stdout);
				}
load_default:;
				if (!ux11_size_font(disp, screen, size, &Font->font, &font_name, bold)) {
					fprintf(StdErr, "Can't find an appropriate default %s\n", FontName( Font) );
					fflush( StdErr );
					exit(10);
				}
				else{
					strncpy( Font->name, font_name, 127);
					RememberFont( disp, Font, font_name);
					if( !(debugFlag && debugLevel== -3) ){
						printf( " \"%s\"", font_name );
						if( !called ){
							printf( " (set -db-3 option for more details)" );
							called= 1;
						}
						fputc( '\n', stdout );
					}
				}
			}
		}
		if( debugFlag && debugLevel== -3 ){
		  char buf[2]= {0,0};
			buf[0]= Font->font->default_char;
			fprintf( StdErr, "%s: characters #%u to #%u, default_char='%s'%s\n",
				Font->name,
				Font->font->min_char_or_byte2, Font->font->max_char_or_byte2,
				(buf[0])? buf : "\\0",
				(buf[0])? "" : " (uses ' ' (space))"
			);
		}

		if( DefFont!= DEFFont ){
			xfree( DefFont );
		}

		return( rfont_name? rfont_name : font_name );
    }
	return( (*(Font->name))? Font->name : NULL );
}

void UnGetFont( XGFontStruct *Font, char *resource_name)
{
	if( debugFlag && debugLevel== -3 ){
		fprintf( StdErr, "UnGetFont('%s',%s)\n",
			Font->name, resource_name
		);
		fflush( StdErr );
	}
	if( Font && Font->font ){
		XFreeFont( disp, Font->font );
		  /* 990412	*/
		Font->font= NULL;
	}
}

int GetPointSize( char *fname, int *pxsize, int *ptsize, int *xres, int *yres )
{ char *ff, *lf;
  int n;
	ff= lf= fname;
	while( lf && *lf && !( *lf== '-' && lf[1]== '-' ) ){
		lf++;
	}
	if( lf && (n= sscanf( lf, "--%d-%d-%d-%d-", pxsize, ptsize, xres, yres ))>= 1 ){
		return(n);
	}
	else{
		return(0);
	}
}

char X11_greek_template[]= "-adobe-symbol-medium-r-normal--%d-%d-%d-%d-p-0-*-*";

XFontStruct *Find_greekFont( char *Name, char *greek )
{ XFontStruct *tempFont= NULL;
  int lf_ptsz= 0, lf_pxsz= 0, lf_xres= 0, lf_yres= 0, n;
	if( greek ){
	  char *name= ParseFontName( Name );
		if( (n= GetPointSize( name, &lf_pxsz, &lf_ptsz, &lf_xres, &lf_yres )) ){
			sprintf( greek, X11_greek_template, lf_pxsz, lf_ptsz, lf_xres, lf_yres );
			if( !(tempFont= XLoadQueryFont( disp, greek )) ){
				sprintf( greek, "-*-symbol-medium-r-normal--%d-%d-%d-%d-p-0-*-*", lf_pxsz, lf_ptsz, lf_xres, lf_yres );
				if( !(tempFont= XLoadQueryFont( disp, greek )) ){
				  int pt= (lf_pxsz)? lf_pxsz : lf_ptsz/ 10;
					fprintf( StdErr, "Find_greekFont(%s): no font matching '%s', trying simpler pattern...\n",
						name, greek
					);
					sprintf( greek, "*-symbol*--%d-*", pt);
					if( !(tempFont= XLoadQueryFont( disp, greek )) ){
						sprintf( greek, "*-symbol*--%d-*", pt+ 1 );
						if( !(tempFont= XLoadQueryFont( disp, greek )) ){
							sprintf( greek, "*-symbol*--%d-*", pt- 1 );
							tempFont= XLoadQueryFont( disp, greek );
						}
					}
				}
			}
		}
		if( name!= Name ){
			xfree( name );
		}
	}
	return( tempFont );
}

int Update_greekFonts(long which)
{ char gf[256];
  XFontStruct *tempFont;
  int r= 0;
	switch( which ){
		case (long) 'TITL':
			tempFont= Find_greekFont( titleFont.name, gf );
			if( tempFont ){
				UnGetFont( &title_greekFont, "title_GreekFont" );
				title_greekFont.font= tempFont;
				strcpy( title_greekFont.name, gf );
				r= 1;
			}
			else if( debugFlag ){
				fprintf( StdErr, "Update_greekFonts(TITL): can't find greek font \"%s\" matching \"%s\"\n",
					gf, titleFont.name
				);
				fflush( StdErr );
			}
			break;
		case (long) 'LEGN':
			tempFont= Find_greekFont( legendFont.name, gf );
			if( tempFont ){
				UnGetFont( &legend_greekFont, "legend_GreekFont" );
				legend_greekFont.font= tempFont;
				strcpy( legend_greekFont.name, gf );
				r= 1;
			}
			else if( debugFlag ){
				fprintf( StdErr, "Update_greekFonts(TITL): can't find greek font \"%s\" matching \"%s\"\n",
					gf, legendFont.name
				);
				fflush( StdErr );
			}
			break;
		case (long) 'LABL':
			tempFont= Find_greekFont( labelFont.name, gf );
			if( tempFont ){
				UnGetFont( &label_greekFont, "label_GreekFont" );
				label_greekFont.font= tempFont;
				strcpy( label_greekFont.name, gf );
				r= 1;
			}
			else if( debugFlag ){
				fprintf( StdErr, "Update_greekFonts(LABL): can't find greek font \"%s\" matching \"%s\"\n",
					gf, labelFont.name
				);
				fflush( StdErr );
			}
			break;
		case (long) 'DIAL':
			tempFont= Find_greekFont( dialogFont.name, gf );
			if( tempFont ){
				UnGetFont( &dialog_greekFont, "dialog_GreekFont" );
				dialog_greekFont.font= tempFont;
				strcpy( dialog_greekFont.name, gf );
				r= 1;
			}
			else if( debugFlag ){
				fprintf( StdErr, "Update_greekFonts(DIAL): can't find greek font \"%s\" matching \"%s\"\n",
					gf, dialogFont.name
				);
				fflush( StdErr );
			}
			break;
		case (long) 'AXIS':
			tempFont= Find_greekFont( axisFont.name, gf );
			if( tempFont ){
				UnGetFont( &axis_greekFont, "axis_GreekFont" );
				axis_greekFont.font= tempFont;
				strcpy( axis_greekFont.name, gf );
				r= 1;
			}
			else if( debugFlag ){
				fprintf( StdErr, "Update_greekFonts(AXIS): can't find greek font \"%s\" matching \"%s\"\n",
					gf, axisFont.name
				);
				fflush( StdErr );
			}
			break;
		case (long) 'FDBK':
			tempFont= Find_greekFont( fbFont.name, gf );
			if( tempFont ){
				UnGetFont( &fb_greekFont, "fb_GreekFont" );
				fb_greekFont.font= tempFont;
				strcpy( fb_greekFont.name, gf );
				r= 1;
			}
			else if( debugFlag ){
				fprintf( StdErr, "Update_greekFonts(FDBK): can't find greek font \"%s\" matching \"%s\"\n",
					gf, fbFont.name
				);
				fflush( StdErr );
			}
			break;
	}
	return(r);
}

CustomFont *Init_CustomFont( char *xfn, char *axfn, char *psfn, double pssize, int psreencode )
{ XFontStruct *tempFont;
  XGFontStruct tF;
  CustomFont *cf= NULL;
  Boolean alternate= False;
	if( xfn && psfn && pssize>= 0 ){
		memset( &tF, 0, sizeof(tF) );
		GetFont( &tF, NULL, xfn, 0, 0, 0 );
		if( !(tempFont= tF.font /* XLoadQueryFont(disp, xfn) */ ) ){
			fprintf( StdErr, "Init_CustomFont(): can't get CustomFont X component '%s'\n", xfn);
			if( axfn ){
				GetFont( &tF, NULL, axfn, 0, 0, 0 );
			}
			if( axfn && (tempFont= tF.font /* XLoadQueryFont(disp, axfn) */ ) ){
				fprintf( StdErr, "Init_CustomFont(): using alternate CustomFont X component '%s' instead\n", axfn);
				alternate= True;
			}
			else{
				fprintf( StdErr, "Init_CustomFont(): can't get alternate CustomFont X component '%s' neither\n", axfn);
				return( NULL );
			}
		}
		if( tempFont ){
		 char gf[256];
		 extern XFontStruct *Find_greekFont();
			if( !(cf= (CustomFont*) calloc( 1, sizeof(CustomFont))) ){
				fprintf( StdErr, "Init_CustomFont(): can't allocate CustomFont structure (%s)\n", serror() );
				XFreeFont( disp, tempFont );
				return(NULL);
			}
			cf->XFont.font= tempFont;
			strncpy( cf->XFont.name, xfn, 127);
			if( axfn ){
				cf->alt_XFontName= strdup(axfn);
			}
			cf->is_alt_XFont= alternate;
			if( !(tempFont= Find_greekFont( xfn, gf )) && axfn ){
				tempFont= Find_greekFont( axfn, gf);
			}
			if( tempFont ){
				cf->X_greekFont.font= tempFont;
				strcpy( cf->X_greekFont.name, gf );
			}
			else if( debugFlag ){
				fprintf( StdErr, "Init_CustomFont(): can't find greek font \"%s\" matching \"%s\"\n",
					gf, cf->X_greekFont.name
				);
				fflush( StdErr );
			}
		}
		cf->PSFont= strdup( psfn );
		cf->PSPointSize= pssize;
		cf->PSreencode= psreencode;
	}
	return( cf );
}

void Free_CustomFont( CustomFont *cf )
{
	if( !cf ){
		return;
	}
	if( cf->XFont.font ){
		XFreeFont( disp, cf->XFont.font );
		cf->XFont.font= NULL;
	}
	if( cf->X_greekFont.font ){
		XFreeFont( disp, cf->X_greekFont.font );
		cf->X_greekFont.font= NULL;
	}
	xfree( cf->PSFont );
	xfree( cf->alt_XFontName );
}

void SetWindows_Cursor( Cursor curs )
{ LocalWindows *WL= WindowList;
	while( WL ){
		XDefineCursor(disp, WL->wi->window, curs);
		WL= WL->next;
	}
}

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
#define R_UsrOrgX	win_geo.R_UsrOrgX
#define R_UsrOrgY	win_geo.R_UsrOrgY
#define R_UsrOppX	win_geo.R_UsrOppX
#define R_UsrOppY	win_geo.R_UsrOppY
#define XUnitsPerPixel	win_geo._XUnitsPerPixel
#define YUnitsPerPixel	win_geo._YUnitsPerPixel
#define R_XUnitsPerPixel	win_geo.R_XUnitsPerPixel
#define R_YUnitsPerPixel	win_geo.R_YUnitsPerPixel

#define TRANX(xval) \
(((double) ((xval) - wi->XOrgX)) * wi->XUnitsPerPixel + wi->UsrOrgX)

#define TRANY(yval) \
(wi->UsrOppY - (((double) ((yval) - wi->XOrgY)) * wi->YUnitsPerPixel))

int Allow_DrawCursorCrosses= True;

/* 20041217: improved. Now takes either the current co-ordinates in ActiveWin, OR,
 \ it will find the window in WindowList that actually contains the pointer.
 */
void DrawCursorCrosses( XEvent *evt, LocalWin *wi )
{ extern int CursorCross;
  Window dum;
  int dum2, curX, curY;
  unsigned int mask_rtn;
  double cx, cy;
	if( !Allow_DrawCursorCrosses || !CursorCross ){
		return;
	}

	if( !wi ){
		if( ActiveWin ){
			wi= ActiveWin;
			XQueryPointer(disp, wi->window, &dum, &dum, &dum2, &dum2,
				  &curX, &curY, &mask_rtn
			);
		}
		else{
		  LocalWindows *WL= WindowList;
		  int r;
			while( WL ){
				wi= WL->wi;
				r= XQueryPointer(disp, wi->window, &dum, &dum, &dum2, &dum2,
					  &curX, &curY, &mask_rtn
				);
				if( !r || curX< 0 || curX> wi->dev_info.area_w ||
					curY< 0 || curY> wi->dev_info.area_h
				){
					if( debugFlag ){
						fprintf( StdErr, "DrawCursorCrosses(): cursor not at (%d,%d) in wi==0x%lx\n",
							curX, curY, wi
						);
					}
					wi= NULL;
					WL= WL->next;
				}
				else{
					WL= NULL;
				}
			}
		}
	}
	else{
		XQueryPointer(disp, wi->window, &dum, &dum, &dum2, &dum2,
			  &curX, &curY, &mask_rtn
		);
	}
	if( wi ){
		cx= Reform_X( wi, TRANX(curX), TRANY(curY) );
		cy= Reform_Y( wi, TRANY(curY), TRANX(curX) );
		DrawCCrosses( wi, evt, cx, cy, curX, curY, NULL, "DrawCursorCrosses()" );
	}
}
