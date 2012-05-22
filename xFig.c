/*
vi:set ts=4|set sw=4:
 */
/*
 * Fig output for xgraph
 *
 * RenE J.V. Bertin
 \ Hacked from new_ps.c
 */

#include "config.h"
IDENTIFY( "xfig device code (unsupported)" );

#include "copyright.h"
#include <stdio.h>
#include <unistd.h>
#include "xgout.h"
#include "xgraph.h"

/*
 * Basic scaling parameters
 */

#define VDPI			1200.0
#define LDIM			11.0
#define SDIM			8.5
  /* 1 MICRON ~= 1e-4 cm	*/
#define MICRONS_PER_INCH	2.54E+04
#define POINTS_PER_INCH		72.0
#define INCHES_PER_POINT	1.0/72.0

/*
 * Aesthetic parameters (inches)
 */

#define PS_BDR_PAD		0.075
#define PS_AXIS_PAD		0.1
#define PS_LEG_PAD		0.025
#define PS_TICK_LEN		0.125
#define BASE_DASH		(1.0/48.0)

#define BASE_WIDTH		(1.0/8.0)
#define PS_AXIS_WBASE		3
#define PS_ZERO_WBASE		5
#define PS_DATA_WBASE		7
#define PS_PIXEL		4
#define PS_DOT			12
#define PS_MARK			12

/*
 * Other constants
 */

#define FONT_WIDTH_EST		0.55
#define PS_MAX_SEGS		1000
#define PS_NO_TSTYLE		-1
#define PS_NO_DSTYLE		-1
#define PS_NO_WIDTH		-1
#define PS_NO_LSTYLE		-1

/*
 * Working macros
 */

#define figFile psFile

#define OUT		(void) fprintf
#define PS(str)		OUT(figFile, str)
#define PSU(str)	OUT(ui->figFile, str)
#define IY(val)		(ui->height_devs - val)
#define MAX(a, b)	((a) > (b) ? (a) : (b))

/*
 * Globals
 */

static double PS_scale;		/* devs/micron */
char fig_comment[128];

/*
 * Externals and forwards
 */

#ifndef __STDC__
	extern char *malloc(), *calloc();
#endif
static void figScale(), figFonts(), figMarks(), figText(), figClear(), figSeg(), figDot(), figEnd();

/*
 * Local structures
 */

#include "new_ps.h"


int rd(dbl)
double dbl;
/* Short and sweet rounding function */
{
    if (dbl < 0.0) {
		return ((int) (dbl - 0.5));
    } else {
		return ((int) (dbl + 0.5));
    }
}

/*ARGSUSED*/
int figInit(figFile, width, height, tf, ts, lf, ls, af, as, outInfo, errmsg)
FILE *figFile;			/* Output file            */
int width, height;		/* In microns             */
char *tf, *lf, *af;			/* Title and axis font    */
double ts, ls, as;			/* Title and axis size    */
xgOut *outInfo;			/* Returned device info   */
char errmsg[ERRBUFSIZE];	/* Returned error message */
/*
 * The basic coordinate system is points (roughly 1/72 inch).
 * However,  most laser printers can do much better than that.
 * We invent a coordinate system based on VDPI dots per inch.
 * This goes along the long side of the page.  The long side
 * of the page is LDIM inches in length,  the short side
 * SDIM inches in length.  We we call this unit a `dev'.
 * We map `width' and `height' into devs.
 */
{
	struct userInfo *ui;
    double font_size;
	extern char Window_Name[256];
	char hnam[80];
	extern FILE *FigFile;

    ui = (struct userInfo *) calloc( 1, sizeof(struct userInfo));
    FigFile= ui->figFile = figFile;
    ui->currentTextStyle = PS_NO_TSTYLE;
    ui->currentDashStyle = PS_NO_DSTYLE;
    ui->currentWidth = PS_NO_WIDTH;
    ui->currentLStyle = PS_NO_LSTYLE;
    ui->title_family = tf;
    ui->title_size = ts;
    ui->label_family = lf;
    ui->label_size = ls;
    ui->axis_family = af;
    ui->axis_size = as;
    /* Roughly,  one-eighth a point in devs */
    ui->baseWidth = rd( VDPI / POINTS_PER_INCH * BASE_WIDTH );

    PS_scale = VDPI / MICRONS_PER_INCH;

    outInfo->dev_flags = 0;
    outInfo->area_w = rd( ((double) width) * PS_scale );
    outInfo->area_h = rd( ((double) height) * PS_scale );
    ui->height_devs = outInfo->area_h;
    outInfo->bdr_pad = rd( PS_BDR_PAD * VDPI );
    outInfo->axis_pad = rd( PS_AXIS_PAD * VDPI );
    outInfo->legend_pad = rd( PS_LEG_PAD * VDPI );
    outInfo->tick_len = rd( PS_TICK_LEN * VDPI );
    outInfo->errortick_len = rd( PS_TICK_LEN * VDPI/ 1.414 );

    /* Font estimates */
    font_size = as * INCHES_PER_POINT * VDPI;
    outInfo->axis_height = rd( font_size );
    outInfo->axis_width = rd( font_size * FONT_WIDTH_EST );

    font_size = ls * INCHES_PER_POINT * VDPI;
    outInfo->label_height = rd( font_size );
    outInfo->label_width = rd( font_size * FONT_WIDTH_EST );

    font_size = ts * INCHES_PER_POINT * VDPI;
    outInfo->title_height = rd( font_size );
    outInfo->title_width = rd( font_size * FONT_WIDTH_EST );

    outInfo->max_segs = PS_MAX_SEGS;

    outInfo->xname_vshift = (int)(3.0* outInfo->axis_height+ 0.5);
    outInfo->xg_text = figText;
    outInfo->xg_clear = figClear;
    outInfo->xg_seg = figSeg;
    outInfo->xg_dot = figDot;
    outInfo->xg_end = figEnd;
    outInfo->user_state = (char *) ui;
	outInfo->user_ssize= sizeof(struct userInfo);

	OpenFigFile( "/dev/null", 0, 0);
    PS("%%!PS-Adobe-1.0\n");
    PS("%% Xgraph postscript output\n");
    PS("%% Rick Spickelmier and David Harrison\n");
    PS("%% University of California, Berkeley\n");

	gethostname( hnam, 80);
	if( strlen(Window_Name) ){
		fprintf( figFile, "#Title: %s\n", Window_Name);
		sprintf( ui->JobName, "%s@%s - %s", getlogin(), hnam, Window_Name);
	}
	else{
		fprintf( figFile, "#Title: XGraph Plot\n");
		sprintf( ui->JobName, "%s@%s - XGraph", getlogin(), hnam );
	}

	fflush(figFile);

	fprintf( figFile, "# %s\n", ui->JobName);

    /* Definitions */
    figScale(figFile, width, height);
    figFonts(figFile);
    figMarks(figFile);

    PS("#\n# Main body begins here\n#\n");
	if( strlen(fig_comment) ){
		PS("# FIGcomment: ");
		PS( fig_comment);
		PS("\n");
		fig_comment[0]= '\0';
	}
	fflush(figFile);
	ui->Printing= PS_PRINTING;
    return 1;
}



static void figScale(figFile, width, height)
FILE *figFile;			/* Output stream */
int width;			/* Output width  */
int height;			/* Output height */
/*
 * This routine figures out how transform the basic postscript
 * transformation into one suitable for direct use by
 * the drawing primitives.  Two variables X-CENTER-PLOT
 * and Y-CENTER-PLOT determine whether the plot is centered
 * on the page.
 */
{
    double factor;
    double pnt_width, pnt_height;

	if( strlen(fig_comment) ){
		PS("# FIGcomment: ");
		PS( fig_comment);
		PS("\n");
		fig_comment[0]= '\0';
	}
    /*
     * Determine page size
     */

    /*
     * First: rotation.  If the width is greater than the short
     * dimension,  do the rotation.
     */
    pnt_width = ((double) width + 10000.0 ) / MICRONS_PER_INCH * POINTS_PER_INCH;
    pnt_height = ((double) height + 10000.0 ) / MICRONS_PER_INCH * POINTS_PER_INCH;
    PS("%% Determine whether rotation is required\n");
    OUT(figFile, "%lg page-width gt\n", pnt_width);
    PS("{ %% Rotation required\n");
    PS("   90 rotate\n");
    PS("   0 page-width neg translate\n");
    PS("   %% Handle centering\n");
    PS("   Y-CENTER-PLOT 1 eq { %% Center in y\n");
    OUT(figFile, "      page-height %lg sub 2 div\n", pnt_width);
    PS("   } { %% Don't center in y\n");
    PS("      0\n");
    PS("   } ifelse\n");
    PS("   X-CENTER-PLOT 1 eq { %% Center in x\n");
    OUT(figFile, "      page-width %lg sub 2 div\n", pnt_height);
    PS("   } { %% Don't center in x\n");
    PS("      0\n");
    PS("   } ifelse\n");
    PS("   translate\n");
    PS("} { %% No rotation - just handle centering\n");
    PS("   X-CENTER-PLOT 1 eq { %% Center in x\n");
    OUT(figFile, "      page-width %lg sub 2 div\n", pnt_width);
    PS("   } { %% Don't center in x\n");
    PS("      0\n");
    PS("   } ifelse\n");
    PS("   Y-CENTER-PLOT 1 eq { %% Center in y\n");
    OUT(figFile, "      page-height %lg sub 2 div\n", pnt_height);
    PS("   } { %% Don't center in y\n");
    PS("      0\n");
    PS("   } ifelse\n");
    PS("   translate\n");
    PS("} ifelse\n");
	PS( "%% end rotation section\n");

    /*
     * Now: scaling.  We have points.  We want devs.
     */
    factor = POINTS_PER_INCH / VDPI;
}


static void figFonts(figFile)
FILE *figFile;			/* Output stream                */
/*
 * Downloads code for drawing title and axis labels
 */
{
	if( strlen(fig_comment) ){
		PS("# FIGcomment: ");
		PS( fig_comment);
		PS("\n");
		fig_comment[0]= '\0';
	}
    PS("%% Font Handling Functions\n");
    PS("%%\n");
    PS("%% Function giving y-offset to center of font\n");
    PS("%% Assumes font is set and uses numbers to gauge center\n");
    PS("%%\n");
    PS("/choose-font	%% stack: fontsize fontname => ---\n");
    PS("{\n");
    PS("   findfont \n");
    PS("   exch scalefont \n");
    PS("   setfont\n");
    PS("   newpath\n");
    PS("   0 0 moveto (0) true charpath flattenpath pathbbox\n");
    PS("   /top exch def pop\n");
    PS("   /bottom exch def pop\n");
    PS("   bottom top bottom top add 2 div\n");
    PS("   /center-font-val exch def \n");
    PS("   /upper-font-val exch def \n");
    PS("   /lower-font-val exch def\n");
    PS("} def\n");
    PS("%%\n");
    PS("%% Justfication offset routines\n");
    PS("%%\n");
    PS("/center-x-just	%% stack: (string) x y => (string) newx y\n");
    PS("{\n");
    PS("   exch 2 index stringwidth pop 2 div sub exch\n");
    PS("} def\n");
    PS("%%\n");
    PS("/left-x-just	%% stack: (string) x y => (string) newx y\n");
    PS("{ \n");
    PS("} def\n");
    PS("%%\n");
    PS("/right-x-just	%% stack: (string) x y => (string) newx y\n");
    PS("{\n");
    PS("   exch 2 index stringwidth pop sub exch\n");
    PS("} def\n");
    PS("%%\n");
    PS("/center-y-just	%% stack: (string) x y => (string) x newy\n");
    PS("{\n");
    PS("   center-font-val sub\n");
    PS("} def\n");
    PS("%%\n");
    PS("/lower-y-just	%% stack: (string) x y => (string) x newy\n");
    PS("{\n");
    PS("   lower-font-val sub\n");
    PS("} def\n");
    PS("%%\n");
    PS("/upper-y-just	%% stack: (string) x y => (string) x newy\n");
    PS("{\n");
    PS("   upper-font-val sub\n");
    PS("} def\n");
    PS("%%\n");
    PS("%% Shows a string on the page subject to justification\n");
    PS("%%   \n");
    PS("/just-string	%% stack: (string) x y just => ---\n");
    PS("{\n");
    PS("   dup 0 eq { pop center-x-just center-y-just 		} if\n");
    PS("   dup 1 eq { pop left-x-just center-y-just		} if\n");
    PS("   dup 2 eq { pop left-x-just upper-y-just	 	} if\n");
    PS("   dup 3 eq { pop center-x-just upper-y-just 		} if\n");
    PS("   dup 4 eq { pop right-x-just upper-y-just	 	} if\n");
    PS("   dup 5 eq { pop right-x-just center-y-just 		} if\n");
    PS("   dup 6 eq { pop right-x-just lower-y-just	 	} if\n");
    PS("   dup 7 eq { pop center-x-just lower-y-just  		} if\n");
    PS("   dup 8 eq { pop left-x-just lower-y-just	 	} if\n");
    PS("   moveto show\n");
    PS("} def\n");
    PS("%%\n");
}


static void figMarks(figFile)
FILE *figFile;
/*
 * Writes out marker definitions
 */
{
	if( strlen(fig_comment) ){
		PS("# FIGcomment: ");
		PS( fig_comment);
		PS("\n");
		fig_comment[0]= '\0';
	}
    PS("%% Marker definitions\n");
    PS("/mark0 {/size exch def /y exch def /x exch def\n");
    PS("newpath x size sub y size sub moveto\n");
    PS("size size add 0 rlineto 0 size size add rlineto\n");
    PS("0 size size add sub 0 rlineto closepath stroke} def\n");
    
    PS("\n/mark1 {/size exch def /y exch def /x exch def\n");
    PS("x y size mark0\n");
    PS("newpath x y size mark3 } def\n");
    
    PS("\n/mark2 {/size exch def /y exch def /x exch def\n");
    PS("newpath x y moveto x y size 0 360 arc stroke} def\n");

    PS("\n/mark3 {/size exch def /y exch def /x exch def\n");
    PS("newpath x size sub y size sub moveto x size add y size add lineto\n");
    PS("x size sub y size add moveto x size add y size sub lineto stroke} def\n");

    PS("\n/mark4 {/size exch def /y exch def /x exch def\n");
    PS("newpath x size sub y moveto x y size add lineto\n");
    PS("x size add y lineto x y size sub lineto\n");
    PS("closepath stroke} def\n");
    
    PS("\n/mark5 {/size exch def /y exch def /x exch def\n");
    PS("x y size mark4\n");
    PS("newpath x y size mark0 } def\n");
    
    PS("\n/mark6 {/size exch def /y exch def /x exch def\n");
    PS("x y size mark2\n");
    PS("newpath x y size mark3 } def\n");

    PS("\n/mark7 {/size exch def /y exch def /x exch def\n");
    PS("newpath x y moveto x size sub y size sub lineto\n");
    PS("x size add y size sub lineto closepath fill\n");
    PS("newpath x y moveto x size add y size add lineto\n");
    PS("x size sub y size add lineto closepath fill} def\n");

}


static void figText(state, x, y, text, just, style)
char *state;			/* Really (struct userInfo *) */
int x, y;			/* Text position (devs)       */
char *text;			/* Text itself                */
int just;			/* Justification              */
int style;			/* Style                      */
/*
 * Draws text at the given location with the given justification
 * and style.
 */
{
    struct userInfo *ui = (struct userInfo *) state;

	if( strlen(fig_comment) ){
		OUT( ui->figFile, "# FIGcomment: %s\n", fig_comment);
		fig_comment[0]= '\0';
	}
    if (style != ui->currentTextStyle) {
		switch (style) {
			case T_AXIS:
				OUT(ui->figFile, "%lg /%s choose-font\n",
				ui->axis_size * INCHES_PER_POINT * VDPI, ui->axis_family);
				break;
			case T_LABEL:
				OUT(ui->figFile, "%lg /%s choose-font\n",
				ui->label_size * INCHES_PER_POINT * VDPI, ui->label_family);
				break;
			case T_TITLE:
				OUT(ui->figFile, "%lg /%s choose-font\n",
				ui->title_size * INCHES_PER_POINT * VDPI, ui->title_family);
				break;
		}
		ui->currentTextStyle = style;
    }
    OUT(ui->figFile, "(%s) %d %d %d just-string\n", text, x, IY(y), just);
}


static void figClear(state)
char  *state;
{ }

int use_fig_LStyle= 0, fig_LStyle= 0;
double figdash_power= 0.75;

/*ARGSUSED*/
static void figSeg(state, ns, seglist, width, style, lappr, color)
char *state;			/* Really (struct userInfo *) */
int ns;				/* Number of segments         */
XSegment *seglist;		/* X array of segments        */
int width;			/* Width of lines (devcoords) */
int style;			/* L_AXIS, L_ZERO, L_VAR      */
int lappr;			/* Zero to seven              */
int color;			/* Zero to seven              */
/*
 * Draws a number of line segments.  Grid lines are drawn using
 * light lines.  Variable lines (L_VAR) are drawn wider.  This
 * version ignores the color argument.
 */
{
    struct userInfo *ui = (struct userInfo *) state;
    int newwidth, i, Style;
	static unsigned long called= 0L;
	extern int debugFlag;

	if( strlen(fig_comment) ){
		OUT( ui->figFile, "# FIGcomment: %s\n", fig_comment);
		fig_comment[0]= '\0';
	}
	Style= (use_fig_LStyle)? fig_LStyle : style;
    if ((Style != ui->currentLStyle) || (width != ui->currentWidth)) {
		  /* Outcomment width*= newwidth lines to return to wildtype
		   * xgraph behaviour. This phenotype draws a line with width 2
		   * twice as broad as the default line (w=1)
		   */
		switch (Style) {
			case L_AXIS:
				newwidth = PS_AXIS_WBASE * ui->baseWidth;
				width*= newwidth;
				PSU(" [] 0 setdash ");
				break;
			case L_ZERO:
				newwidth = PS_ZERO_WBASE * ui->baseWidth;
				width*= newwidth;
				PSU(" [] 0 setdash ");
				break;
			case L_VAR:
				newwidth = PS_DATA_WBASE * ui->baseWidth;
				width*= newwidth;
				break;
		}
		ui->currentWidth = MAX(newwidth, width);
		ui->currentLStyle = Style;
		OUT(ui->figFile, " %d setlinewidth\n", ui->currentWidth);
    }
    if ((lappr != ui->currentDashStyle) && (style == L_VAR)) {
		if (lappr == 0) {
			PSU(" [] 0 setdash ");
		} else {
		  extern double pow();
		  double __lappr, _lappr= ((double) lappr) * BASE_DASH * VDPI;
			if( _lappr< 200){
				OUT(ui->figFile, " [%lg] 0 setdash ", _lappr);
			}
			else{
				_lappr= pow((double)lappr, figdash_power ) * BASE_DASH * VDPI;
				__lappr= pow((double)lappr, figdash_power * figdash_power ) * BASE_DASH * VDPI;
				OUT(ui->figFile, " [%lg %lg %lg %lg] 0 setdash ",
					_lappr, __lappr, __lappr, __lappr
				);
			}
		}
		ui->currentDashStyle = lappr;
    }
    PSU(" newpath\n");
	if( debugFlag){
		OUT( ui->figFile, "%% seglist %lu\n", called);
	}
    OUT(ui->figFile, "  %d %d moveto ", seglist[0].x1, IY(seglist[0].y1));
	if( debugFlag){
		OUT(ui->figFile, "  %d %d lineto %% seg %d\n", seglist[0].x2, IY(seglist[0].y2), 0 );
	}
	else{
		OUT(ui->figFile, "  %d %d lineto\n", seglist[0].x2, IY(seglist[0].y2));
	}
    for (i = 1;  i < ns;  i++) {
		if( !(seglist[i].x1 == seglist[i-1].x2 && seglist[i].y1 == seglist[i-1].y2) ) {
				PSU( "   stroke\n");
				OUT(ui->figFile, "  %d %d moveto ", seglist[i].x1, IY(seglist[i].y1));
		}
		if( debugFlag){
			OUT(ui->figFile, "  %d %d lineto %% seg %d\n", seglist[i].x2, IY(seglist[i].y2), i);
		}
		else{
			OUT(ui->figFile, "  %d %d lineto\n", seglist[i].x2, IY(seglist[i].y2));
		}
    }
    PSU(" stroke\n");
	called++;
}


extern double psm_incr, psm_base;

/*ARGSUSED*/
static void figDot(state, x, y, style, type, color, setnr)
char *state;    	/* state information */
int x,y;    		/* coord of dot */
int style;  		/* type of dot */
int type;   		/* dot style variation */
int color;  		/* color of dot */
int setnr;
/*
 * Prints out a dot at the given location
 */
{
    struct userInfo *ui = (struct userInfo *) state;

	type%= MAXATTR;
	if( strlen(fig_comment) ){
		OUT( ui->figFile, "# FIGcomment: %s\n", fig_comment);
		fig_comment[0]= '\0';
	}
    if (ui->currentDashStyle != PS_NO_DSTYLE) {
		OUT(ui->figFile, "[] 0 setdash ");
		ui->currentDashStyle = PS_NO_DSTYLE;
    }
    if (ui->currentWidth != PS_ZERO_WBASE * ui->baseWidth) {
		ui->currentWidth = PS_ZERO_WBASE * ui->baseWidth;
		OUT(ui->figFile, "%d setlinewidth ", ui->currentWidth);
    }
    
    switch (style) {
		case P_PIXEL:
			OUT(ui->figFile, "newpath %d %d moveto %d %d %d 0 360 arc fill\n",
				x, IY(y), x, IY(y), PS_PIXEL * ui->baseWidth);
			break;
		case P_DOT:
			OUT(ui->figFile, "newpath %d %d moveto %d %d %d 0 360 arc fill\n",
				x, IY(y), x, IY(y), PS_DOT * ui->baseWidth);
			break;
		case P_MARK:
			OUT(ui->figFile, "%d %d %g mark%d\n",
				x, IY(y), PS_MARK * ui->baseWidth* ( (int)(setnr/8)* psm_incr+ psm_base), type % 8);
			break;
    }
    return;
}


figMessage( state, message)
char *state;
char *message;
{  extern FILE *StdErr;
   struct userInfo *ui= (struct userInfo*) state;
   extern int debugFlag;
	if( debugFlag){
		fprintf( StdErr, "figMessage(%s)\n", message);
		fflush( StdErr);
	}
}

static void figEnd(LocalWin *info)
{
    struct userInfo *ui = (struct userInfo *) info->dev_info.user_state;

	if( strlen(fig_comment) ){
		OUT( ui->figFile, "# FIGcomment: %s\n", fig_comment);
		fig_comment[0]= '\0';
	}
	OUT( ui->figFile, "(%s - Finished)jn\n", ui->JobName);
    PSU("showpage\n");
	ui->Printing= PS_FINISHED;
}
