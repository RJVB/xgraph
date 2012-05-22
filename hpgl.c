
#define MAPX(state,x) ( (x) + P1X + state->clipminX ) 
#define MAPY(state,y) ( MAXY - (y) + P1Y - state->clipminY)

#include "config.h"
IDENTIFY( "hpgl device code" );

#include "copyright.h"
#include "xgout.h"
#include "plotter.h"
#include <stdio.h>
#include <math.h>
#include "xgraph.h"
#define MAX(a,b) ( ((a)>(b)) ? (a) : (b) )
#define MIN(a,b) ( ((a)<(b)) ? (a) : (b) )
#ifndef __STDC__
	char *malloc(), *calloc();
#endif

extern void exit();
extern void free();

extern void hpglClear();
extern void hpglText();
extern void hpglSeg();
extern void hpglDot();
extern void hpglEnd();


static xgOut hpglInfo = {
    D_COLOR,    /* device characteristics */
    MAXX,   /* width */
    MAXY,   /* height */
    200,    /* border padding */
    0,      /* extra space around axis labels */
    250,    /* tick length - approx 1/4 inch */
    50,	    /* spacing above legend lables */
    0,      /* legend font width */
    0,      /* legend font height */
    0,      /* label font width */
    0,      /* label font height */
    0,      /* axis font width */
    0,      /* axis font height */
    0,      /* shift	*/
    0,      /* title font width */
    0,      /* title font height */
	1000000,/* maximum number of segments */

	hpglClear,	/* stub clear function (pity for those plotters with wipers..)	*/
    hpglText,   /* text output function */
    hpglSeg,    /* segment  drawing function */
    hpglDot,    /* dot/marker drawing function */
    hpglEnd,    /* end of plot function */
	NULL,

    NULL,   /* userInfo */
};

typedef struct {
	double axis_w;
	double axis_h;
	double title_w;
	double title_h;
	double label_w, label_h,
		legend_w, legend_h;
	FILE *plotterFile;
	int clipminX;
	int clipminY;
	int clipmaxX;
	int clipmaxY;
} mydata;

int hpglInit(stream,width,height,orient,title_family, title_size,
		legend_family, legend_size,
		label_family, label_size,
		axis_family, axis_size, win_info,errmsg, initFile)
    FILE *stream;	/* output stream */
	int width;		/* desired width of space in microns */
	int height,orient;		/* desired height in microns */
	char *title_family;	/* name of font for titles */
	double title_size;	/* size of font for titles */
	char *legend_family;	/* name of font for legends */
	double legend_size;	/* size of font for legends */
	char *label_family;	/* name of font for labels */
	double label_size;	/* size of font for labels */
	char *axis_family;	/* name of font for axes */
	double axis_size;	/* size of font for axes */
	LocalWin *win_info;
	char errmsg[ERRBUFSIZE];	/* a place to complain to */
	int initFile;
{
    char *cmd;
    xgOut *outInfo= &(win_info->dev_info);	/* my structure */
	mydata *myInfo= (mydata*) outInfo->userstate;

	if( !myInfo ){
		myInfo = (mydata*)calloc( 1, sizeof(mydata));
	}
	if(myInfo == NULL){
		return(0);
	}
	  /* copy hpgl info	*/
     *outInfo = hpglInfo;
	outInfo->area_w = MIN(MAXX,width/25);
	outInfo->area_h = MIN(MAXY,height/25);
	/* magic formulas:  input sizes are in points = 1/72 inch */
	/* my sizes are in cm */
	/* plotter units are in units of .025mm ~= 1/1016 inch */
	/* have to warn of height 1.5 times larger or get bitten by
	   plotter's internal padding */
	/* widths are (arbitrarily) selected to be 2/3 of the height */
	/*     (cancels with width factor) */
	myInfo->axis_w = axis_size * .666 * 2.54/72.;
	myInfo->axis_h = axis_size * 2.54/72.;
	myInfo->title_w = title_size * .666 * 2.54/72.;
	myInfo->title_h = title_size * 2.54/72.;
	myInfo->legend_w = legend_size * .666 * 2.54/72.;
	myInfo->legend_h = legend_size * 2.54/72.;
	myInfo->label_w = label_size * .666 * 2.54/72.;
	myInfo->label_h = label_size * 2.54/72.;

    outInfo->axis_pad = axis_size*1016.*1.5/72.;
    outInfo->axis_width = axis_size*1016.*1.5/72.;
    outInfo->axis_height = axis_size*1016.*.666/72.;
    outInfo->title_width = title_size*1016.*1.5/72.;
    outInfo->title_height = title_size*1016.*.666/72.;
	outInfo->legend_width = legend_size * .666 * 2.54/72.;
	outInfo->legend_height = legend_size * 2.54/72.;
	outInfo->label_width = label_size * .666 * 2.54/72.;
	outInfo->label_height = label_size * 2.54/72.;

    outInfo->user_state = (char *)myInfo;
    outInfo->user_ssize= sizeof(mydata);
	myInfo->plotterFile = stream;
    myInfo->clipminX = 0;
    myInfo->clipminY = 0;
    myInfo->clipmaxX = MAXX;
    myInfo->clipmaxY = MAXY;
	if( initFile ){
		fprintf(myInfo->plotterFile,"PG;IN;\n");
		fprintf(myInfo->plotterFile,"DI1,0;\n");
	    fprintf(myInfo->plotterFile,"IW%d,%d,%d,%d;\n",MAPX(myInfo,0),
				MAPY(myInfo,myInfo->clipmaxY-myInfo->clipminY),
			  MAPX(myInfo,myInfo->clipmaxX-myInfo->clipminX),
				MAPY(myInfo,0));
	}
    return(1);
}

static void hpglClip(userState,ulx,uly,lrx,lry)
    mydata *userState;    /* my state information  */
    int ulx,uly,lrx,lry;    /* corners of plotting area */
{
    userState->clipminX = ulx;
    userState->clipminY = uly;
    userState->clipmaxX = lrx;
    userState->clipmaxY = lry;
    fprintf(userState->plotterFile,"IW%d,%d,%d,%d;\n",MAPX(userState,0),
			MAPY(userState,userState->clipmaxY-userState->clipminY),
            MAPX(userState,userState->clipmaxX-userState->clipminX),
			MAPY(userState,0));
    return;
}

static void hpglClear(state)
mydata *state;
{ }

static void hpglText(userState,x,y,text,just,style)
    mydata *userState;    /* my state information  */
    int x,y;    /* coords of text origin */
    char *text; /* what to put there */
    int just;   /* how to justify */
    /* where the origin is relative to where the text should go
     * as a function of the various values of just 

    T_UPPERLEFT     T_TOP       T_UPPERRIGHT
    T_LEFT          T_CENTER    T_RIGHT
    T_LOWERLEFT     T_BOTTOM    T_LOWERRIGHT

    */
    int style;  /* T_AXIS = axis font, T_TITLE = title font */

{  char *Text= (char*) calloc( 1,  strlen(text)+ 1);

    fprintf(userState->plotterFile,"PU;SP%d;",TEXTCOLOR);
    fprintf(userState->plotterFile,"PA%d,%d;",MAPX(userState,x),MAPY(userState,y));

	if( !Text ){
		printf( "can't get memory in hpglText\n");
		return;
	}
	{ char *a= Text, *b= text;
		while( *b ){
			if( *b!= '\\'  && isascii(*b) ){
				*a++= *b;
			}
			b++;
		}
		*a= '\0';
	}
    switch(style) {
		case T_LEGEND:
            fprintf(userState->plotterFile,"SI%f,%f;",userState->legend_w,userState->legend_h);
			break;
        case T_LABEL:
            fprintf(userState->plotterFile,"SI%f,%f;",userState->label_w,userState->label_h);
			break;
        case T_AXIS:
            fprintf(userState->plotterFile,"SI%f,%f;",userState->axis_w,userState->axis_h);
            break;
        case T_TITLE:
            fprintf(userState->plotterFile,"SI%f,%f;",userState->title_w,userState->title_h);
            break;
		case T_MARK:
			break;
        default:
            printf("bad text style %d in hpglText\n",style);
			return;
            break;
    }
    switch(just) {
        case T_UPPERLEFT:
            fprintf(userState->plotterFile,"LO3;\n");
            break;
        case T_TOP:
            fprintf(userState->plotterFile,"LO6;\n");
            break;
        case T_UPPERRIGHT:
            fprintf(userState->plotterFile,"LO9;\n");
            break;
        case T_LEFT:
            fprintf(userState->plotterFile,"LO2;\n");
            break;
        case T_CENTER:
            fprintf(userState->plotterFile,"LO5;\n");
            break;
        case T_RIGHT:
            fprintf(userState->plotterFile,"LO8;\n");
            break;
        case T_LOWERLEFT:
            fprintf(userState->plotterFile,"LO1;\n");
            break;
        case T_BOTTOM:
            fprintf(userState->plotterFile,"LO4;\n");
            break;
        case T_LOWERRIGHT:
            fprintf(userState->plotterFile,"LO7;\n");
            break;
        default:
            printf("bad justification type %d in hpglText\n",just);
            return;
            break;
    }
    fprintf(userState->plotterFile,"LB%s\03;", Text);
	free( Text );
}



static int penselect[8] = { PEN1, PEN2, PEN3, PEN4, PEN5, PEN6, PEN7, PEN8};
static int lineselect[8] = { LINE1, LINE2, LINE3, LINE4, LINE5, LINE6, 
        LINE7, LINE8};



static void hpglSeg(userState,ns,segs,width,style,lappr,color)
    mydata *userState;    /* my state information (not used) */
    int ns;         /* number of segments */
    XSegment *segs; /* X array of segments */
    int width;      /* width of lines in pixels */
    int style;      /* L_VAR = dotted, L_AXIS = grid, L_ZERO = axis*/
    int lappr;      /* line style */
    int color;      /* line color */
{
    int i;

    if (style == L_ZERO) {
        fprintf(userState->plotterFile,"SP%d;",PENAXIS); /* select correct pen */
        fprintf(userState->plotterFile,"LT;"); /* solid line style */
    } else if (style == L_AXIS) {
        fprintf(userState->plotterFile,"SP%d;",PENGRID); /* select correct pen */
        fprintf(userState->plotterFile,"LT;"); /* solid line style */
    } else if (style == L_VAR) {
        if( (color < 0) || (color >7) ) {
            printf("out of range line color %d in hpglLine\n",color);
            exit(1);
        }
        fprintf(userState->plotterFile,"SP%d;",penselect[color]); /* select correct pen */
        if( (lappr < 0) || (lappr >7) ) {
            printf("out of range line style %d in hpglLine\n",lappr);
            exit(1);
        }
        if(lappr == 0) {
            fprintf(userState->plotterFile,"LT;");/*select solid line type*/
        } else {
            fprintf(userState->plotterFile,"LT%d;",lineselect[lappr]);/*select line type*/
        }
    } else {
        printf("unknown style %d in hpglLine\n",style);
        exit(1);
    }
    for(i=0;i<ns;i++) {
	double denom;
		if(!i || !( (segs[i].x1==segs[i-1].x2) && (segs[i].y1==segs[i-1].y2) ) ){
            /* MOVE */
            fprintf(userState->plotterFile,"PU;PA%d,%d;\n",MAPX(userState,segs[i].x1),
                    MAPY(userState,segs[i].y1));
        }
			denom = sqrt((double)
					((segs[i].x1-segs[i].x2)*
					(segs[i].x1-segs[i].x2))+
					((segs[i].y1-segs[i].y2)*
					(segs[i].y1-segs[i].y2))
			);
		/* DRAW */
		if(width <= 1 || !denom ) {
			fprintf(userState->plotterFile,"PD;PA%d,%d;\n",MAPX(userState,segs[i].x2),
					MAPY(userState,segs[i].y2));
		} else { /* ugly - wide lines -> rectangles */
			double frac;
			int lx,ly;
			int urx,ury,ulx,uly,llx,lly,lrx,lry;

			frac = (width/2)/denom;
			lx = frac * (segs[i].y2 - segs[i].y1);
			ly = -frac * (segs[i].x2 - segs[i].x1);
			urx = segs[i].x2 +lx;
			ury = segs[i].y2 +ly;
			ulx = segs[i].x2 -lx;
			uly = segs[i].y2 -ly;
			llx = segs[i].x1 -lx;
			lly = segs[i].y1 -ly;
			lrx = segs[i].x1 +lx;
			lry = segs[i].y1 +ly;
			fprintf(userState->plotterFile,"PU;PA%d,%d;",MAPX(userState,llx),
					MAPY(userState,lly));
			fprintf(userState->plotterFile,"PM0;");
			fprintf(userState->plotterFile,"PD,PA%d,%d;PA%d,%d;PA%d,%d;\n",
				MAPX(userState,lrx),MAPY(userState,lry),
				MAPX(userState,urx),MAPY(userState,ury),
				MAPX(userState,ulx),MAPY(userState,uly) );
			fprintf(userState->plotterFile,"PM2;FP;EP;");
		}
    }
	fprintf(userState->plotterFile,"PU;");
}

static char *markselect[MARKS] = { MARK1, MARK2, MARK3, MARK4, MARK5, MARK6, 
        MARK7, MARK8, MARK9, MARK10, MARK11, MARK12, MARK13, MARK14, MARK15, MARK16};

static void hpglDot(userState,x,y,style,type,color)
    mydata *userState;    /* my state information (not used) */
    int x,y;    /* coord of dot */
    int style;  /* type of dot */
    int type;   /* dot style variation */
    int color;  /* color of dot */
{
	type%= MAXATTR;
    /* move to given coord */
    fprintf(userState->plotterFile,"PU;PA%d,%d;\n",MAPX(userState,x), MAPY(userState,y));
    if( (color<0) || (color>7) ) {
        printf("unknown color %d in hpglDot\n",color);
        exit(1);
    }
    fprintf(userState->plotterFile,"SP%d;",penselect[color]);
    if(style == P_PIXEL) {
        fprintf(userState->plotterFile,"PD;PU;\n");
    } else if (style == P_DOT) {
        fprintf(userState->plotterFile,"LT;PM0;CI40;PM2;FT;EP;\n");
    } else if (style == P_MARK) {
        if( (type<0) || (type>=MARKS) ) {
            printf("unknown marker type %d in hpglDot\n",type);
            exit(1);
        }
        /*fprintf(userState->plotterFile,"LT;CA5;LO4;SI0.1;LB%s\03;\n",markselect[type]);*/
		fprintf(userState->plotterFile,"LT;CS5;LO4;SI0.15,0.15;SM%s;PR0,0;SM;CS;\n",markselect[type]);
    } else {
        printf("unknown marker style %d in hpglDot\n",style);
        exit(1);
    }
}

static void hpglEnd(userState)
    mydata *userState;    /* my state information (not used) */

{
	fprintf(userState->plotterFile,"SP;PG;IN;\n");
    fflush(userState->plotterFile);
    return;
}
