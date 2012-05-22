#ifndef NEW_PS_H

typedef enum { PS_PRINTING=0x6abccba6,
	PS_FINISHED=0x9cbaabc9, X_DISPLAY=0xbca00acb, XG_DUMPING=0x12344321
} PS_Printing;

typedef struct userInfo {
	PS_Printing Printing;
    FILE *psFile;
    int currentTextStyle;
    int currentDashStyle;
    double currentWidth;
    int currentLStyle;
    int baseWidth;
    int height_devs;
    char *title_family;
    double title_size;
    char *legend_family;
    double legend_size;
    char *label_family;
    double label_size;
    char *axis_family;
    double axis_size;
	char *current_family;
	double current_size;
	char JobName[512];
	Boolean silent;
	int truncated;
	long clear_all_pos;
	xgOut *dev_info;
} psUserInfo;

extern psUserInfo *PS_STATE(LocalWin *wi );

extern int ps_old_font_offsets, ps_page_nr, ps_previous_dimensions;


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

#define PNT_WIDTH(width)	((double)(width)+10000.0) / MICRONS_PER_INCH * POINTS_PER_INCH
#define inv_PNT_WIDTH(w)	((w)/ POINTS_PER_INCH)* MICRONS_PER_INCH- 10000.0

/*
 * Aesthetic parameters (inches)
 */

#define PS_BDR_PAD		0.075
#define PS_AXIS_PAD		0.1
#define PS_LEG_PAD		0.025
#define PS_TICK_LEN		0.125
#define BASE_DASH		(1.0/48.0)

#define BASE_WIDTH		(1.0/8.0)
/* #define PS_AXIS_WBASE		3	*/
/* #define PS_ZERO_WBASE		5	*/
#define PS_DATA_WBASE		7
#define PS_AXIS_WBASE		PS_DATA_WBASE
#define PS_ZERO_WBASE		PS_DATA_WBASE
#define PS_PIXEL		4
#define PS_DOT			12
#define PS_MARK			12


/*
 * Other constants
 */

/* 991105: due to addition of width('M') to XFontWidth():
#	define FONT_WIDTH_EST		0.56
 */
#define FONT_WIDTH_EST		0.675

/* #define FONT_WIDTH_EST		0.5867	*/

#define PS_MAX_SEGS		1000
#define PS_NO_TSTYLE		-1
#define PS_NO_DSTYLE		-1
#define PS_NO_WIDTH		-1
#define PS_NO_LSTYLE		-1

extern double ps_MarkSize_X(), ps_MarkSize_Y();
/* 990801: these should be declared as returning double also!	*/
extern int X_ps_MarkSize_X(), X_ps_MarkSize_Y();
extern int internal_psMarkers, psMarkers;

extern RGB *psThisRGB;

extern char ps_comment[1024];

#define NEW_PS_H
#endif /* NEW_PS_H	*/
