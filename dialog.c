/*
vim:ts=4:sw=4:
 * Xgraph Dialog Boxes
 *
 * This file constructs the hardcopy dialog
 * box used by xgraph.
 */

#include "config.h"
IDENTIFY( "Hardcopy Dialog code" );

#include <stdio.h>
#include <sys/param.h>

#include <unistd.h>
#include "xgout.h"
#include "xgraph.h"
#include "hard_devices.h"
#include "new_ps.h"
#include "xtb/xtb.h"
#include <X11/Xutil.h>
#include <X11/keysym.h>

#include <math.h>

#define POPUP_IDENT
#include "copyright.h"

#include <float.h>


extern char *getcwd();

#include "fdecl.h"


#ifndef MAX
#	define MAX(a,b)	(((a)<(b))?(b):(a))
#endif
#ifndef CLIP
#	define CLIP(var,low,high)	if((var)<(low)){\
		(var)=(low);\
	}else if((var)>(high)){\
		(var)=(high);}
#endif

void do_error();

#define MAXCHBUF	1024

static LocalWin *theWin_Info;

#define D_VPAD	3
#define D_HPAD	2
#define D_INT	2
#define D_BRDR	2
#define D_INP	50,MAXCHBUF
#define D_DSP	10
#define D_FS	10
#define D_SN	3

typedef struct ho_d_info {
    char *prog;			/* Program name              */
    xtb_data cookie;		/* Info used by do_harcopy   */
    Window choices;		/* Output device choices     */
    Window fod;			/* File or device flag       */
    Window fodspec;		/* File or device spec       */
    Window height_dimspec;		/* Maximum dimension spec    */
    Window width_dimspec;		/* Maximum dimension spec    */
    Window tf_family;		/* Title font family spec    */
    Window tf_size;		/* Title font size spec      */
    Window lef_family;		/* Legend font family spec     */
    Window lef_size;		/* Legend font size spec       */
	Window laf_family,		/* Label font family, size spec	*/
		laf_size;
    Window af_family;		/* Axis font family spec     */
    Window af_size;		/* Axis font size spec       */
	int errnr, printOK;
} ho_d_info;

int lef_size, tf_size, af_size, laf_size;

ho_d_info *HO_d_info= NULL;

static ho_d_info *HO_info= NULL;

#define	TAB			'\t'
#define	BACKSPACE	0010
#define DELETE		0177
#define	CONTROL_P	0x1b	/* actually ESC	*/
#define CONTROL_U	0025
#define CONTROL_W	0027
#define CONTROL_X	0030

static char word_sep[]= ";:,./ \t-_";

#define DONE_LEN	512

/* Indices for frames made in make_HO_dialog */
enum HO_frames_defn {
    TITLE_F, ODEVLBL_F, ODEVROW_F, DISPLBL_F, DISPROW_F, PRESTIME_F, FDLBL_F, DIR_F,
    FDINP_F, /* OPTLBL_F,*/ MDIMLBL_F, MWDIMI_F, MHDIMI_F, XWR_F, PSSA_F, XGAWD_F, UHPTC_F,
	PSMBILBL_F, PSMB_F, PSMI_F,
	PSPOSLBL_F, PSSC_F, PSPOSX_F, PSPOSY_F,
/* 	PS_LOFF_F, PS_BOFF_F,	*/
	PSORIENT_F, PSSP_F, PSEPS_F, PSDSC_F, PSSetPage_F, DONE_F, PAGE_XGPS_F, SETS_F,
	UXLF_F, UGSTW_F, SPAX_F, SPAY_F, PSPC_F, SPSS_F, XGSB_F, XGSTR_F, XGDAV_F, XGDPR_F, XGDPFRESH_F,
	XGSLDSC_F, XGDBIN_F, XGDASC_F, XGDHEX_F, XGDPEN_F,
	PSMP_F, PSSM_F, PSRGB_F, PSTRS_F, XGINI_F,
	TFFAMLBL_F, TFFAM_F, TFSIZLBL_F, TFSIZ_F,
	LEFFAMLBL_F, LEFFAM_F, LEFSIZLBL_F, LEFSIZ_F,
	LAFFAMLBL_F, LAFFAM_F, LAFSIZLBL_F, LAFSIZ_F,
	AFFAMLBL_F, AFFAM_F, AFSIZLBL_F, AFSIZ_F,
 
    OK_F, CAN_F, ABOUT_F, REDRAW_F, REWRITE_F, SETTINGS_F, BAR_F, LAST_F
} HO_frames;

Window HO_printit_win= 0;
xtb_frame HO_okbtn, HO_canbtn, HO_redrawbtn;
static struct ho_d_info *ok_info, *cinfo, *rinfo;

static void *HO_nothing= NULL;

/* RJB: array of text_boxes that can be activated sequentially by hitting a
 * tab
 */
int text_box[]= { DIR_F, FDINP_F, MWDIMI_F, MHDIMI_F, PSMB_F, PSMI_F, TFFAM_F, TFSIZ_F, LEFFAM_F, LEFSIZ_F, LAFFAM_F, LAFSIZ_F, AFFAM_F, AFSIZ_F };
int text_boxes= sizeof(text_box)/sizeof(int);

extern int data_sn_number, data_sn_linestyle, data_sn_elinestyle, legend_changed;
extern double data_sn_lineWidth;
extern char data_sn_number_buf[D_SN], data_sn_lineWidth_buf[D_SN], data_sn_linestyle_buf[D_SN],
	data_sn_elinestyle_buf[D_SN], data_legend_buf[LEGEND];

extern int PS_PrintComment, Sort_Sheet, XG_SaveBounds, XG_Stripped, dump_average_values, splits_disconnect,
	DumpProcessed, Init_XG_Dump, XG_Really_Incomplete;
int set_preserve_screen_aspect= 2, set_scale_plot_area= 3,
	set_PS_PrintComment, set_Sort_Sheet, set_winsize= 0;
extern double Font_Width_Estimator;

extern int XG_preserve_filetime;

extern int ps_xpos, ps_ypos;
extern double ps_scale, ps_l_offset, ps_b_offset;
int ps_mpage= 0;
extern int ps_show_margins, ps_coloured, ps_transparent;
extern int XGStoreColours, TrueGray;

int showpage= 1;
extern int ps_page_nr, psEPS, psDSC, psSetPage;
extern double psSetPage_width, psSetPage_height;
extern double psSetHeight_corr_factor, psSetWidth_corr_factor;
int PrintSetsNR= -1;

extern char *XGstrdup(const char*);

int last_gsTextWidthBatch= 0;
extern int use_X11Font_length, use_gsTextWidth, auto_gsTextWidth, scale_plot_area_x, scale_plot_area_y;

extern int preserve_screen_aspect;

#include "xfree.h"

extern Pixmap dotMap ;
int fod_spot;
int device_nr;

#define AF(ix)	ho_af[(int) (ix)]
#define aF(ix)	&AF(ix)
xtb_frame AF(LAST_F);
int ho_last_f= LAST_F;
xtb_frame HO_Dialog = { (Window) 0, 0, 0, 0, 0 };

extern double psm_incr, psm_base;
extern int psm_changed;

static xtb_hret HO_cycle_focus_button( int firstchar )
{  static xtb_frame *current= NULL, *brcurrent= NULL;
   static int currentN= 0, brcurrentN= -1;
   xtb_frame *hit;
	if( (hit= xtb_find_next_named_button( ho_af, LAST_F, &current, &currentN, &brcurrent, &brcurrentN, NULL, firstchar )) ){
#ifdef TAB_WARPS
{
	  int x_loc, y_loc;
		x_loc= hit->width/2;
		y_loc= hit->height/2;
		XWarpPointer( disp, None, hit->win, 0, 0, 0, 0, x_loc, y_loc);
}
#else
		XSetInputFocus( disp, hit->win, RevertToParent, CurrentTime);
#endif
	}
	else{
		Boing(5);
	}
	return( XTB_HANDLED);
}

static xtb_frame *get_text_box(int i)
{ xtb_frame *frame= NULL;
 	if( i>=0 && i< text_boxes ){
		frame= aF(text_box[i]);
		if( frame->info->type== xtb_TI2 ){
			frame= xtb_ti2_ti_frame( frame->win );
		}
	}
	return(frame);
}

static xtb_hret goto_next_text_box( Window win)
{  int i, hit= 0;
#ifdef TAB_WARPS
   int x_loc, y_loc;
#endif
	if( hit ){
		win= AF(hit).win;
#ifdef TAB_WARPS
		x_loc= AF(hit).width/2;
		y_loc= AF(hit).height/2;
		XWarpPointer( disp, None, *win, 0, 0, 0, 0, x_loc, y_loc);
#else
		XSetInputFocus( disp, win, RevertToParent, CurrentTime);
#endif
	}
	else{
		for( i= 0, hit= 0; !hit && i< text_boxes; i++){
			if( get_text_box(i)->win== win ){
				hit= i+1;
			}
		}
		{
			if( hit== text_boxes)
				hit= 0;
			win= get_text_box(hit)->win;
#ifdef TAB_WARPS
			x_loc= get_text_box(hit)->width/2;
			y_loc= get_text_box(hit)->height/2;
			XWarpPointer( disp, None, win, 0, 0, 0, 0, x_loc, y_loc);
#else
			XSetInputFocus( disp, win, RevertToParent, CurrentTime);
#endif
		}
	}
	return( XTB_HANDLED);
}

static xtb_hret goto_prev_text_box( Window win)
{  int i, hit;
#ifdef TAB_WARPS
   int x_loc, y_loc;
#endif
	for( i= 0, hit= 0; !hit && i< text_boxes; i++){
		if( get_text_box(i)->win== win ){
			hit= i+1;
		}
	}
	if( hit){
		hit-= 2;
		if( hit== -1)
			hit= text_boxes-1;
		win= get_text_box(hit)->win;
#ifdef TAB_WARPS
		x_loc= get_text_box(hit)->width/2;
		y_loc= get_text_box(hit)->height/2;
		XWarpPointer( disp, None, win, 0, 0, 0, 0, x_loc, y_loc);
#else
		XSetInputFocus( disp, win, RevertToParent, CurrentTime);
#endif
	}
	else
		Boing(5);
	return( XTB_HANDLED);
}

char UPrintFileName[MAXCHBUF];

extern FILE *StdErr;
extern int debugFlag;

xtb_hret SimpleFileDialog( Window sourcedest, Window parent, int bval, xtb_data info, int append )
{ ALLOCA( command, char, xtb_ti_length(sourcedest,0)+ MAXPATHLEN+ 1, command_len);
  ALLOCA( result, char, MAXBUFSIZE+ 1, result_len);
  FILE *fp;
	sprintf( command, "gtk-shell -t \"Please select a file (or directory)\" -fs" );
	if( !(fp= popen(command, "r")) ){
	  static char done= 0;
		if( !done ){
			xtb_error_box(parent, "You need gtk-shell installed and in the path for this to work!", "Error" );
			done= 1;
		}
	}
	else{
		if( fgets( result, MAXBUFSIZE, fp ) && strlen(result) ){
		  char *new;
			xtb_ti_get( sourcedest, command, (xtb_data) NULL );
			if( append && *command && (new= concat( command, "  ", result, NULL )) ){
				xtb_ti_set( sourcedest, new, 0 );
				xfree(new);
			}
			else{
				xtb_ti_set( sourcedest, result, 0 );
			}
		}
		pclose(fp);
	}
	return( XTB_HANDLED );
}

static xtb_hret HO_df_fun(Window win, int ch, char *text, xtb_data val);

static xtb_hret FSfun( Window win, int bval, xtb_data Val)
{ xtb_hret r;
	if( Val== &chdir ){
		r= SimpleFileDialog( AF(DIR_F).win, win, bval, Val, False);
		return HO_df_fun( AF(DIR_F).win, XK_Up, "", Val);
	}
	else if( Val== &OfileDev ){
	  char cwd[MAXCHBUF], file[MAXCHBUF], *c= cwd, *f= file;
		r= SimpleFileDialog( AF(FDINP_F).win, win, bval, Val, False);
		xtb_ti_get( AF(DIR_F).win, cwd, 0);
		xtb_ti_get( AF(FDINP_F).win, file, 0);
		  /* strip the common part between the current working dir and the selected filename: */
		while( *c && *f && *c== *f ){
			c++, f++;
		}
		  /* If the cwd is not empty, and the filename starts with a /, strip it too! */
		if( *cwd && *f== '/' ){
			f++;
		}
		xtb_ti_set( AF(FDINP_F).win, f, 0);
		// 20101104:
		return( XTB_HANDLED );
	}
	else{
		return( XTB_NOTDEF );
	}
}

/*ARGSUSED*/
static xtb_hret df_fun( Window win, int ch, char *text, xtb_data Val)
/*
 * This is the handler function for the text widget for
 * specifing the file or device name.  It supports simple
 * line editing operations.
 */
{ char Text[MAXCHBUF];
  int changed= 0;
  int dev, fod, accept;
  void *val= Val;

	if( !theWin_Info ){
		return( XTB_STOP );
	}

    dev = xtb_br_get(HO_info->choices);
	fod = xtb_br_get(HO_info->fod);

    if ((ch == BACKSPACE) || (ch == DELETE)) {
		if( !( (ch== DELETE)? xtb_ti_dch_right(win) : xtb_ti_dch(win)) )
			Boing( 5);
		changed= 1;
		return( df_fun( win, 0, text, val) );
    }
	else if ((ch == CONTROL_U) || (ch == CONTROL_X)) {
		(void) xtb_ti_set(win, "", (xtb_data) 0);
		changed= 1;
		return( df_fun( win, 0, text, val) );
    }
	else if( ch== TAB){
		df_fun( win, 0, text, val );
		changed= 1;
		return( goto_next_text_box( win) );
	}
	else if( ch== CONTROL_P){
		df_fun( win, 0, text, val );
		changed= 1;
		return( goto_prev_text_box( win) );
	}
	else if( ch== CONTROL_W){
	  char *str;
		if( *text){
			str= &text[ strlen(text)-1 ];
		}
		else{
			Boing( 5);
			return( df_fun( win, 0, text, val) );
		}
		if( index( word_sep, *str) ){
			if( !xtb_ti_dch(win) ){
				Boing( 5);
			}
			return( df_fun( win, 0, text, val) );
		}
		while( *str && !index( word_sep, *str) ){
			if( !xtb_ti_dch(win) ){
				Boing( 5);
				return( df_fun( win, 0, text, val) );
			}
			str--;
		}
		changed= 1;
	}
	else if( ch &&
		ch!= TAB && ch!= XK_Down && ch!= XK_Up && ch!= 0x12 &&
		ch!= XK_Meta_L && ch!= XK_Meta_R && ch!= XK_Alt_L && ch!= XK_Alt_R &&
		ch!= XK_Super_L && ch!= XK_Super_R && ch!= XK_Hyper_L && ch!= XK_Hyper_R
	){
	  /* Insert if printable - ascii dependent */
		if( val== &OfileDev && (fod== 1 || fod== 2) ){
		 /* No spaces in filenames. Actually, this is only to be able to
		  \ export them correctly (without hassle with quotes).
			if( ch== ' ' && text[0]!= '|' ){
				ch= '_';
			}
			 (20040609: now done while copying into UPrintFileName)
		  */
		}
		if ((ch < ' ') /* || (ch >= DELETE)*/ || !xtb_ti_ins(win, ch)) {
			Boing( 5);
		}
		else{
			changed= 1;
		}
    }
	xtb_ti_get( win, Text, (xtb_data) NULL );
	if( val== &OfileDev && (fod== 1 || fod== 2)){
	  char *c= &UPrintFileName[0], *d= Text;
	  extern char *rindex();
		while( *d ){
			if( *d== ' ' && Text[0]!= '|' ){
				*c= '_';
			}
			else{
				*c = *d;
			}
			c++, d++;
		}
		*c= '\0';
		if( (c= rindex( UPrintFileName, '.')) ){
			*c= '\0';
		}
		if( ch== 0 && debugFlag ){
			fprintf( StdErr, "UPrintFileName=\"%s\"\n", UPrintFileName );
		}
	}
	if( (accept= ( ch== TAB || ch== XK_Down || ch== XK_Up || ch== 0x12 )) ){
		if( val== &titleFont ){
			if( device_nr== XGRAPH_DEVICE && strcmp( titleFont.name, Text) ){
				if( New_XGFont( 'TITL', Text) ){
					if( strcmp( Text, titleFont.name ) ){
						xtb_ti_set( win, titleFont.name, 0 );
					}
					changed= 1;
				}
			}
		}
		else if( val== &legendFont ){
			if( device_nr== XGRAPH_DEVICE && strcmp( legendFont.name, Text) ){
				if( New_XGFont( 'LEGN', Text) ){
					if( strcmp( Text, legendFont.name ) ){
						xtb_ti_set( win, legendFont.name, 0 );
					}
					changed= 1;
				}
			}
		}
		else if( val== &labelFont ){
			if( device_nr== XGRAPH_DEVICE && strcmp( labelFont.name, Text) ){
				if( New_XGFont( 'LABL', Text) ){
					if( strcmp( Text, labelFont.name ) ){
						xtb_ti_set( win, labelFont.name, 0 );
					}
					changed= 1;
				}
			}
		}
		else if( val== &axisFont ){
			if( device_nr== XGRAPH_DEVICE && strcmp( axisFont.name, Text) ){
				if( New_XGFont( 'AXIS', Text) ){
					if( strcmp( Text, axisFont.name ) ){
						xtb_ti_set( win, axisFont.name, 0 );
					}
					changed= 1;
				}
			}
		}
		if( changed ){
			if( changed== 1 ){
				theWin_Info->redraw= 1;
			}
			theWin_Info->printed= 0;
		}
	}
    return XTB_HANDLED;
}

int Isdigit( unsigned int ch )
{
	if( (ch & 0xFFFFFF00) ){
	  /* no byte	*/
		return(0);
	}
	else{
		return( isdigit((unsigned char)ch) );
	}
}

static xtb_hret HO_df_fun(Window win, int ch, char *text, xtb_data val)
{ char Text[MAXCHBUF];
  int accept;

	if( !theWin_Info ){
		return( XTB_STOP );
	}

    if ((ch == BACKSPACE) || (ch == DELETE)) {
		if( !( (ch== DELETE)? xtb_ti_dch_right(win) : xtb_ti_dch(win)) )
			Boing(5);
		return XTB_HANDLED;
    }
	else if ((ch == CONTROL_U) || (ch == CONTROL_X)) {
		xtb_ti_set(win, "", (xtb_data) 0);
		return XTB_HANDLED;
    }
	else if( ch== TAB){
		( goto_next_text_box( win) );
	}
	else if( ch== CONTROL_P){
		return( goto_prev_text_box( win) );
	}
	else if( ch== CONTROL_W){
	  char *str;
		if( *text)
			str= &text[ strlen(text)-1 ];
		else{
			Boing( 5);
			return( XTB_HANDLED);
		}
		if( index( word_sep, *str) ){
			if( !xtb_ti_dch(win) ){
				Boing( 5);
			}
			return( XTB_HANDLED);
		}
		while( *str && !index( word_sep, *str) ){
			if( !xtb_ti_dch(win) ){
				Boing( 5);
				return( XTB_HANDLED);
			}
			str--;
		}
	}
	else if( ch!= XK_Down && ch!= XK_Up && ch!= 0x12 && ch!= TAB &&
		ch!= XK_Meta_L && ch!= XK_Meta_R && ch!= XK_Alt_L && ch!= XK_Alt_R &&
		ch!= XK_Super_L && ch!= XK_Super_R && ch!= XK_Hyper_L && ch!= XK_Hyper_R
	){
		if( !xtb_ti_ins(win, ch) ){
			Boing( 5);
		}
    }
	if( (accept= ( ch== TAB || ch== XK_Down || ch== XK_Up || ch== 0x12 )) ){
		xtb_ti_get( win, Text, (xtb_data) NULL );
		if( val== &chdir ){
		  char *c, exp[MAXCHBUF*2];
		  extern char *tildeExpand( char*, const char *);
			if( chdir( tildeExpand( exp, Text) ) ){
				xtb_error_box( theWin_Info->window, Text, "Warning: can't chdir");
			}
			c= getcwd( Text, MAXCHBUF);
			if( !c ){
				c= serror();
			}
			xtb_ti_set( win, c, NULL );
		}
		else{
			Boing(5);
		}
	}
    return XTB_HANDLED;
}

int set_width, set_height;

int StringCheck( char *s, int maxlen, char *file, int line )
{  int len= strlen(s);
	if( len> maxlen ){
		fprintf( StdErr, "StringCheck(): possibly fatal error at %s::%d - %d bytes in string of %d\n",
			file, line, len, maxlen
		);
		fflush( StdErr );
		return(1);
	}
#ifdef DEBUG
	else if( len== maxlen ){
		fprintf( StdErr, "StringCheck(): warning at %s::%d - %d bytes in string of %d\n",
			file, line, len, maxlen
		);
		fflush( StdErr );
		return(-1);
	}
#endif
	return(0);
}
#define STRINGCHECK(s,max)	StringCheck(s,max,__FILE__,__LINE__)

extern char *d3str( char *, char *, double), d3str_format[];

/*ARGSUSED*/
static xtb_hret dfn_fun(win, ch, text, val)
Window win;			/* Widget window   */
int ch;				/* Typed character */
char *text;			/* Copy of text    */
xtb_data val;			/* User info       */
/*
 * This is the handler function for the text widget for
 * specifing a number.  It supports simple
 * line editing operations.
 */
{ double value;
  char number[MAXCHBUF];
  int changed= 0, accept;
  extern double cus_log10X(), cus_log10Y(), Reform_X(), Reform_Y();

	if( !theWin_Info ){
		return( XTB_STOP );
	}

    if ((ch == BACKSPACE) || (ch == DELETE)) {
		if( !( (ch== DELETE)? xtb_ti_dch_right(win) : xtb_ti_dch(win)) )
			Boing(5);
		return XTB_HANDLED;
    }
	else if ((ch == CONTROL_U) || (ch == CONTROL_X)) {
		xtb_ti_set(win, "", (xtb_data) 0);
		return XTB_HANDLED;
    }
	else if( ch== TAB){
		( goto_next_text_box( win) );
	}
	else if( ch== CONTROL_P){
		return( goto_prev_text_box( win) );
	}
	else if( ch== CONTROL_W){
	  char *str;
		if( *text)
			str= &text[ strlen(text)-1 ];
		else{
			Boing( 5);
			return( XTB_HANDLED);
		}
		if( index( word_sep, *str) ){
			if( !xtb_ti_dch(win) ){
				Boing( 5);
			}
			return( XTB_HANDLED);
		}
		while( *str && !index( word_sep, *str) ){
			if( !xtb_ti_dch(win) ){
				Boing( 5);
				return( XTB_HANDLED);
			}
			str--;
		}
	}
	else if( ch== XK_Up ){
		xtb_ti_get( win, number, (xtb_data) NULL );
		if( sscanf( number, "%lf", &value) ){
			value+= 1.0;
			sprintf( number, "%g", value);
			STRINGCHECK( number, MAXCHBUF );
			xtb_ti_set( win, number, (xtb_data) 0);
		}
		else{
			Boing(1);
		}
	}
	else if( ch== XK_Down ){
		xtb_ti_get( win, number, (xtb_data) NULL );
		if( sscanf( number, "%lf", &value) ){
			value-= 1.0;
			sprintf( number, "%g", value);
			STRINGCHECK( number, MAXCHBUF );
			xtb_ti_set( win, number, (xtb_data) 0);
		}
		else{
			Boing(1);
		}
	}
	else if( ch!= ' ' &&
		ch!= XK_Meta_L && ch!= XK_Meta_R && ch!= XK_Alt_L && ch!= XK_Alt_R &&
		ch!= XK_Super_L && ch!= XK_Super_R && ch!= XK_Hyper_L && ch!= XK_Hyper_R
	){
	  /* Insert if valid for a number */
		if( !xtb_ti_ins(win, ch)) {
			Boing( 5);
		}
		xtb_ti_get( win, number, (xtb_data) NULL );
		if( !sscanf( number, "%lf", &value) ){
			value= 0.0;
		}
    }
	if( (accept= ( ch== TAB || ch== ' ' || ch== XK_Down || ch== XK_Up || ch== 0x12 )) ){
	  int n= 1;
		xtb_ti_get( win, number, (xtb_data) NULL );
			if( !fascanf( &n, number, &value, NULL, NULL, NULL, NULL) ){
				value= 0.0;
			}
			d3str( number, d3str_format, value);
			xtb_ti_set( win, number, (xtb_data) 0);
			if( val== &psm_base ){
				if( value> 0 ){
					if( value!= psm_base ){
						psm_changed+= 1;
						psm_base= value;
						changed= 1;
					}
				}
				else{
					Boing(5);
				}
			}
			else if( val== &psm_incr ){
				if( value!= psm_incr ){
					psm_changed+= 1;
					psm_incr= value;
					changed= 1;
				}
			}
			else if( val== &ps_scale ){
				if( value> 0 && value!= theWin_Info->ps_scale ){
					theWin_Info->ps_scale= value;
					changed= 1;
				}
			}
			else if( val== &ps_l_offset ){
				if( value> 0 && value!= theWin_Info->ps_l_offset ){
					theWin_Info->ps_l_offset= value;
					changed= 1;
				}
			}
			else if( val== &ps_b_offset ){
				if( value> 0 && value!= theWin_Info->ps_b_offset ){
					theWin_Info->ps_b_offset= value;
					changed= 1;
				}
			}
			else if( val== &set_width ){
				if( theWin_Info->hard_devices[device_nr].dev_max_width != value ){
					theWin_Info->hard_devices[device_nr].dev_max_width = value;
					changed= 2;
				}
			}
			else if( val== &set_height ){
				if( theWin_Info->hard_devices[device_nr].dev_max_height != value ){
					theWin_Info->hard_devices[device_nr].dev_max_height = value;
					changed= 2;
				}
			}
			else if( val== &tf_size ){
				if( theWin_Info->hard_devices[device_nr].dev_title_size != value ){
					theWin_Info->hard_devices[device_nr].dev_title_size = value;
					changed= 1;
				}
			}
			else if( val== &lef_size ){
				if( theWin_Info->hard_devices[device_nr].dev_legend_size != value ){
					theWin_Info->hard_devices[device_nr].dev_legend_size = value;
					changed= 1;
				}
			}
			else if( val== &laf_size ){
				if( theWin_Info->hard_devices[device_nr].dev_label_size != value ){
					theWin_Info->hard_devices[device_nr].dev_label_size = value;
					changed= 1;
				}
			}
			else if( val== &af_size ){
				if( theWin_Info->hard_devices[device_nr].dev_axis_size != value ){
					theWin_Info->hard_devices[device_nr].dev_axis_size = value;
					changed= 1;
				}
			}
			else{
				Boing(5);
			}
	}
	if( changed ){
		if( changed== 1 ){
			theWin_Info->redraw= 1;
		}
		theWin_Info->printed= 0;
	}
    return XTB_HANDLED;
}

static xtb_hret orient_fun(win, old, new, info)
Window win;			/* Button row window */
int old;			/* Previous button   */
int new;			/* Current button    */
xtb_data info;			/* User data         */
{ int or= xtb_br_get( AF(PSORIENT_F).win );
	if( !theWin_Info ){
		return( XTB_STOP );
	}

	if( or>= 0 ){
		theWin_Info->print_orientation= or;
	}
	return( XTB_HANDLED);
}

static xtb_hret pos_fun(win, old, new, info)
Window win;			/* Button row window */
int old;			/* Previous button   */
int new;			/* Current button    */
xtb_data info;			/* User data         */
{ int or= (info== &ps_xpos)? xtb_br_get( AF(PSPOSX_F).win ) : xtb_br_get( AF(PSPOSY_F).win );
	if( !theWin_Info ){
		return( XTB_STOP );
	}

	if( CheckMask(xtb_modifier_state, Mod1Mask) && or<= 0 ){
	  char ebuf[64], message[256], *nbuf;
	  double *target;
		if( info== &ps_xpos ){
			sprintf( ebuf, "%s", d2str(theWin_Info->ps_l_offset, NULL, NULL) );
			sprintf( message, " Enter the offset (in points) from the left margin for left-aligned printing:");
			target= &theWin_Info->ps_l_offset;
			  /* Don't change the selected field in this case ;-)	*/
			xtb_br_set( win, theWin_Info->ps_xpos );
		}
		else{
			sprintf( ebuf, "%s", d2str(theWin_Info->ps_b_offset, NULL, NULL) );
			sprintf( message, " Enter the offset (in points) from the bottom margin for bottom-aligned printing:");
			target= &theWin_Info->ps_b_offset;
			xtb_br_set( win, theWin_Info->ps_ypos );
		}
		if( (nbuf= xtb_input_dialog( HO_Dialog.win, ebuf, 16, 64, message, "Enter a number", False, "", NULL, "", NULL, NULL, NULL )) ){
		  double x;
			if( sscanf( ebuf, "%lf", &x)== 1 ){
				if( *target!= x ){
					set_HO_printit_win();
				}
				*target= x;
			}
			else{
				xtb_error_box( theWin_Info->window, ebuf, "Error: can't parse floating point number" );
			}
			if( nbuf!= ebuf ){
				xfree( nbuf );
			}
		}
	}
	else if( or>= 0 ){
		if( info== &ps_xpos ){
			theWin_Info->ps_xpos= or;
		}
		else{
			theWin_Info->ps_ypos= or;
		}
	}
	return( XTB_HANDLED);
}

static char *PRSN_desc=
	"To print/save any <n> sets in as many files as necessary, enter\n"
	" a value <n> in the field below. In the filename, the first printf(2)\n"
	" format field (e.g. %03d) will be replaced by the file sequential number\n"
	" starting at 0.\n"
	" Hence, n=1 will cause all datasets to be printed/saved in separate files.\n"
	" A value n<=0 will deactivate this feature.\n"
	" NB: No check is made against multiple printf() format fields!\n"
;

static xtb_hret PrintSetsNR_fun( Window win, int old, int new, xtb_data info)
{ char ebuf[64], *nbuf;
	if( !theWin_Info ){
		return( XTB_STOP );
	}

	sprintf( ebuf, "%d", PrintSetsNR );
	if( (nbuf= 
		xtb_input_dialog( HO_Dialog.win, ebuf, 16, sizeof(ebuf)*sizeof(char), 
			PRSN_desc, "Enter a number", False, "", NULL, "", NULL, NULL, NULL )
		)
	){
	  int x;
		if( sscanf( nbuf, "%d", &x)== 1 ){
			PrintSetsNR= x;
			xtb_bt_set( AF(SETS_F).win, (PrintSetsNR<=0)? 0 : 1, (xtb_data) 0 );
		}
		else{
			xtb_error_box( theWin_Info->window, nbuf, "Error: can't parse decimal number" );
		}
		if( nbuf!= ebuf ){
			xfree( nbuf );
		}
	}
	return( XTB_HANDLED);
}

#include <stdio.h>

extern int no_legend, no_title, legend_always_visible, logXFlag, logYFlag, sqrtXFlag, sqrtYFlag;

extern xtb_hret HO_dev_fun(Window win, int previous, int current, xtb_data info);

Boolean Close_OK;

extern int XGDump_PrintPars, XGDump_AllWindows, DumpKeyParams, Use_HO_Previous_TC;
extern int DumpBinary, DumpDHex, DumpPens, DProcFresh, DumpAsAscanf;

extern double window_width, window_height;
extern double data_width, data_height;

int XG_NUp_X= 1, XG_NUp_Y= 1, XG_NUp_aspect= 0;
double XG_NUp_scale= 1;
int XG_NUp_scale_W= 1, XG_NUp_scale_H= 1;

/*ARGSUSED*/
xtb_hret HO_ok_fun( Window win, int bval, xtb_data info)
/*
 * This is the handler function for when the `Ok' button
 * is hit.  It sets the button,  does the hardcopy output,
 * and turns the button off.  It returns a status which
 * deactivates the dialog.
 */
{
    struct ho_d_info *real_info = (struct ho_d_info *) info;
    int val, dev_p, append= 1, do_iconify;
    char *f_o_d, file_or_dev[MAXCHBUF], hdim_spec[MAXCHBUF], wdim_spec[MAXCHBUF], *dev_spec, *viewcommand= NULL;
    char tfam[MAXCHBUF], lefam[MAXCHBUF], lafam[MAXCHBUF], afam[MAXCHBUF];
    char tsizstr[MAXCHBUF], lesizstr[MAXCHBUF], lasizstr[MAXCHBUF], asizstr[MAXCHBUF], *HO_oktext= NULL;
    double hcentimeters, wcentimeters, hcent_corr= 0, wcent_corr= 0, hcent_corr_factor= 1, wcent_corr_factor= 1,
		tsize, lesize, lasize, asize;
    xtb_hret rtn;
	int (*dev_init)();
	extern Boolean PIPE_error;
	static char active= False, gs_batching= False;

	if( !theWin_Info ){
		return( XTB_STOP );
	}
	if( active ){
		Boing(10);
		return( XTB_STOP );
	}

	xtb_bt_set2( AF(XGINI_F).win, Init_XG_Dump, XG_Really_Incomplete, NULL);

    xtb_bt_set(win, 1, (xtb_data) 0);
    val = xtb_br_get(real_info->choices);
	do_iconify= ( CheckMask( xtb_modifier_state, ShiftMask) )? True : False;

	if( use_gsTextWidth ){
		if( theWin_Info->textrel.gs_batch ){
			val= PS_DEVICE;
		}
		else if( auto_gsTextWidth && val== PS_DEVICE && !theWin_Info->printed ){
			gs_batching= True;
			gsTextWidth_Batch();
			gs_batching= False;
		}
	}
	active= True;
	Close_OK= False;
    if( (val >= 0) && (val < hard_count) ){
		  /* dev_p: the button-row number	*/
		dev_p = xtb_br_get(real_info->fod);
		if( use_gsTextWidth && theWin_Info->textrel.gs_batch ){
			dev_p= 2;
		}
		if ((dev_p == 0) || (dev_p == 1) || (dev_p==2) || (dev_p== 3) ) {
			if( use_gsTextWidth && theWin_Info->textrel.gs_batch ){
				strcpy( file_or_dev, "/dev/null");
			}
			else{
				xtb_ti_get(real_info->fodspec, file_or_dev, (xtb_data *) 0);
			}
			xtb_ti_get(real_info->height_dimspec, hdim_spec, (xtb_data *) 0);
			xtb_ti_get(real_info->width_dimspec, wdim_spec, (xtb_data *) 0);
			if (sscanf( hdim_spec, "%lf", &hcentimeters) == 1 &&
				sscanf( wdim_spec, "%lf", &wcentimeters) == 1 
			) {
				xtb_ti_get(real_info->tf_family, tfam, (xtb_data *) 0);
				xtb_ti_get(real_info->lef_family, lefam, (xtb_data *) 0);
				xtb_ti_get(real_info->laf_family, lafam, (xtb_data *) 0);
				xtb_ti_get(real_info->af_family, afam, (xtb_data *) 0);
				xtb_ti_get(real_info->tf_size, tsizstr, (xtb_data *) 0);
				if (sscanf(tsizstr, "%lf", &tsize) == 1 || val== XGRAPH_DEVICE ) {
					xtb_ti_get(real_info->lef_size, lesizstr, (xtb_data *) 0);
					xtb_ti_get(real_info->laf_size, lasizstr, (xtb_data *) 0);
					if( (sscanf(lesizstr, "%lf", &lesize) == 1 &&
						sscanf(lasizstr, "%lf", &lasize)== 1) || val== XGRAPH_DEVICE
					) {
						xtb_ti_get(real_info->af_size, asizstr, (xtb_data *) 0);
						if( sscanf(asizstr, "%lf", &asize) == 1 || val== XGRAPH_DEVICE ) {
						  int free_fod= 0;
						  int sp= showpage, pspn= ps_page_nr;
						  int xdpp= XGDump_PrintPars;
						  ALLOCA( ldraw_set, short, setNumber, ds_len );
						  int loop= -1, ostrip, fileNum= 0, firstDrawn, numDrawn;
						  char *lfod= NULL, *HO_wmtitle;
							XFetchName( disp, HO_Dialog.win, &HO_wmtitle );
							HO_oktext= XGstrdup( xtb_bt_get_text(win) );
							/* Got all the info */
							if( dev_p && dev_p!= 3 ) {
								dev_spec = (char *) 0;
								if( dev_p== 2)
									append= 0;
							} else {
								dev_spec = theWin_Info->hard_devices[val].dev_spec;
							}
							f_o_d= file_or_dev;
							dev_init= theWin_Info->hard_devices[val].dev_init;
							if( XGDump_PrintPars ){
								XGDump_PrintPars= 2;
							}
							if( !f_o_d || ! *f_o_d ||
								(val== XGRAPH_DEVICE && dev_p== 3 && f_o_d && strcmp( f_o_d, "xgraph.pspreview")== 0)
							){
							  /* blank field. So we pick a sensible default.	*/
								switch( val ){
									case PS_DEVICE:
										if( dev_p== 0 ){
											f_o_d= "atprint";
										}
										else if( dev_p== 3 ){
											f_o_d= "xgraph.pspreview";
										}
										else{
										  /* We might look at PrintFileName in this case...	*/
											f_o_d= "xgraph.ps";
										}
										break;
									case SPREADSHEET_DEVICE:
										f_o_d= (dev_p!= 0 && dev_p!= 3)? "xgraph.csv" : "cat";
										break;
									case CRICKET_DEVICE:
										f_o_d= (dev_p!= 0 && dev_p!= 3)? "xgraph.cg" : "cat";
										break;
#ifdef HPGL_DUM
									case HPGL_DEVICE:
										f_o_d= (dev_p!= 0 && dev_p!= 3)? "xgraph.hpgl" : "cat";
										break;
#endif
#ifdef IDRAW_DEVICE
									case IDRAW_DEVICE:
										f_o_d= (dev_p!= 0 && dev_p!= 3)? "~/.clipboard" : "cat";
										break;
#endif
									case COMMAND_DEVICE:
										if( dev_p!= 0 && dev_p!= 3 ){
											f_o_d= "xgraph.sh";
											break;
										}
										else if( dev_p== 3 ){
											f_o_d= "sh";
											break;
										}
										else{
											dev_init= theWin_Info->hard_devices[XGRAPH_DEVICE].dev_init;
										 /* fall through for direct printing or previewing.	*/
										}
									case XGRAPH_DEVICE:{
									  extern char *PrintFileName;
/* 									  extern char **Argv;	*/
/* 									  char *self= (Argv[0])? Argv[0] : "xgraph";	*/
									  char self[]= "XGraph -";
									  char *buf;
										if( dev_p== 0 ){
										  /* User requests a (direct) print of an XGraph dump. So we pipe the
										   \ output to a copy of ourselves, with direction to print immediately as
										   \ PostScript. The use of the specified PrintFileName is maybe useful
										   \ when something goes wrong...
										   */
											if( !XGDump_PrintPars ){
												XGDump_PrintPars= 1;
											}
											if( UPrintFileName[0] ){
												buf= concat( "env \"Disposition=To Device\" ",
													self, " -print_as ", theWin_Info->hard_devices[PS_DEVICE].dev_name, " -pf ",
													UPrintFileName, " -printOK ", NULL
												);
											}
											else if( PrintFileName ){
												buf= concat( "env \"Disposition=To Device\" ",
													self, " -print_as ", theWin_Info->hard_devices[PS_DEVICE].dev_name, " -pf ",
													PrintFileName, " -printOK ", NULL
												);
											}
											else{
												buf= concat( "env \"Disposition=To Device\" ",
													self, " -print_as ", theWin_Info->hard_devices[PS_DEVICE].dev_name, " -printOK ", NULL
												);
											}
											free_fod= 1;
											f_o_d= buf;
										}
										else if( dev_p== 3 ){
										  /* Previewing an XGraph dump is best done using a copy of ourselves. We
										   \ pass on the specified PrintFileName in the commandline, the (=most) other
										   \ arguments are set in _XGraphdump(), and passed in the dump. The env.var
										   \ XGRAPH_ARGUMENTS is set to "" to prevent interference with arguments possibly
										   \ passed through this variable (these override commandline & file arguments).
										   */
										  char *geo= ( theWin_Info->dev_info.resized== 1 )?  " -print_sized" : "";
											if( !XGDump_PrintPars ){
												XGDump_PrintPars= 1;
											}
											if( UPrintFileName[0] ){
												f_o_d= concat( "env XGRAPH_ARGUMENTS= ", self, " -pf ", UPrintFileName, geo, NULL);
												free_fod= 1;
											}
											else if( PrintFileName ){
												f_o_d= concat( "env XGRAPH_ARGUMENTS= ", self, " -pf ", PrintFileName, geo, NULL);
												free_fod= 1;
											}
											else{
												f_o_d= concat( "env XGRAPH_ARGUMENTS= ", self, geo, NULL);
												free_fod= 1;
											}
										}
										else if( !f_o_d ){
											f_o_d= "xgraph.xg";
										}
										break;
									}
								}
								xtb_ti_set( real_info->fodspec, f_o_d, (xtb_data *) 0);
							}
							if( dev_p== 3 ){
								if( val== XGRAPH_DEVICE || val== COMMAND_DEVICE ){
								  /* This should create a new copy of the current window	*/
									Close_OK= True;
								}
								else{
								  /* prevent accidentily closing the window while previewing e.g. a PS dump	*/
									Close_OK= False;
								}
							}
							else{
								Close_OK= True;
							}
							PIPE_error= False;
							if( !dev_spec ){
								if( (viewcommand= strstr( f_o_d, " | ")) || (viewcommand= strstr( f_o_d, "_|_")) ){
									viewcommand[0]= '\0';
									viewcommand+= 3;
								}
								  /* 20040609: do space substitution here, when necessary */
								if( f_o_d[0]!= '|' ){
									substitute( f_o_d, ' ', '_' );
								}
							}
							if( PrintSetsNR> 0
								/* && index( f_o_d, '%') && (val== XGRAPH_DEVICE) && dev_p== 2 */
							){
							  int i=0;
								memcpy( ldraw_set, theWin_Info->draw_set, setNumber* sizeof(short) );
								ostrip= XG_Stripped;
								XG_Stripped= True;
								loop= 1;
								  /* 20030304: quick hack to support 'subsetting', = printing/saving
								   \ a specified number of datasets in as many files to save them all.
								   */
								lfod= f_o_d;
								f_o_d= calloc( MAXPATHLEN, sizeof(char) );
								for( i= 0, firstDrawn= -1; i< setNumber; i++ ){
									if( draw_set(theWin_Info,i) ){
										if( ostrip && firstDrawn< 0 ){
										  /* be sure to find the true first displayed set */
											firstDrawn= i;
										}
										numDrawn+= 1;
									}
								}
								if( !ostrip ){
									firstDrawn= 0;
								}
								if( !numDrawn ){
									loop= 0;
								}
							}
							else{
								loop= -1;
								lfod= NULL;
							}
							if( val== PS_DEVICE ){
							  /* 20040610: a correction to get the exact (?) requested sizes: */
								hcent_corr= -1;
								wcent_corr= -1;
							}
							while( loop ){
							  int lastSet= 0, dumped= 0, last_cycle, last_accepted;
								if( lfod ){
								  int i, j, r= 0;
									snprintf( f_o_d, MAXPATHLEN, lfod, fileNum );
									if( strcmp( f_o_d, lfod)== 0 && dev_p== 2 && fileNum ){
										if( debugFlag ){
											fprintf( StdErr,
												"HO_ok_fun(): non-changing filename '%s' #%d in subsetting,New file mode: "
												"switched to Append mode.\n",
												f_o_d, fileNum
											);
										}
										append= dev_p= 1;
									}
									memset( theWin_Info->draw_set, 0, setNumber*sizeof(short) );
									if( (i= (loop-1)* PrintSetsNR+ firstDrawn)< setNumber ){
										if( i+ PrintSetsNR>= setNumber ){
										  /* during the last cycle, we accept dumping less than the requested
										   \ number of sets. Note that this code is almost sure not to handle
										   \ all possible combinations of total number of sets, drawn sets and
										   \ to-be-dumped sets... (But it does appear to handle the most current
										   \ cases :)).
										   */
											last_cycle= True;
											  /* How much sets ought to be dumped in the last cycle: */
											if( numDrawn> PrintSetsNR ){
												if( (last_accepted= numDrawn % PrintSetsNR)== 0 ){
													last_accepted= PrintSetsNR;
												}
											}
											else if( numDrawn< PrintSetsNR ){
												if( (last_accepted= PrintSetsNR % numDrawn)== 0 ){
													last_accepted= PrintSetsNR;
												}
											}
											else{
												last_accepted= PrintSetsNR;
											}
										}
										else{
											last_cycle= False;
											last_accepted= PrintSetsNR;
										}
										for( j= 0; j< PrintSetsNR && j< setNumber && i< setNumber; i++, j++ ){
											if( ostrip ){
											  /* if XG_Stripped was set, dump only the sets that were shown */
												DRAW_SET( theWin_Info, ldraw_set, i, r );
												if( r ){
													theWin_Info->draw_set[ (lastSet=i) ] = True;
													dumped+= 1;
												}
											}
											else{
												theWin_Info->draw_set[ (lastSet=i) ] = True;
												dumped+= 1;
											}
										}
									}
									else{
									  /* This is the last cycle. */
										loop= 0;
									}
								}
								else{
									dumped= -1;
								}
								if( !(dumped== -1 || dumped== PrintSetsNR || (dumped== last_accepted && last_cycle)) ){
									goto next_loop;
								}
								  /* 20040615: do this in the medium-specific initialisation routine!
									ps_xpos= theWin_Info->ps_xpos;
									ps_ypos= theWin_Info->ps_ypos;
									ps_scale= theWin_Info->ps_scale;
									ps_l_offset= theWin_Info->ps_l_offset;
									ps_b_offset= theWin_Info->ps_b_offset;
								   */
								if( val== PS_DEVICE && dev_p== 2 ){
								  /* This is to be a new file: reset page counter	*/
									ps_page_nr= 1;
								}

								{ LocalWin *pwi= theWin_Info;
								  LocalWindows *WL= WindowListTail;
								  extern unsigned short psNgraph_wide, psNgraph_high;
								  extern unsigned short psGraph_x_idx, psGraph_y_idx;
								  extern int Use_HO_Previous_TC;
								  unsigned short psNw= psNgraph_wide, psNh= psNgraph_high;
								  int lsp= showpage, psize= psSetPage, uhptc= Use_HO_Previous_TC;
								  double psscl= ps_scale, wpsscl= theWin_Info->ps_scale,
								  	psSHcf= psSetHeight_corr_factor, psSWcf= psSetWidth_corr_factor;
/* 								  char msg[256];	*/
									if( XGDump_AllWindows && (XG_NUp_X> 1 || XG_NUp_Y> 1) && WL ){
										if( val== PS_DEVICE ){
											if( XG_NUp_aspect ){
												psNgraph_wide= psNgraph_high= MAX(XG_NUp_X,XG_NUp_Y);
												  /* to preserve aspect ratios, we impose an even (nxn) N-Up layout, and
												   \ we correct the printing width or height accordingly. E.g, a 2x4
												   \ layout will become a 4x4 layout, so we'll have to double the printing
												   \ width in order to preserve the original aspect ratio.
												   \ Of course, this supposes that psGraph_{x,y}_idx are controlled by
												   \ XG_NUp_{X,Y}, not by psNgraph_{wide,high}!
												   \ NB: The printing width/height adjustment is only done in 2 of the 4
												   \ possible cases, for the moment.
												   \ IT IS NOT CLEAR TO WHAT EXTENT THIS IS EVEN NECESSARY. (Too lame to
												   \ figure that out now.)
												   */
												if(
													  /* portrait, and a "tall/thin" layout */
													(!theWin_Info->print_orientation && XG_NUp_Y> XG_NUp_X) ||
													  /* landscape, and a "wide/thin" layout */
													(theWin_Info->print_orientation && XG_NUp_X> XG_NUp_Y)
												){
													wcent_corr_factor= psNgraph_wide/XG_NUp_X;
													hcent_corr_factor= psNgraph_high/XG_NUp_Y;
													  /* 20040611: This will give problems if we output code that will cause the
													   \ postscript "page" size to be adapted to the printed graph. In that case,
													   \ we'll have to apply the inverse correction. In the above example, that
													   \ would mean that we'll set the page width to half the printing width --
													   \ because the actual width *used* is only half the specified width.
													   \ We set these variables even if psSetPage is not set, as they also serve
													   \ as a flag.
													   */
													psSetWidth_corr_factor= 1.0/ wcent_corr_factor;
													psSetHeight_corr_factor= 1.0/ hcent_corr_factor;
												}
											}
											else{
												psNgraph_wide= XG_NUp_X;
												psNgraph_high= XG_NUp_Y;
											}
											psGraph_x_idx= psGraph_y_idx= 0;
											if( uhptc ){
												Use_HO_Previous_TC= False;
											}
											pwi= WL->wi;
											WL= WL->prev;
											showpage= (WL)? 0 : 1;
											if( gs_batching ){
												sprintf( ps_comment, "PS Graph %d:%d fonts...", psGraph_x_idx, psGraph_y_idx );
												xtb_bt_set_text(win, 1, ps_comment, (xtb_data) 0);
											}
											else{
												sprintf( ps_comment, "PS Graph %d:%d...", psGraph_x_idx, psGraph_y_idx );
												xtb_bt_set_text(win, 1, ps_comment, (xtb_data) 0);
											}
											XStoreName(disp, HO_Dialog.win, ps_comment);
											if( !RemoteConnection ){
												XSync( disp, False );
											}
										}
									}

									if( XGDump_AllWindows ){
										if( XG_NUp_scale_W ){
											hcentimeters= hcentimeters* hcent_corr_factor/ XG_NUp_scale+ hcent_corr;
										}
										else{
											hcentimeters= hcentimeters* hcent_corr_factor+ hcent_corr;
										}
										if( XG_NUp_scale_H ){
											wcentimeters= wcentimeters* wcent_corr_factor/ XG_NUp_scale+ wcent_corr;
										}
										else{
											wcentimeters= wcentimeters* wcent_corr_factor+ wcent_corr;
										}
									}

									ps_scale*= XG_NUp_scale;
									theWin_Info->ps_scale*= XG_NUp_scale;

									while( pwi ){
									  int gsb= pwi->textrel.gs_batch;
									  int _ps_xpos= pwi->ps_xpos, _ps_ypos= pwi->ps_ypos;
									  double _ps_scale= pwi->ps_scale, _ps_l_offset= pwi->ps_l_offset,
											  _ps_b_offset= pwi->ps_b_offset;

										if( pwi!= theWin_Info ){
											 /* we always force identical ps_scale values when printing all windows */
											pwi->ps_scale= theWin_Info->ps_scale;
											if( Use_HO_Previous_TC ){
												  /* 20040615: use identical settings for all windows,
												   \ positional settings only when sizes should be really identical
												   */
												pwi->ps_xpos= theWin_Info->ps_xpos;
												pwi->ps_ypos= theWin_Info->ps_ypos;
												pwi->ps_l_offset= theWin_Info->ps_l_offset;
												pwi->ps_b_offset= theWin_Info->ps_b_offset;
											}
										}
										if( gs_batching ){
										  /* 20040610:
										   \ must (temporarily) set this flag to really activate batch mode for this win!
										   */
											pwi->textrel.gs_batch= True;
										}
										real_info->errnr= do_hardcopy(real_info->prog, pwi,
												dev_init, dev_spec,
												f_o_d, append, &hcentimeters, &wcentimeters, theWin_Info->print_orientation,
												tfam, tsize, lefam, lesize, lafam, lasize, afam, asize
										);

										if( PIPE_error ){
										  char err[1024];
											if( dev_spec ){
												sprintf(err, "Error issuing command:\n  \"%s\",%s\n", dev_spec, f_o_d );
											}
											else{
												sprintf(err, "Error issuing command:\n  ... %s\n", f_o_d );
											}
											do_error( err);
											real_info->errnr= -1;
										}

										if( use_gsTextWidth && pwi->textrel.gs_batch ){
										  char msg[256], *wn= NULL;
											sprintf( msg, "Determining %d widths with gs...", pwi->textrel.gs_batch_items );
											XFetchName( disp, HO_Dialog.win, &wn );
											XStoreName(disp, HO_Dialog.win, msg);
											if( !RemoteConnection ){
												XSync( disp, False );
											}
											last_gsTextWidthBatch= gsTextWidthBatch( pwi );
											if( wn ){
												XStoreName(disp, HO_Dialog.win, wn);
												XFree( wn );
											}
											if( !RemoteConnection ){
												XSync( disp, False );
											}
										}
										pwi->textrel.gs_batch= gsb;
										if( pwi!= theWin_Info ){
											pwi->ps_scale= _ps_scale;
											if( Use_HO_Previous_TC ){
												  /* 20040615: restore */
												pwi->ps_xpos= _ps_xpos;
												pwi->ps_ypos= _ps_ypos;
												pwi->ps_l_offset= _ps_l_offset;
												pwi->ps_b_offset= _ps_b_offset;
											}
										}

										if( XGDump_AllWindows && (XG_NUp_X> 1 || XG_NUp_Y> 1) && WL ){
											if( val== PS_DEVICE ){
												if( HO_printit_win ){
													pwi->printed= 1;
												}
												pwi= WL->wi;
												WL= WL->prev;
												append= dev_p= 1;
												showpage= (pwi && WL)? 0 : 1;
												if( uhptc ){
													Use_HO_Previous_TC= True;
												}
												if( ++psGraph_x_idx>= XG_NUp_X ){
													psGraph_x_idx= 0;
													if( (psGraph_y_idx+= 1)>= XG_NUp_Y ){
														pwi= NULL;
														showpage= 1;
													}
												}
												if( pwi ){
													if( HO_printit_win ){
														xtb_bt_set( HO_printit_win, (pwi->printed= 0), NULL);
													}
													if( gs_batching ){
														sprintf( ps_comment, "PS Graph %d:%d fonts...", psGraph_x_idx, psGraph_y_idx );
														xtb_bt_set_text(win, 1, ps_comment, (xtb_data) 0);
													}
													else{
														sprintf( ps_comment, "PS Graph %d:%d...", psGraph_x_idx, psGraph_y_idx );
														xtb_bt_set_text(win, 1, ps_comment, (xtb_data) 0);
													}
													XStoreName(disp, HO_Dialog.win, ps_comment);
													if( !RemoteConnection ){
														XSync( disp, False );
													}
													psSetPage= False;
												}
											}
											else{
												pwi= NULL;
											}
										}
										else{
											pwi= NULL;
										}
									}
									if( psGraph_x_idx || psGraph_y_idx ){
										sprintf( ps_comment, "Finished PS Graph %d:%d...", psGraph_x_idx, psGraph_y_idx );
										XStoreName(disp, HO_Dialog.win, ps_comment);
										if( !RemoteConnection ){
											XSync( disp, False );
										}
									}
									ps_scale= psscl;
									theWin_Info->ps_scale= wpsscl;
									psNgraph_wide= psNw, psNgraph_high= psNh;
									psGraph_x_idx= psGraph_y_idx= 0;
									showpage= lsp;
									psSetPage= psize;
									Use_HO_Previous_TC= uhptc;
									psSetHeight_corr_factor= psSHcf, psSetWidth_corr_factor= psSWcf;
								}

								if( XGDump_AllWindows ){
									if( XG_NUp_scale_W ){
										hcentimeters= (hcentimeters- hcent_corr)/hcent_corr_factor* XG_NUp_scale;
									}
									else{
										hcentimeters= (hcentimeters- hcent_corr)/hcent_corr_factor;
									}
									if( XG_NUp_scale_H ){
										wcentimeters= (wcentimeters- wcent_corr)/wcent_corr_factor* XG_NUp_scale;
									}
									else{
										wcentimeters= (wcentimeters- wcent_corr)/wcent_corr_factor;
									}
								}

								sprintf( wdim_spec, "%lg", wcentimeters);
								STRINGCHECK( wdim_spec, MAXCHBUF);
								xtb_ti_set( real_info->width_dimspec, wdim_spec, (xtb_data) 0);
								theWin_Info->hard_devices[val].dev_max_width= wcentimeters;

								sprintf( hdim_spec, "%lg", hcentimeters);
								STRINGCHECK( hdim_spec, MAXCHBUF);
								xtb_ti_set( real_info->height_dimspec, hdim_spec, (xtb_data) 0);
								theWin_Info->hard_devices[val].dev_max_height= hcentimeters;

								xtb_bt_set(AF(PSSA_F).win, preserve_screen_aspect, NULL);

								if( val== PS_DEVICE ){
									if( ps_mpage ){
										if( !sp && showpage ){
											if( theWin_Info->print_orientation ){
											  /* landscape */
												theWin_Info->ps_xpos= 2;
												theWin_Info->ps_ypos= 1;
											}
											else{
											  /* portrait */
												theWin_Info->ps_xpos= 1;
												theWin_Info->ps_ypos= 0;
											}
										}
										else if( sp ){
										  /* just "finished" a page - reset */
											if( theWin_Info->print_orientation ){
											  /* landscape */
												theWin_Info->ps_xpos= 0;
												theWin_Info->ps_ypos= 1;
											}
											else{
											  /* portrait */
												theWin_Info->ps_xpos= 1;
												theWin_Info->ps_ypos= 2;
											}
											showpage= 0;
										}
										xtb_br_set( AF(PSPOSX_F).win, theWin_Info->ps_xpos);
										xtb_br_set( AF(PSPOSY_F).win, theWin_Info->ps_ypos);
									}
									{ char buf[128];
										if( dev_p== 3 ){
											ps_page_nr= pspn;
										}
										sprintf( buf, "PS Page %d", ps_page_nr );
										xtb_bt_set_text( AF(PAGE_XGPS_F).win, 0, buf, (xtb_data) &ps_page_nr );
									}
								}

								xtb_bt_set(AF(PSSP_F).win, showpage, NULL);
								xtb_bt_set( AF(PSEPS_F).win, psEPS, NULL );
								xtb_bt_set( AF(PSDSC_F).win, psDSC, NULL );
								xtb_bt_set( AF(PSSetPage_F).win, psSetPage, NULL );
								xtb_bt_set(AF(XGDBIN_F).win, theWin_Info->DumpBinary, NULL);
								xtb_bt_set(AF(XGDASC_F).win, theWin_Info->DumpAsAscanf, NULL);
								xtb_bt_set(AF(XGDHEX_F).win, DumpDHex, NULL);
								xtb_bt_set(AF(XGDPEN_F).win, DumpPens, NULL);
								xtb_bt_set(AF(XGDPFRESH_F).win, DProcFresh, NULL);
								
								XGDump_PrintPars= xdpp;

								  /* successfully saved/dumped data/printout. So store the
								   \ current configuration
								   */
								if( !real_info->errnr ){
								  int dn= device_nr;
								  extern char PrintTime[];
									HO_dev_fun( AF(ODEVROW_F).win, val, -1, HO_d_info );
									device_nr= dn;
									if( !gs_batching ){
										theWin_Info->printed= 1;
										snprintf( AF(DONE_F).description, DONE_LEN,
											"Indicates whether the current window's last print may be uptodate\n Last %s @ %s\n %s",
											theWin_Info->hard_devices[val].dev_name, PrintTime,
											ps_comment
										);
										ps_comment[0]= '\0';
										strncat( AF(DONE_F).description, f_o_d, 510- strlen(AF(DONE_F).description) );
										if( do_iconify ){
										  extern char event_read_buf[64];
										  extern double *ascanf_ReadBufVal;
										  extern int handle_event_times;
											strcpy( event_read_buf, "0" );
											*ascanf_ReadBufVal= 0;
											handle_event_times= 0;
											XGIconify( theWin_Info );
										}
									}
next_loop:;
									if( lfod && loop ){
									  /* (setNumber points to first unused set...) */
										loop+= 1;
										  /* 20030306: this is for multipage printing: */
										sp= showpage;
										XSync( disp, 0 );
										if( dumped ){
											fileNum+= 1;
										}
									}
									else{
										loop= 0;
									}
								}
								else{
									theWin_Info->printed= 0;
									if( CheckMask(xtb_modifier_state, Mod1Mask) ){
										HO_Dialog.mapped= -1;
									}
									loop= 0;
								}
							}

							if( lfod ){
								memcpy( theWin_Info->draw_set, ldraw_set, setNumber* sizeof(short) );
								xfree( f_o_d );
								f_o_d= lfod;
								XG_Stripped= ostrip;
							}
							if( free_fod ){
								xfree( f_o_d );
							}
							rtn = XTB_HANDLED;
							if( HO_wmtitle ){
								XStoreName(disp, HO_Dialog.win, HO_wmtitle);
								XFree( HO_wmtitle );
							}
							if( viewcommand ){
							  char *command= concat( viewcommand, " < ", f_o_d, NULL );
								if( command ){
									system( command );
									xfree( command );
								}
							}
						} else {
						  /* Bad axis size */
							real_info->errnr= -2;
							do_error("Bad axis font size\n");
							rtn = XTB_HANDLED;
						}
					}
					else{
						  /* Bad label size */
							sprintf( lafam, "Bad legend (%s) or label (%s) size\n",
								lesizstr, lasizstr
							);
							STRINGCHECK( lafam, MAXCHBUF);
							real_info->errnr= -3;
							do_error( lafam );
							rtn = XTB_HANDLED;
					}
				} else {
					/* Bad title size */
					real_info->errnr= -4;
					do_error("Bad title font size\n");
					rtn = XTB_HANDLED;
				}
			}
			else {
			  char buf[4*MAXCHBUF];
				/* Bad max dimension */
				real_info->errnr= -5;
				sprintf( buf, "Bad maximum dimension(s)\n\"%s\",\"%s\" = %s,%s\n",
					hdim_spec, wdim_spec,
					d2str(hcentimeters, "%g", NULL), d2str(wcentimeters, "%g", NULL)
				);
				STRINGCHECK( buf, sizeof(buf));
				do_error(buf);
				rtn = XTB_HANDLED;
			}
		} else {
			/* Bad device spec */
			real_info->errnr= -6;
			do_error("Must specify `Append File', 'New File' or `To Device'\n");
			rtn = XTB_HANDLED;
		}
    } else {
	  /* Bad value spec */
		real_info->errnr= -7;
		do_error("Must specify an output device\n");
		rtn = XTB_HANDLED;
    }
	if( HO_oktext ){
		xtb_bt_set_text(win, gs_batching, HO_oktext, (xtb_data) 0 );
		xfree( HO_oktext );
	}
	else{
		xtb_bt_set(win, gs_batching, (xtb_data) 0);
	}
	active= False;
    return rtn;
}

extern void CloseHO_Dialog(xtb_frame *dial), _CloseHO_Dialog(xtb_frame *dial, Boolean delete);

/*ARGSUSED*/
static xtb_hret can_fun(win, val, info)
Window win;			/* Button window     */
int val;			/* Button value      */
xtb_data info;			/* Local button info */
/*
 * This is the handler function for the cancel button.  It
 * turns itself on and off and then exits with a status 
 * which kills the dialog.
 */
{
	if( !theWin_Info ){
		return( XTB_STOP );
	}

    (void) xtb_bt_set(win, 1, (xtb_data) 0);
    (void) xtb_bt_set(win, 0, (xtb_data) 0);
	HO_Dialog.mapped= -1;
	_CloseHO_Dialog( &HO_Dialog, CheckMask( xtb_modifier_state, Mod1Mask) );
    return XTB_HANDLED;
}

static xtb_hret copyright_function(Window win, int val, xtb_data info)
/*
 * This is the handler function for the cancel button.  It
 * turns itself on and off and then exits with a status 
 * which kills the dialog.
 */
{ extern char *XGraph_Compile_Options;
  char *pinfo= NULL;
	if( !theWin_Info ){
		return( XTB_STOP );
	}

    (void) xtb_bt_set(win, 1, (xtb_data) 0);
	_Dump_Arg0_Info( theWin_Info, NULL, &pinfo, False );
	XG_error_box( &theWin_Info, "About this file and XGraph in general:",
		COPYRIGHT, "\n ", xg_id_string_stub(), "\n Compiled with ",
		XGraph_Compile_Options, "\n Current process info:\n ", pinfo,
		NULL );
    (void) xtb_bt_set(win, 0, (xtb_data) 0);
	xfree( pinfo );
    return XTB_HANDLED;
}

static xtb_hret redraw_fun(Window win, int val, xtb_data info)
/* 
 \ This is the handler for the redraw button.
 \ It sets the redraw field of theWin_Info to 1
 \ and calls can_fun() to undo the hardcopy dialog.
 */
{  extern int DrawWindow();
   extern Boolean dialog_redraw;
   extern Boolean set_sn_nr;
   extern int Synchro_State;
   int ss= -1;
	if( !theWin_Info ){
		return( XTB_STOP );
	}

	if( !theWin_Info->redraw ){
		theWin_Info->redraw= 1;
	}
	theWin_Info->halt= 0;
	theWin_Info->draw_count= 0;

	if( win== HO_redrawbtn.win ){
		xtb_bt_set( win, 1, info);

		if( CheckMask(xtb_button_state, Button3Mask) ){
			XMapRaised( disp, theWin_Info->window );
		}
		if( CheckMask(xtb_modifier_state, Mod1Mask) ){
			set_sn_nr= True;
			dialog_redraw= False;
		}
		else{
			set_sn_nr= False;
			dialog_redraw= True;
		}
		if( CheckMask( xtb_modifier_state, ShiftMask) ){
			ss= Synchro_State;
			Synchro_State= 0;
			X_Synchro(theWin_Info);
		}
	}
	else{
		set_sn_nr= False;
		dialog_redraw= True;
	}
	{ extern psUserInfo *PS_STATE(LocalWin *);
		if( theWin_Info->raw_once< 0 ){
			theWin_Info->raw_display= theWin_Info->raw_val;
		}
		theWin_Info->raw_once= 0;
		if( PS_STATE(theWin_Info)->Printing!= PS_PRINTING){
			AdaptWindowSize( theWin_Info, theWin_Info->window, 0, 0 );
		}
		files_and_groups( theWin_Info, NULL, NULL );
	}
	DrawWindow( theWin_Info );
	if( (win== HO_redrawbtn.win) ){
		if( ss!= -1 ){
			Synchro_State= ss;
			X_Synchro(theWin_Info);
		}
		xtb_bt_set( win, 0, info);
	}
	return( XTB_HANDLED );
}

static xtb_hret rewrite_fun( Window win, int val, xtb_data info)
/* 
 \ This is the handler for the rewrite button.
 */
{  extern int DumpFILES();
	if( !theWin_Info ){
		return( XTB_STOP );
	}

	if( CheckMask( xtb_modifier_state, Mod1Mask|ShiftMask) ){
		xtb_bt_set( win, 1, NULL);
		DumpFILES( theWin_Info );
		xtb_bt_set( win, 0, NULL);
	}
	else{
		xtb_error_box( theWin_Info->window, "Sorry, you must hold down the Mod1 and Shift keys to activate this function!\n", "Notice" );
	}
	return( XTB_HANDLED );
}

static int do_settings= 0;

static xtb_hret settings_fun(win, val, info)
Window win;			/* Button window     */
int val;			/* Button value      */
xtb_data info;			/* Local button info */
{
	if( !theWin_Info ){
		return( XTB_STOP );
	}

	do_settings= 1;
	HO_Dialog.mapped= -1;
	CloseHO_Dialog( &HO_Dialog);
	return( XTB_HANDLED );
}

extern int use_errors, triangleFlag, error_regionFlag;

char *fodstrs[] = { "To Device", "Append File", "New File", "PS Preview" };
int fod_num= sizeof(fodstrs)/sizeof(char*);

xtb_hret HO_dev_fun(Window win, int previous, int current, xtb_data info)
/* Button row window */
/* Previous button   */
/* Current button    */
/* User data         */
/*
 * This routine swaps the device specific information
 * in the dialog based on what device is selected.  The
 * information passed is the information for the whole
 * dialog.
 */
{
    struct ho_d_info *data = (struct ho_d_info *) info;
    char text[MAXCHBUF];

	if( !theWin_Info ){
		return( XTB_STOP );
	}

	if( !data ){
		fprintf( StdErr, "HO_dev_fun(%ld,%d,%d) called with NULL data\n",
			win, previous, current
		);
		return( XTB_NOTDEF );
	}
	{ int b= xtb_br_get(data->fod);
		if( b>= 0 ){
			fod_spot= b;
		}
		else{
			fprintf( StdErr, "HO_dev_fun(%ld,%d,%d) can't read \"Output to\"\n",
				win, previous, current
			);
			return( XTB_NOTDEF );
		}
	}
	if( current!= previous ){
		device_nr= current;
		theWin_Info->printed= 0;
		if( (previous >= 0) && (previous < hard_count)) {
		  /* Save previous info */
			xtb_ti_get(data->fodspec, text, (xtb_data *) 0);
			if (fod_spot == 1 || fod_spot== 2 ) {
				stralloccpy( &theWin_Info->hard_devices[previous].dev_file, text, MFNAME-1);
			} else if (fod_spot == 0) {
				stralloccpy( &theWin_Info->hard_devices[previous].dev_printer, text, MFNAME-1);
			}
			xtb_ti_get(data->height_dimspec, text, (xtb_data *) 0);
			if (sscanf(text, "%lf", &theWin_Info->hard_devices[previous].dev_max_height) != 1) {
				do_error("Warning: can't read maximum height\n");
			}
			if( xtb_enabled( data->width_dimspec ) ){
				xtb_ti_get(data->width_dimspec, text, (xtb_data *) 0);
				if( sscanf(text, "%lf", &theWin_Info->hard_devices[previous].dev_max_width) != 1) {
					do_error("Warning: can't read maximum width\n");
				}
			}
			if( xtb_enabled( data->tf_family ) ){
				xtb_ti_get(data->tf_family, text, (xtb_data *) 0);
				strncpy(theWin_Info->hard_devices[previous].dev_title_font, text, sizeof(hard_devices[previous].dev_title_font)-1);
			}
			if( xtb_enabled( data->tf_size ) ){
				xtb_ti_get(data->tf_size, text, (xtb_data *) 0);
				if (sscanf(text, "%lf", &theWin_Info->hard_devices[previous].dev_title_size) != 1) {
					do_error("Warning: can't read title font size\n");
				}
			}

			if( xtb_enabled( data->lef_family ) ){
				xtb_ti_get(data->lef_family, text, (xtb_data *) 0);
				strncpy(theWin_Info->hard_devices[previous].dev_legend_font, text, sizeof(hard_devices[previous].dev_legend_font)-1);
			}
			if( xtb_enabled( data->lef_size ) ){
				xtb_ti_get(data->lef_size, text, (xtb_data *) 0);
				if (sscanf(text, "%lf", &theWin_Info->hard_devices[previous].dev_legend_size) != 1) {
					do_error("Warning: can't read legend font size\n");
				}
			}

			if( xtb_enabled( data->laf_family ) ){
				xtb_ti_get(data->laf_family, text, (xtb_data *) 0);
				strncpy(theWin_Info->hard_devices[previous].dev_label_font, text, sizeof(hard_devices[previous].dev_label_font)-1);
			}
			if( xtb_enabled( data->laf_size ) ){
				xtb_ti_get(data->laf_size, text, (xtb_data *) 0);
				if (sscanf(text, "%lf", &theWin_Info->hard_devices[previous].dev_label_size) != 1) {
					do_error("Warning: can't read label font size\n");
				}
			}

			if( xtb_enabled( data->af_family ) ){
				xtb_ti_get(data->af_family, text, (xtb_data *) 0);
				strncpy(theWin_Info->hard_devices[previous].dev_axis_font, text, sizeof(hard_devices[previous].dev_axis_font)-1);
			}
			if( xtb_enabled( data->af_size ) ){
				xtb_ti_get(data->af_size, text, (xtb_data *) 0);
				if (sscanf(text, "%lf", &theWin_Info->hard_devices[previous].dev_axis_size) != 1) {
					do_error("Warning: can't read axis font size\n");
				}
			}

		}
		/* Insert current info */
		if( (current >= 0) && (current < hard_count)) {

			theWin_Info->current_device= current;

			xfree(Odevice);
			Odevice= XGstrdup( theWin_Info->hard_devices[current].dev_name );

			if (fod_spot == 1 || fod_spot== 2 ) {
				xtb_ti_set(data->fodspec, theWin_Info->hard_devices[current].dev_file, (xtb_data) 0);
			} else if (fod_spot == 0) {
				xtb_ti_set(data->fodspec, theWin_Info->hard_devices[current].dev_printer,
					   (xtb_data) 0);
			} else {
				xtb_ti_set(data->fodspec, "", (xtb_data) 0);
			}
			(void) sprintf(text, "%lg", theWin_Info->hard_devices[current].dev_max_height);
			STRINGCHECK( text, sizeof(text));
			xtb_ti_set(data->height_dimspec, text, (xtb_data) 0);
			(void) sprintf(text, "%lg", theWin_Info->hard_devices[current].dev_max_width);
			STRINGCHECK( text, sizeof(text));
			xtb_ti_set(data->width_dimspec, text, (xtb_data) 0);

			if( current!= XGRAPH_DEVICE ){
				xtb_ti_set(data->tf_family, theWin_Info->hard_devices[current].dev_title_font,
					   (xtb_data) 0);
				(void) sprintf(text, "%lg", theWin_Info->hard_devices[current].dev_title_size);
				STRINGCHECK( text, sizeof(text));
				xtb_ti_set(data->tf_size, text, (xtb_data) 0);
				xtb_enable( data->tf_size );

				xtb_ti_set(data->lef_family, theWin_Info->hard_devices[current].dev_legend_font,
					   (xtb_data) 0);
				(void) sprintf(text, "%lg", theWin_Info->hard_devices[current].dev_legend_size);
				STRINGCHECK( text, sizeof(text));
				xtb_ti_set(data->lef_size, text, (xtb_data) 0);
				xtb_enable( data->lef_size );

				xtb_ti_set(data->laf_family, theWin_Info->hard_devices[current].dev_label_font,
					   (xtb_data) 0);
				(void) sprintf(text, "%lg", theWin_Info->hard_devices[current].dev_label_size);
				STRINGCHECK( text, sizeof(text));
				xtb_ti_set(data->laf_size, text, (xtb_data) 0);
				xtb_enable( data->laf_size );

				xtb_ti_set(data->af_family, theWin_Info->hard_devices[current].dev_axis_font,
					   (xtb_data) 0);
				(void) sprintf(text, "%lg", theWin_Info->hard_devices[current].dev_axis_size);
				STRINGCHECK( text, sizeof(text));
				xtb_ti_set(data->af_size, text, (xtb_data) 0);
				xtb_enable( data->af_size );
			}
			else{
				xtb_ti_set(data->tf_family, titleFont.name, (xtb_data) 0);
				xtb_disable( data->tf_size );
				xtb_ti_set(data->lef_family, legendFont.name, (xtb_data) 0);
				xtb_disable( data->lef_size );
				xtb_ti_set(data->laf_family, labelFont.name, (xtb_data) 0);
				xtb_disable( data->laf_size );
				xtb_ti_set(data->af_family, axisFont.name, (xtb_data) 0);
				xtb_disable( data->af_size );
			}

			switch( current ){
				case PS_DEVICE:
					xtb_bt_set_text( AF(PSTRS_F).win, ps_transparent, " PS trsp ", (xtb_data) &ps_transparent );
					xtb_bt_set_text( AF(PSRGB_F).win, ps_coloured, "PS RGB", (xtb_data) &ps_coloured);
					sprintf( text, "PS Page %d", ps_page_nr );
					xtb_bt_set_text( AF(PAGE_XGPS_F).win, 0, text, (xtb_data) &ps_page_nr );
					break;
				case XGRAPH_DEVICE:
					xtb_bt_set_text( AF(PSTRS_F).win, DumpKeyParams, "KeyEVAL", (xtb_data) &DumpKeyParams );
					xtb_bt_set_text( AF(PSRGB_F).win, XGStoreColours, "XG RGB", (xtb_data) &XGStoreColours );
					xtb_bt_set_text( AF(PAGE_XGPS_F).win, XGDump_PrintPars, "PS Settings", (xtb_data) &XGDump_PrintPars );
					sprintf( AF(PSRGB_F).description,
						" [PS RGB]: Output PostScript using colours (RGB)\n"
						" [XG RGB]: Store colournames in XGraph dump\n"
						"           (Hold Mod1 to toggle TrueGray display [%3s])\n"
						"<Mod1>-r\n", (TrueGray)? "ON" : "OFF"
					);
					break;
				default:
					xtb_bt_set_text( AF(PAGE_XGPS_F).win, 0, "<No pages>", (xtb_data) &HO_nothing );
					break;
			}
		}

	}
    return XTB_HANDLED;
}

static xtb_hret fd_fun( Window win, int old, int new, xtb_data info)
/*
 * This routine swaps the default file or device names
 * based on the state of the file or device buttons.
 * The information passed is the information for the whole
 * dialog.
 */
{
    struct ho_d_info *data = (struct ho_d_info *) info;
    char text[MAXCHBUF];
    int which_one;
    
	if( !theWin_Info ){
		return( XTB_STOP );
	}

	if( !data ){
		fprintf( StdErr, "HO::fd_fun(%ld,%d,%d) called with NULL data\n",
			win, old, new
		);
		return( XTB_NOTDEF );
	}
	{ int b= xtb_br_get(data->choices);
		if( b>= 0 ){
			which_one= b;
		}
		else{
			fprintf( StdErr, "HO::fd_fun(%ld,%d,%d) can't read \"Output device\"\n",
				win, old, new
			);
			return( XTB_NOTDEF );
		}
	}
	if( new== -1 ){
		new= which_one;
		xtb_br_set( data->choices, new );
	}
    if ((which_one >= 0) && (which_one < hard_count)) {
		if (old == 0) {
			/* Save into device space */
			xtb_ti_get(data->fodspec, text, (xtb_data *) 0);
			stralloccpy( &theWin_Info->hard_devices[which_one].dev_printer, text, MFNAME-1);
		} else if (old == 1 || old== 2) {
			/* Save into file space */
			xtb_ti_get(data->fodspec, text, (xtb_data *) 0);
			which_one = xtb_br_get(data->choices);
			stralloccpy( &theWin_Info->hard_devices[which_one].dev_file, text, MFNAME-1);
		}
		else if( old== 3 ){
			which_one = xtb_br_get(data->choices);
/* 			strncpy( theWin_Info->hard_devices[which_one].dev_spec, "%s", MFNAME-1);	*/
			stralloccpy( &theWin_Info->hard_devices[which_one].dev_spec, "%s", MFNAME-1);
			stralloccpy( &theWin_Info->hard_devices[which_one].dev_printer, "xgraph.pspreview", MFNAME-1);
		}
		if (new == 0 ) {
			/* Restore into device */
			xtb_ti_set(data->fodspec, theWin_Info->hard_devices[which_one].dev_printer,
				   (xtb_data *) 0);
		} else if( new == 3 ) {
			/* Restore into device */
/* 			strncpy( theWin_Info->hard_devices[which_one].dev_spec, "%s", MFNAME-1);	*/
			stralloccpy( &theWin_Info->hard_devices[which_one].dev_spec, "%s", MFNAME-1);
			stralloccpy( &theWin_Info->hard_devices[which_one].dev_printer, "xgraph.pspreview", MFNAME-1);
			xtb_ti_set(data->fodspec, theWin_Info->hard_devices[which_one].dev_printer, (xtb_data *) 0);
		} else if (new == 1 || new== 2 ) {
			xtb_ti_set(data->fodspec, theWin_Info->hard_devices[which_one].dev_file,
				   (xtb_data *) 0);
		}
		xfree(Odisp);
		Odisp= XGstrdup( fodstrs[new] );
    }
    return XTB_HANDLED;
}

int gsTextWidth_Batch()
{ xtb_hret r;
  TextRelated textrel= theWin_Info->textrel;
  int outto= xtb_br_get(HO_d_info->choices), pD= psDSC, psn= PrintSetsNR, sp= showpage, psmp= ps_mpage;
  int twbN0= 0, twbN1= 0, ppn= ps_page_nr;
	xtb_modifier_state= 0;
	theWin_Info->textrel.gs_batch= True;
	psDSC= False;
	PrintSetsNR= -1;
	showpage= 1;
	ps_mpage= 0;
	xtb_bt_set( AF(PSDSC_F).win, psDSC, NULL );
	xtb_br_set( AF(ODEVROW_F).win, PS_DEVICE );
	HO_dev_fun(HO_d_info->choices, outto, PS_DEVICE, HO_d_info );

	do{
		twbN1= twbN0;
		twbN0= last_gsTextWidthBatch;
		r= HO_ok_fun( HO_okbtn.win, PS_DEVICE, ok_info);
	} while( auto_gsTextWidth && last_gsTextWidthBatch && r!= XTB_STOP && !(twbN1== twbN0 && twbN0== last_gsTextWidthBatch) );

	xtb_br_set( AF(ODEVROW_F).win, outto );
	HO_dev_fun(HO_d_info->choices, PS_DEVICE, outto, HO_d_info );
	psDSC= pD;
	PrintSetsNR= psn;
	xtb_bt_set( AF(PSDSC_F).win, psDSC, NULL );
	theWin_Info->textrel= textrel;
	ps_page_nr= ppn;
	showpage= sp;
	ps_mpage= psmp;
	return(r);
}

char XG_PS_NUp_buf[256] = "";

static xtb_hret HO_sdds_sl_fun(Window win, int bval, xtb_data info)
{  int i, *field= NULL;
   int *Info= ((int*) info);
   int do_redraw= -1;

	if( !theWin_Info ){
		return( XTB_STOP );
	}

	if( info== &HO_nothing ){
		Boing(5);
		return( XTB_HANDLED );
	}

	bval= !bval;

	if( Info== &use_X11Font_length ){
		field= &use_X11Font_length;
		if( CheckMask(xtb_modifier_state, Mod1Mask) ){
		  char ebuf[64], message[512], *nbuf;
			sprintf( ebuf, "%s", d2str(Font_Width_Estimator, NULL, NULL) );
			sprintf( message, " Enter fontwidth estimator EST used to calculate approximate (proportional) PS fontwidth;\n"
				" approx. font width= pointsize* EST * VirtualDPI/ DPI= pointsize * %g * %g/ %g= pointsize * %g\n",
					Font_Width_Estimator, VDPI, POINTS_PER_INCH, Font_Width_Estimator* INCHES_PER_POINT* VDPI
			);
			STRINGCHECK( message, sizeof(message)/sizeof(char) );
			if( (nbuf= xtb_input_dialog( HO_Dialog.win, ebuf, 16, 64, message, "Enter a number", False, "", NULL, "", NULL, NULL, NULL )) ){
			  double x;
			  int n=1;
				if( fascanf( &n, nbuf, &x, NULL, NULL, NULL, NULL)>=0 && n== 1 ){
					if( Font_Width_Estimator!= x ){
						set_HO_printit_win();
					}
					if( x!= Font_Width_Estimator ){
						theWin_Info->axis_stuff.__XLabelLength= 0;
						theWin_Info->axis_stuff.__YLabelLength= 0;
						theWin_Info->axis_stuff.XLabelLength= 0;
						theWin_Info->axis_stuff.YLabelLength= 0;
					}
					Font_Width_Estimator= x;
					sprintf( AF(UXLF_F).description,
						"Determine the widths of strings on the metrics of the X-font used (scales the on-screen sizes)\n"
						" <Mod1>click to change fontwidth estimator EST used to calculate approximate (proportional) PS fontwidth:\n"
						" afw= pointsize* EST * VirtualDPI/ DPI= pointsize * %g * %g/ %g= pointsize * %g\n",
							Font_Width_Estimator, VDPI, POINTS_PER_INCH, Font_Width_Estimator* INCHES_PER_POINT* VDPI
					);
				}
				else{
					xtb_error_box( theWin_Info->window, nbuf, "Error: can't parse floating point number" );
				}
				if( nbuf!= ebuf ){
					xfree( nbuf );
				}
			}
			  /* Don't change the selected field in this case ;-)	*/
			bval= *field;
		}
		do_redraw= 1;
	}
	else if( Info== &use_gsTextWidth ){
		if( use_gsTextWidth && CheckMask(xtb_modifier_state, Mod1Mask) ){
		  int resp;
			if( auto_gsTextWidth ){
				resp= xtb_error_box( theWin_Info->window,
					"Automatic batch determination of PostScript string widths using ghostscript\n"
					" before generating PostScript output is ACTIVated. Should it be turned OFF?\n",
					"Question"
				);
				if( resp> 0 ){
					auto_gsTextWidth= False;
				}
			}
			else{
				resp= xtb_error_box( theWin_Info->window,
					"Automatic batch determination of PostScript string widths using ghostscript\n"
					" before generating PostScript output is OFF. Should it be turned ON?\n",
					"Question"
				);
				if( resp> 0 ){
					auto_gsTextWidth= True;
				}
			}
			return(XTB_HANDLED);
		}
		else{
			field= Info;
			do_redraw= 0;
			theWin_Info->printed= False;
		}
	}
	else if( Info== &ps_page_nr ){
		xtb_bt_set_text( AF(PAGE_XGPS_F).win, 0, "PS Page 1", (xtb_data) &ps_page_nr );
		ps_page_nr= 1;
		return( XTB_HANDLED );
	}
	else if( Info== &showpage || Info== &XGDump_PrintPars || Info== &XGDump_AllWindows ||
		Info== &psEPS || Info== &psDSC || Info== &psSetPage || Info== &Use_HO_Previous_TC
	){
		field= Info;
		if( Info== &showpage && bval && !showpage ){
			psEPS= False;
			xtb_bt_set( AF(PSEPS_F).win, psEPS, NULL );
		}
		else if( Info== &psEPS && bval && !psEPS ){
			showpage= False;
			xtb_bt_set( AF(PSSP_F).win, showpage, NULL );
		}
		else if( Info== &XGDump_AllWindows && CheckMask(xtb_modifier_state, Mod1Mask) ){
		  char *nbuf;

			sprintf( XG_PS_NUp_buf, "%d x %d", XG_NUp_X, XG_NUp_Y );
			if( XG_NUp_aspect ){
				strcat( XG_PS_NUp_buf, " 1:1" );
			}
			if( XG_NUp_scale!= 1 ){
				sprintf( XG_PS_NUp_buf, "%s *%g", XG_PS_NUp_buf, XG_NUp_scale );
			}
			if( !XG_NUp_scale_W ){
				sprintf( XG_PS_NUp_buf, "%s *W", XG_PS_NUp_buf );
			}
			if( !XG_NUp_scale_H ){
				sprintf( XG_PS_NUp_buf, "%s *H", XG_PS_NUp_buf );
			}
			if( (nbuf= 
				xtb_input_dialog( HO_Dialog.win, XG_PS_NUp_buf, 16, sizeof(XG_PS_NUp_buf)*sizeof(char), 
					"In PostScript mode, XGraph can print all open windows in a <columns> x <rows>\n"
					" layout, when AllWin is set. Enter the desired numbers here.\n"
					" Add 1:1 to preserve aspect ratios.\n"
					" Add *x to _decrease_ width and height by <x> and _increase_ the PS scale by\n"
					" the same amount (necessary for large desired printing sizes which still are problematic.\n"
					"  (add *H and/or *W to not have to adapt the metric width and/or height [try...!])\n"
					, "Enter columns x rows", False, "", NULL, "", NULL, NULL, NULL )
				)
			){
			  int x, y;
			  char *c= strstr( nbuf, "1:1");
			  char *d= strstr( nbuf, "*" );
			  char *dW= strstr( nbuf, "*W" );
			  char *dH= strstr( nbuf, "*H" );
				if( c ){
					XG_NUp_aspect= True;
					*c= '\0';
				}
				else{
					XG_NUp_aspect= False;
				}
				if( d ){
				  int n= 1;
					fascanf( &n, &d[1], &XG_NUp_scale, NULL, NULL, NULL, NULL);
					if( XG_NUp_scale<= 0 ){
						XG_NUp_scale= 1;
					}
					*d= '\0';
					XG_NUp_scale_W= (dW)? 0 : 1;
					XG_NUp_scale_H= (dH)? 0 : 1;
				}
				else{
					XG_NUp_scale= 1;
				}
				if( sscanf( nbuf, "%d x %d", &x, &y)== 2 ||
					sscanf( nbuf, "%d X %d", &x, &y)== 2
				){
					XG_NUp_X= x;
					XG_NUp_Y= y;
				}
				else{
					xtb_error_box( theWin_Info->window, nbuf, "Error: can't parse 2 decimal numbers" );
				}
				if( c ){
					*c = '1';
				}
				if( d ){
					// restore the asterix:
					*d = '*';
				}
				if( nbuf!= XG_PS_NUp_buf ){
					xfree( nbuf );
				}
			}
			bval= XGDump_AllWindows;
		}
		else if( Info== &psSetPage && CheckMask(xtb_modifier_state, Mod1Mask) ){
		  char ebuf[256], *nbuf;

			sprintf( ebuf, "%g x %g", psSetPage_width, psSetPage_height );
			if( (nbuf= 
				xtb_input_dialog( HO_Dialog.win, ebuf, 16, sizeof(ebuf)*sizeof(char), 
					"Specify the desired paper's width x height, in cm., for portrait orientation.\n"
					" Specify -1 to scale to the corresponding graph's dimension.\n"
					, "Enter width x height", False, "", NULL, "", NULL, NULL, NULL )
				)
			){
			  double x, y;
				if( sscanf( nbuf, "%lf x %lf", &x, &y)== 2 ||
					sscanf( nbuf, "%lf X %lf", &x, &y)== 2
				){
					psSetPage_width= x;
					psSetPage_height= y;
				}
				else{
					xtb_error_box( theWin_Info->window, nbuf, "Error: can't parse 2 decimal numbers" );
				}
				if( nbuf!= ebuf ){
					xfree( nbuf );
				}
			}
			bval= psSetPage;
		}
		do_redraw= 0;
		theWin_Info->printed= False;
	}
	else if( Info== &ps_coloured || Info== &XGStoreColours ){
		field= Info;
		do_redraw= 0;
		if( CheckExclMask( xtb_modifier_state, Mod1Mask)){
			TrueGray= !TrueGray;
			ReallocColours(True);
			bval= !bval;
			do_redraw= 1;
			xtb_error_box( win, (TrueGray)? "ON" : "OFF", "Toggled TrueGray display" );
		}
		sprintf( AF(PSRGB_F).description,
			" [PS RGB]: Output PostScript using colours (RGB)\n"
			" [XG RGB]: Store colournames in XGraph dump\n"
			"           (Hold Mod1 to toggle TrueGray display [%3s])\n"
			"<Mod1>-r\n", (TrueGray)? "ON" : "OFF"
		);
		theWin_Info->printed= False;
	}
	else if( Info== &ps_show_margins ){
		field= Info;
		do_redraw= 0;
	}
	else if( Info== &ps_transparent || Info== &DumpKeyParams ){
		field= Info;
		do_redraw= 0;
		theWin_Info->printed= False;
	}
	else if( Info== &ps_mpage ){
		field= Info;
		do_redraw= 0;
		if( bval ){
			if( theWin_Info->print_orientation ){
			  /* landscape */
				theWin_Info->ps_xpos= 0;
				theWin_Info->ps_ypos= 1;
			}
			else{
			  /* portrait */
				theWin_Info->ps_xpos= 1;
				theWin_Info->ps_ypos= 2;
			}
			showpage= 0;
		}
		else{
			showpage= 1;
		}
		xtb_br_set( AF(PSPOSX_F).win, theWin_Info->ps_xpos);
		xtb_br_set( AF(PSPOSY_F).win, theWin_Info->ps_ypos);
		xtb_bt_set( AF(PSSP_F).win, showpage, NULL);
		xtb_bt_set( AF(PSEPS_F).win, psEPS, NULL );
		xtb_bt_set( AF(PSDSC_F).win, psDSC, NULL );
		xtb_bt_set( AF(PSSetPage_F).win, psSetPage, NULL );
	}
	else if( Info== &set_PS_PrintComment ){
		field= &PS_PrintComment;
		theWin_Info->printed= 0;
		do_redraw= 0;
	}
	else if( Info== &set_Sort_Sheet ){
		field= &Sort_Sheet;
		do_redraw= 0;
	}
	else if( Info== &dump_average_values ){
		field= &(theWin_Info->dump_average_values);
		do_redraw= 0;
	}
	else if( Info== &DumpProcessed ){
		if( bval && !CheckMask(xtb_modifier_state, Mod1Mask) ){
			xtb_error_box( theWin_Info->window,
				"Warning: only the columns with X, Y and Error\n"
				" will contain processed values; the other columns\n"
				" will contain the original (raw) values!\n"
				" In addition, when a *TRANSFORM_Y* process is specified,\n"
				" Error values will be the average of the\n"
				" processed <raw Y - raw Error> and <raw Y + raw Error>,\n"
/* 				" unless when in vector (-vector) mode when the processed error is dumped\n"	*/
				" otherwise, the processed error is dumped\n"
				,"Note"
			);
		}
		field= &(theWin_Info->DumpProcessed);
		do_redraw= 0;
	}
	else if( Info== &splits_disconnect ){
		if( bval && !CheckMask(xtb_modifier_state, Mod1Mask) ){
			xtb_error_box( theWin_Info->window,
				"Warning: setting this causes set splits (*SPLIT* commands) set by\n"
				" <Mod1>-x;<Mod1>-button1 or the split[] function to start a new dataset\n"
				" at the corresponding point the next time the data is read. The only way\n"
				" to rejoin the sets afterwards is to edit the XGraph file by hand!\n"
				,"Note"
			);
		}
		field= Info;
		do_redraw= 0;
	}
	else if( Info== &Init_XG_Dump ){
		if( !bval && !CheckMask(xtb_modifier_state, Mod1Mask) ){
			xtb_error_box( theWin_Info->window,
				"Warning: only set-data and set-headers will be dumped!\n"
				" When DumpProc is not selected, global transformations\n"
				" and processing will be lost!\n"
				" Use this for generating \"include\" files.\n"
				,"Note"
			);
		}
		field= &Init_XG_Dump;
		do_redraw= 0;
	}
	else if( Info== &DumpDHex ){
		field= &DumpDHex;
		do_redraw= 0;
	}
	else if( Info== &DumpAsAscanf ){
		field= &theWin_Info->DumpAsAscanf;
		do_redraw= 0;
	}
	else if( Info== &DProcFresh ){
		field= &DProcFresh;
		do_redraw= 0;
	}
	else if( Info== &DumpPens ){
		field= &DumpPens;
		do_redraw= 0;
	}
	else if( Info== &DumpBinary ){
		field= &(theWin_Info->DumpBinary);
		if( CheckMask( xtb_modifier_state, Mod1Mask ) ){
			bval= *field;
			{ char ebuf[64], *nbuf, desc[256];
			  extern int BinarySize;

				sprintf( ebuf, "%d", BinarySize );
				snprintf( desc, sizeof(desc),
					"Please enter a size for binary floating point output;\n"
					" for doubles, enter %d;\n"
					" for floats, enter %d;\n"
					" for 16bit ints, enter %d;\n"
					" for 8bit ints, enter %d;\n"
					" all other values are ignored!\n",
					sizeof(double), sizeof(float), sizeof(unsigned short), sizeof(unsigned char)
				);
				if( (nbuf= 
					xtb_input_dialog( HO_Dialog.win, ebuf, 16, sizeof(ebuf)*sizeof(char), 
						desc, "Enter a number", False, "", NULL, "", NULL, NULL, NULL )
					)
				){
				  int x= 0;
					sscanf( nbuf, "%d", &x);
					switch( x ){
						case sizeof(double):
						case sizeof(float):
						case sizeof(unsigned short):
						case sizeof(unsigned char):
							BinarySize= x;
							break;
						default:
							xtb_error_box( theWin_Info->window, nbuf, "Error: invalid or unparseable number ignored" );
							break;
					}
					if( nbuf!= ebuf ){
						xfree( nbuf );
					}
				}
			}
		}
		do_redraw= 0;
	}
	else if( Info== &XG_Stripped ){
		field= &XG_Stripped;
		do_redraw= 0;
	}
	else if( Info== &XG_SaveBounds ){
		field= &XG_SaveBounds;
		do_redraw= 0;
	}
	else if( Info== &set_scale_plot_area ){
		field= &scale_plot_area_x;
		do_redraw= 1;
	}
	else if( Info== &scale_plot_area_y ){
		field= &scale_plot_area_y;
		do_redraw= 1;
	}
	else if( Info== &set_winsize ){
		ZoomWindow_PS_Size( theWin_Info, 0, 0, 1 );
		bval= (theWin_Info->dev_info.resized==1);
		field= &set_winsize;
	}
	else if( Info== &set_preserve_screen_aspect ){
		field= &preserve_screen_aspect;
		if( do_redraw< 0 ){
			do_redraw= 0;
		}
		theWin_Info->printed= False;
	}
	else if( Info== &XG_preserve_filetime ){
		field= &XG_preserve_filetime;
		do_redraw= 0;
	}
	if( field || (data_sn_number>= 0 && data_sn_number< setNumber) ){
		xtb_bt_set( win, bval, (xtb_data) 0);
		i= data_sn_number;
		if( !field ){
			do_error( "Illegal button in HO_sdds_sl_fun\nReport to sysop!\n");
			return( XTB_HANDLED );
		}
		if( *field!= bval ){
			if( do_redraw!= theWin_Info->redraw ){
				theWin_Info->redraw= do_redraw;
				xtb_bt_set( HO_redrawbtn.win, theWin_Info->redraw, NULL);
			}
			if( do_redraw ){
				theWin_Info->printed= 0;
			}
			*field= bval;
			if( field== &theWin_Info->DumpBinary ){
				if( theWin_Info->DumpBinary && theWin_Info->DumpAsAscanf ){
					xtb_bt_set(AF(XGDASC_F).win, (theWin_Info->DumpAsAscanf= 0), NULL);
				}
			}
			else if( field== &theWin_Info->DumpAsAscanf ){
				if( theWin_Info->DumpAsAscanf && theWin_Info->DumpBinary ){
					xtb_bt_set( AF(XGDBIN_F).win, (theWin_Info->DumpBinary= 0), NULL);
				}
			}
			set_HO_printit_win();
		}
	}
	else{
		Boing( 5);
		Boing( 5);
	}
	return( XTB_HANDLED );
}


#define BAR_SLACK	10

char *orientstrs[] = { "P", "L" };
char *xposstrs[]= { "L", "C", "R" }, *yposstrs[]= { "B", "C", "T" };

  /* Warning: one such static variable per module that can perform "smart" xtb_init calling	*/
static int xtb_UseColours= False;

extern Atom wm_delete_window;

static void make_HO_dialog( Window win, Window spawned, char *prog, xtb_data cookie,
	xtb_frame *HO_okbtn, xtb_frame *HO_canbtn, xtb_frame *HO_redrawbtn, xtb_frame *frame,
	char *title, char *in_title)
/*
 * This routine constructs a new dialog for asking the user about
 * hardcopy devices.  The dialog and its size is returned in 
 * `frame'.  The window of the `ok' button is returned in `btnwin'.  
 * This can be used to reset some of the button state to reuse the dialog.
 */
{
    xtb_fmt *def, *cntrl, *mindim, *font_area;
    Cursor diag_cursor;
    XColor fg_color, bg_color;
    XSizeHints hints;
    unsigned long wamask;
    XSetWindowAttributes wattr;
    struct ho_d_info *info;
#define MIN_LEGEND_LENGTH	MAX(MFNAME+D_FS,16)
    int i, max_width;
    char **names;
	extern int bdrSize;			/* Width of border         */
	static char done_descr[DONE_LEN], fontest_descr[512];
	static xtb_frame *local_ho_af= ho_af;
	char pc[]= "\\#x7f\\\\";
	char *bt= parse_codes(pc);

	{ XColor zero, grid, norm, bg;
	  extern Pixel gridPixel;
	  extern int ButtonContrast;
	  char lightgreen[256];
	  Pixel lgreenPixel;
		zero.pixel= zeroPixel;
		XQueryColor( disp, cmap, &zero );
		grid.pixel= gridPixel;
		XQueryColor( disp, cmap, &grid );
		norm.pixel= normPixel;
		XQueryColor( disp, cmap, &norm );
		bg.pixel= bgPixel;
		XQueryColor( disp, cmap, &bg );

		if( *AllowGammaCorrection ){
			sprintf( lightgreen, "rgbi:%g/%g/%g",
				192.0/ 255.0, 208.0/ 255.0, 152.0/ 255.0
			);
		}
		else{
			sprintf( lightgreen, "rgb:%04x/%04x/%04x",
				(int) (192.0/ 255.0* 65535),
				(int) (208.0/ 255.0* 65535),
				(int) (152.0/ 255.0* 65535)
			);
		}
		STRINGCHECK( lightgreen, 256 );
		if( xtb_UseColours ){
		  double contrast;
		  /* If there's enough luminance-contrast between any of the following 2 colours, use 'm for
		   \ the dialog.
		   */
			if( GetColor( lightgreen, &lgreenPixel) &&
				(contrast= fabs( xtb_PsychoMetric_Gray(&norm) - xtb_PsychoMetric_Gray(&bg) ))>= ButtonContrast
			){
				xtb_init(disp, screen, lgreenPixel, bgPixel, dialogFont.font, dialog_greekFont.font, True );
				FreeColor( &lgreenPixel, NULL );
			}
			else if( (contrast= fabs( xtb_PsychoMetric_Gray(&zero) - xtb_PsychoMetric_Gray(&grid) ))>= ButtonContrast ){
				xtb_init(disp, screen, zeroPixel, gridPixel, dialogFont.font, dialog_greekFont.font, True );
			}
			else if( (contrast= fabs( xtb_PsychoMetric_Gray(&norm) - xtb_PsychoMetric_Gray(&bg) ))>= ButtonContrast ){
				xtb_init(disp, screen, normPixel, bgPixel, dialogFont.font, dialog_greekFont.font, True );
			}
		}
		else{
			xtb_init(disp, screen, black_pixel, white_pixel, dialogFont.font, dialog_greekFont.font, True );
		}
	}

    wamask = ux11_fill_wattr(&wattr, CWBackPixel, bgPixel,
			     CWBorderPixel, bdrPixel,
/* 			     CWOverrideRedirect, True,	*/
				CWBackingStore, Always,
			     CWSaveUnder, True,
			     CWColormap, cmap, UX11_END);
    XGetNormalHints(disp, spawned, &hints);
    HO_Dialog.win = XCreateWindow(disp, win, hints.x, hints.y, hints.width, hints.height, bdrSize,
			    depth, InputOutput, vis,
			    wamask, &wattr);
	XSetWMProtocols( disp, HO_Dialog.win, &wm_delete_window, 1 );
    frame->win = HO_Dialog.win;
    frame->width = frame->height = frame->x_loc = frame->y_loc = 0;
	frame->mapped= 0;
	frame->description= strdup( "This is the Hardcopy Dialog's main window" );
    if( !title)
	    XStoreName(disp, HO_Dialog.win, "Hardcopy Dialog");
	else
	    XStoreName(disp, HO_Dialog.win, title);
/*     XSetTransientForHint(disp, spawned, HO_Dialog.win);	*/
    info = (struct ho_d_info *) calloc( 1, sizeof(struct ho_d_info));
	HO_d_info= info;
    info->prog = prog;
    info->cookie = cookie;
	xtb_register( frame, frame->win, NULL, NULL );

    /* Make all frames */
    xtb_to_new(HO_Dialog.win, in_title, XTB_TOP_LEFT, &dialogFont.font, &dialog_greekFont.font, aF(TITLE_F));
    xtb_to_new(HO_Dialog.win, "Output device:", XTB_TOP_LEFT, &dialogFont.font, &dialog_greekFont.font, aF(ODEVLBL_F));
    device_nr = -1;
    names = (char **) calloc( 1, (unsigned) (sizeof(char *) * hard_count));
    for (i = 0;  i < hard_count;  i++) {
		names[i] = theWin_Info->hard_devices[i].dev_name;
		if (strcmp(Odevice, theWin_Info->hard_devices[i].dev_name) == 0) {
			device_nr = i;
		}
    }

    xtb_br_new(HO_Dialog.win, hard_count, names, device_nr,
	       HO_dev_fun, (xtb_data) info, aF(ODEVROW_F));
    info->choices = AF(ODEVROW_F).win;
	theWin_Info->current_device= device_nr;
	xtb_describe( AF(ODEVROW_F).framelist[PS_DEVICE],
		"Dump the graphwindow in PostScript(tm) format, heeding the formatting parameters set below\n"
		" The resulting file is directly printable, but can also be read by Adobe Illustrator(tm)\n"
		" for further editing.\n"
		"<Shift><Mod1>-P\n"
	);
	xtb_describe( AF(ODEVROW_F).framelist[SPREADSHEET_DEVICE],
		"Dump data in format readable by\ngeneric spreadsheet programmes\nLog/Pow axis-transformations are not done,\n\
*TRANSFORM_?* and *DATA_PROCESS* expressions\nare evaluated (errorbars are problematic)\n"
	);
	xtb_describe( AF(ODEVROW_F).framelist[CRICKET_DEVICE],
"Dump data in format readable by Mac (Tm) CricketGraph (Tm)\n\
Log/Pow axis-transformations are not done,\n\
Polar data will be dumped as shown!\n\
*TRANSFORM_?* and *DATA_PROCESS* expressions\nare evaluated (errorbars are problematic)\n"
	);
	xtb_describe( AF(ODEVROW_F).framelist[XGRAPH_DEVICE],
		"Dump data in xgraph's dataformat.\n*TRANFORM_?* expressions are exported;\n"
		" *DATA_???* expressions are exported in deactivated fashion.\n"
		" Some command-line options are not exported!\n"
		"<Shift><Mod1>-X\n"
	);
	xtb_describe( AF(ODEVROW_F).framelist[COMMAND_DEVICE],
"Dump the commandline that should give rise to this graph.\n"
"This is the same command as executed when \"restarting\" a graph (with ^N)\n"
	);
    xtb_to_new(HO_Dialog.win, "Output to:", XTB_TOP_LEFT, &dialogFont.font, &dialog_greekFont.font, aF(DISPLBL_F));
    fod_spot = -1;
    for (i = 0;  i < fod_num;  i++) {
		if (strcmp(Odisp, fodstrs[i]) == 0) {
			fod_spot = i;
		}
    }
    xtb_br_new(HO_Dialog.win, sizeof(fodstrs)/sizeof(char*), fodstrs, fod_spot,
	       fd_fun, (xtb_data) info, aF(DISPROW_F)
	);
    info->fod = AF(DISPROW_F).win;
	xtb_describe( AF(DISPROW_F).framelist[0],
		"Dump to device: specify command\n"
		"<Shift><Mod1>-T\n"
	);
	xtb_describe( AF(DISPROW_F).framelist[1],
		"Append to specified file\n"
		"<Shift><Mod1>-A\n"
	);
	xtb_describe( AF(DISPROW_F).framelist[2],
		"(Re)Create specified file\n"
		"<Shift><Mod1>-N\n"
	);
	xtb_describe( AF(DISPROW_F).framelist[3],
		"Preview command: empy entry gives a default handler.\n"
		"<Shift><Mod1>-V\n"
	);
	{ char buf[1024], *c= getcwd( buf, 1024);
		if( !c ){
			c= serror();
		}
		xtb_ti2_new(HO_Dialog.win, c, D_INP, HO_df_fun, bt, FSfun, &chdir, aF(DIR_F));
	}
	xtb_bt_new( HO_Dialog.win, "Presrv Times", HO_sdds_sl_fun, (xtb_data) &XG_preserve_filetime, aF(PRESTIME_F) );
	xtb_describe( aF(PRESTIME_F),
		"When set, an attempt will be made to preserve the access and modification times on\n"
		" the target file. This will of course work only when dumping to a file.\n"
	);

    xtb_to_new(HO_Dialog.win, "Work Dir & File/Device/Command", XTB_TOP_LEFT, &dialogFont.font, &dialog_greekFont.font, aF(FDLBL_F));
    xtb_ti2_new(HO_Dialog.win, OfileDev, D_INP, df_fun, bt, FSfun, (xtb_data) &OfileDev, aF(FDINP_F));
    info->fodspec = AF(FDINP_F).win;
	xtb_describe( aF(FDINP_F), "Enter pipe or filename here\nAppend '.gz' to filename to compress it\nwith gzip(1).\n"
		" or '.bg2' to compress with bzip2\n"
		"Spaces are not allowed in filenames\n"
		"The last entered non-empty filename replaces a possible\nfilename specified with -pf upon export\n"
	);

/*     xtb_to_new(HO_Dialog.win, "Optional Parameters", XTB_TOP_LEFT, titleFont.font, title_greekFont.font, aF(OPTLBL_F));	*/

    xtb_to_new(HO_Dialog.win, "Maximum Dimensions (WxH cm):", XTB_TOP_LEFT, &dialogFont.font, &dialog_greekFont.font, aF(MDIMLBL_F));
    xtb_ti_new(HO_Dialog.win, "", D_DSP, MAXCHBUF, dfn_fun, (xtb_data) &set_width, aF(MWDIMI_F));
    info->width_dimspec = AF(MWDIMI_F).win;
	xtb_describe( aF(MWDIMI_F), "Width of the (PS) printout in centimeters (on standing paper)\n=> height if width<height\n(i.e. landscape)\n");
    xtb_ti_new(HO_Dialog.win, "", D_DSP, MAXCHBUF, dfn_fun, (xtb_data) &set_height, aF(MHDIMI_F));
    info->height_dimspec = AF(MHDIMI_F).win;
	xtb_describe( aF(MHDIMI_F), "Height of the (PS) printout in centimeters (on standing paper)\n=> width if height>width\n(i.e. landscape)\n");

	xtb_bt_new( HO_Dialog.win, "Size", HO_sdds_sl_fun, (xtb_data) &set_winsize, aF(XWR_F) );
	xtb_bt_set(AF(XWR_F).win, (theWin_Info->dev_info.resized==1), NULL);
	xtb_describe( aF(XWR_F), "Resize window to specified dimensions\n");
	xtb_bt_new( HO_Dialog.win, "Aspect", HO_sdds_sl_fun, (xtb_data) &set_preserve_screen_aspect, aF(PSSA_F) );
	xtb_bt_set(AF(PSSA_F).win, preserve_screen_aspect, NULL);
	xtb_describe( aF(PSSA_F), "Preserve the window aspect ratio; specify only height\n");
	xtb_bt_new( HO_Dialog.win, "AllWin", HO_sdds_sl_fun, (xtb_data) &XGDump_AllWindows, aF(XGAWD_F) );
	xtb_bt_set(AF(XGAWD_F).win, XGDump_AllWindows, NULL);
	xtb_describe( aF(XGAWD_F),
		"For XGraph dumps, save information to restore all currently visible windows\n"
		" For PostScript, print all windows in the specified <colums> x <rows> format\n"
		"        (Mod1-click to specify the columns and rows)\n"
	);
	xtb_bt_new( HO_Dialog.win, "1st Size", HO_sdds_sl_fun, (xtb_data) &Use_HO_Previous_TC, aF(UHPTC_F) );
	xtb_bt_set(AF(UHPTC_F).win, Use_HO_Previous_TC, NULL);
	xtb_describe(aF(UHPTC_F),
		"For PostScript windows: use size information from the last hardcopy output. Currently, this\n"
		" ensures that the plot area of all following graphs is identical to that of the last graph\n"
		" printed without this option. No sanity checking is performed! Also not that some text may fall\n"
		" off the printed page (PSize won't correct for this; this does not matter when importing in e.g.\n"
		" Adobe Illustrator)\n"
	);

    xtb_to_new(HO_Dialog.win, "PS Marker size, incr:", XTB_TOP_LEFT, &dialogFont.font, &dialog_greekFont.font, aF(PSMBILBL_F));
    xtb_ti_new(HO_Dialog.win, "", D_DSP, MAXCHBUF, dfn_fun, (xtb_data) &psm_base, aF(PSMB_F));
	xtb_describe( aF(PSMB_F), "The initial size of PostScript markers\n");
    xtb_ti_new(HO_Dialog.win, "", D_DSP, MAXCHBUF, dfn_fun, (xtb_data) &psm_incr, aF(PSMI_F));
	xtb_describe( aF(PSMI_F), "The increase in size of PostScript markers after all have been used\n");
	{ char text[MAXCHBUF];
		sprintf(text, "%lg", psm_base);
		STRINGCHECK( text, sizeof(text));
		xtb_ti_set(AF(PSMB_F).win, text, (xtb_data) 0);
		sprintf(text, "%lg", psm_incr);
		STRINGCHECK( text, sizeof(text));
		xtb_ti_set(AF(PSMI_F).win, text, (xtb_data) 0);
	}
    xtb_to_new(HO_Dialog.win, "PS scale, pos, orient.:", XTB_TOP_LEFT, &dialogFont.font, &dialog_greekFont.font, aF(PSPOSLBL_F));
    xtb_ti_new(HO_Dialog.win, "", D_DSP, MAXCHBUF, dfn_fun, (xtb_data) &ps_scale, aF(PSSC_F));
	xtb_describe( aF(PSSC_F), "Scale (in percent) of PostScript dump\n 100% yields the dimensions given above\n");
/* 
    xtb_ti_new(HO_Dialog.win, "", D_DSP, MAXCHBUF, dfn_fun, (xtb_data) &ps_l_offset, aF(PS_LOFF_F));
	xtb_describe( aF(PS_LOFF_F), "Offset in pt. for left-aligned picture\n" );
    xtb_ti_new(HO_Dialog.win, "", D_DSP, MAXCHBUF, dfn_fun, (xtb_data) &ps_b_offset, aF(PS_BOFF_F));
	xtb_describe( aF(PS_BOFF_F), "Offset in pt. for bottom-aligned picture\n" );
 */
	xtb_br_new( HO_Dialog.win, 3, xposstrs, theWin_Info->ps_xpos, pos_fun, &ps_xpos, aF(PSPOSX_F) );
	xtb_describe( AF(PSPOSX_F).framelist[0], "Left of page (PS)\n Hold <Mod1> to edit the left margin offset\n" );
	xtb_describe( AF(PSPOSX_F).framelist[1], "Centre of page (PS)\n" );
	xtb_describe( AF(PSPOSX_F).framelist[2], "Right of page (PS)\n" );
	xtb_br_new( HO_Dialog.win, 3, yposstrs, theWin_Info->ps_ypos, pos_fun, &ps_ypos, aF(PSPOSY_F) );
	xtb_describe( AF(PSPOSY_F).framelist[0], "Bottom of page (PS)\n Hold <Mod1> to edit the bottom margin offset\n" );
	xtb_describe( AF(PSPOSY_F).framelist[1], "Centre of page (PS)\n" );
	xtb_describe( AF(PSPOSY_F).framelist[2], "Top of page (PS)\n" );

	xtb_br_new( HO_Dialog.win, 2, orientstrs, theWin_Info->print_orientation, orient_fun, NULL, aF(PSORIENT_F) );
	xtb_describe( AF(PSORIENT_F).framelist[0], "Portrait orientation (PS)\n" );
	xtb_describe( AF(PSORIENT_F).framelist[1], "Landscape orientation (PS)\n" );

	xtb_bt_new( HO_Dialog.win, "Page", HO_sdds_sl_fun, (xtb_data) &showpage, aF(PSSP_F) );
	xtb_describe( aF(PSSP_F), "Insert the PostScript print-and-eject-the-page\n command after hardcopying\n" );
	xtb_bt_new( HO_Dialog.win, "EPS", HO_sdds_sl_fun, (xtb_data) &psEPS, aF(PSEPS_F) );
	xtb_describe( aF(PSEPS_F),
		"Try to make better EPSF code. Notably:\n"
		" Align the figure to the paper's origin (lowerleft corner as seen in Portrait)\n"
		" Don't output a showpage at the end\n"
		" Don't call statusdict\n"
	);
	xtb_bt_new( HO_Dialog.win, "DSC", HO_sdds_sl_fun, (xtb_data) &psDSC, aF(PSDSC_F) );
	xtb_describe( aF(PSDSC_F),
		"Output a minimal number of DSC (document structure convention)\n"
		" information lines, like the bounding box containing the image, and\n"
		" the number of pages in the document. Since XGraph's estimation may not always\n"
		" be quite near the exact values, it is sometimes desirable to leave them out\n"
		" altogether. Bounding box information, for example, can be computed by ghostscript/\n"
		" gsview - but this is most straightforward when none is present in the input.\n"
		" The bounding box information is saved at the top of the file instead of at the end;\n"
		" this is for compatibility with programmes that expect it in the EPS header, but \n"
		" requires the window to be printed twice...\n"
	);
	xtb_bt_new( HO_Dialog.win, "PSize", HO_sdds_sl_fun, (xtb_data) &psSetPage, aF(PSSetPage_F) );
	xtb_describe( aF(PSSetPage_F),
		"Include PostScript code that will cause (certain) printers/imagers to adapt\n"
		" the canvas (paper) to the currently requested printing sizes, or otherwise\n"
		" to adapt the printing scale to the paper printed on.\n"
		" Set this option when PDF should be created from the generated PostScript, and/or\n"
		" the graph is larger than the default paper (usually A4).\n"
	);
	xtb_bt_new( HO_Dialog.win, "Done", NULL, (xtb_data) NULL, aF(DONE_F) );
	AF(DONE_F).description= done_descr;
	sprintf( AF(DONE_F).description, "Indicates whether the current window's last print may be uptodate\n" );
	HO_printit_win= AF(DONE_F).win;
	xtb_bt_new( HO_Dialog.win, "PS Page ###", HO_sdds_sl_fun, (xtb_data) &ps_page_nr, aF(PAGE_XGPS_F) );
	xtb_describe( aF(PAGE_XGPS_F),
		" PostScript: Current pagenumber: click to reset to 1\n"
		" [PS Settings]: dump PS print settings (fonts, dimensions, ..) in an XGraph dump\n"
	);
	xtb_bt_new( HO_Dialog.win, "Subset", PrintSetsNR_fun, (xtb_data) 0, aF(SETS_F) );
	xtb_bt_set( AF(SETS_F).win, (PrintSetsNR<=0)? 0 : 1, (xtb_data) 0 );
	xtb_describe( aF(SETS_F), PRSN_desc );

	xtb_bt_new( HO_Dialog.win, "X Font size", HO_sdds_sl_fun, (xtb_data) &use_X11Font_length, aF(UXLF_F) );
	AF(UXLF_F).description= fontest_descr;
	sprintf( AF(UXLF_F).description,
		"Determine the widths of strings on the metrics of the X-font used (scales the on-screen sizes)\n"
		" <Mod1>click to change fontwidth estimator EST used to calculate approximate (proportional) PS fontwidth:\n"
		" afw= pointsize* EST * VirtualDPI/ DPI= pointsize * %g * %g/ %g= pointsize * %g\n",
			Font_Width_Estimator, VDPI, POINTS_PER_INCH, Font_Width_Estimator* INCHES_PER_POINT* VDPI
	);
	xtb_bt_new( HO_Dialog.win, "GS", HO_sdds_sl_fun, (xtb_data) &use_gsTextWidth, aF(UGSTW_F) );
	xtb_describe( aF(UGSTW_F),
		"Use ghostscript (gs) to determine the printing widths of strings when generating PostScript\n"
		" output. gs must be in your path. In this mode, the Font_Width_Estimator (see [X Font Size])\n"
		" is used only in the approximation of the printed font widths. If a requested font is not in\n"
		" gs' font path ($GS_FONTPATH), an attempt is made to fall back to scaling the on-screen sizes\n"
		" <Mod1>click or hit <Mod1>-g to determine all currently relevant unknown widths in a single batch.\n"
	);
	xtb_bt_new( HO_Dialog.win, "Scale Plot Area: X", HO_sdds_sl_fun, (xtb_data) &set_scale_plot_area, aF(SPAX_F) );
	xtb_describe( aF(SPAX_F), "Controls whether or not the plot-area \nis scaled to maxWidth. NOTE:\n"
		" Undoes \"Preserve Aspect\" largely.\n"
	);
	xtb_bt_new( HO_Dialog.win, ": Y", HO_sdds_sl_fun, (xtb_data) &scale_plot_area_y, aF(SPAY_F) );
	xtb_describe( aF(SPAY_F), "Controls whether or not the plot-area \nis scaled to maxheigth. NOTE:\n"
		" Undoes \"Preserve Aspect\" largely.\n"
	);
	xtb_bt_new( HO_Dialog.win, "Info", HO_sdds_sl_fun, (xtb_data) &set_PS_PrintComment, aF(PSPC_F) );
	xtb_describe( aF(PSPC_F), "Print text under Info box on separate page\nOnly in PostScript\n");
	xtb_bt_new( HO_Dialog.win, "Sort Sheet", HO_sdds_sl_fun, (xtb_data) &set_Sort_Sheet, aF(SPSS_F) );
	xtb_describe( aF(SPSS_F), "Sort a (Cricket) SpreadSheet\non X-value\n");
	xtb_bt_new( HO_Dialog.win, "XGBounds", HO_sdds_sl_fun, (xtb_data) &XG_SaveBounds, aF(XGSB_F) );
	xtb_describe( aF(XGSB_F), "Include the bounding-box definition in\nan XGraph dump of the current window\n");
	xtb_bt_new( HO_Dialog.win, "XGStrip", HO_sdds_sl_fun, (xtb_data) &XG_Stripped, aF(XGSTR_F) );
	xtb_describe( aF(XGSTR_F),
		"Dump only the displayed data (sets AND points) in\n"
		" an XGraph or SpreadSheet dump of the current window\n"
		"<Mod1>-s\n"
	);

	xtb_bt_new( HO_Dialog.win, "DumpAv", HO_sdds_sl_fun, (xtb_data) &dump_average_values, aF(XGDAV_F) );
	xtb_describe( aF(XGDAV_F), "Dump *AVERAGE* values in\nan XGraph dump of the current window\ninstead of the command\n<Mod1>-a\n");

	xtb_bt_new( HO_Dialog.win, "DumpProc", HO_sdds_sl_fun, (xtb_data) &DumpProcessed, aF(XGDPR_F) );
	xtb_describe( aF(XGDPR_F), "Dump processed values in\nan XGraph dump of the current window\ninstead of the raw data\n<Mod1>-p\n");
	xtb_bt_new( HO_Dialog.win, "Fresh", HO_sdds_sl_fun, (xtb_data) &DProcFresh, aF(XGDPFRESH_F) );
	xtb_describe( aF(XGDPFRESH_F),
		"Redraw the window if necessary to obtain up-to-date values\n"
		" when dumping processed values to an XGraph dump\n"
	);

	xtb_bt_new( HO_Dialog.win, "SDisc", HO_sdds_sl_fun, (xtb_data) &splits_disconnect, aF(XGSLDSC_F) );
	xtb_describe( aF(XGSLDSC_F),
		"When set, set splits (*SPLIT*) disconnect at the split to form a new dataset\n"
		" Unset, splits are interpreted as pen-lifts within a single dataset.\n"
	);

	xtb_bt_new( HO_Dialog.win, "Binary", HO_sdds_sl_fun, (xtb_data) &DumpBinary, aF(XGDBIN_F) );
	xtb_describe( aF(XGDBIN_F), "Dump binary values in\nan XGraph dump of the current window\ninstead of ASCII data\n<Mod1>-b\n");
	xtb_bt_new( HO_Dialog.win, "DAsc", HO_sdds_sl_fun, (xtb_data) &DumpAsAscanf, aF(XGDASC_F) );
	xtb_describe( aF(XGDASC_F),
		"While dumping an XGraph file, check each value if it is an ascanf string.\n"
		" If so, print the string instead of the value. This setting is mutually exclusive\n"
		" with the [Binary] dump setting.\n"
	);
	xtb_bt_new( HO_Dialog.win, "DHex", HO_sdds_sl_fun, (xtb_data) &DumpDHex, aF(XGDHEX_F) );
	xtb_describe( aF(XGDHEX_F),
		"Set the *DPRINTF* format to '%dhex' while dumping. This has the effect that ascanf double variables,\n"
		" set associations and set-data (when Binary is off) are printing in a hexadecimal format\n"
		" This preserves the full precision, taking at most 19 bytes instead of 16 for binary printing.\n"
		" The advantage of binary is that it is a textual/printable format, independant of the machine's\n"
		" endianness. <Mod1>-h\n"
	);
	xtb_bt_new( HO_Dialog.win, "Pens", HO_sdds_sl_fun, (xtb_data) &DumpPens, aF(XGDPEN_F) );
	xtb_describe( aF(XGDPEN_F), "Add *EVAL* statements that restore the current state of the pens.\n");

	xtb_bt_new( HO_Dialog.win, "Complete", HO_sdds_sl_fun, (xtb_data) &Init_XG_Dump, aF(XGINI_F) );
	xtb_describe( aF(XGINI_F),
		"Dump a complete XGraph file, or just the set-related info+data\n"
		" Unselect to generate \"include\" files, notably combined with\n"
		" XGStrip to dump only displayed sets.\n"
	);

	xtb_bt_new( HO_Dialog.win, "2Page", HO_sdds_sl_fun, (xtb_data) &ps_mpage, aF(PSMP_F) );
	xtb_describe( aF(PSMP_F), "Automatically adjust PS position\n and showpage settings to print\n 2 graphs on a single page\n");
	xtb_bt_new( HO_Dialog.win, "Margins", HO_sdds_sl_fun, (xtb_data) &ps_show_margins, aF(PSSM_F) );
	xtb_describe( aF(PSSM_F), "Print crosses in corners as safe-margins indicators\n");
	xtb_bt_new( HO_Dialog.win, "PS  RGB", HO_sdds_sl_fun, (xtb_data) &ps_coloured, aF(PSRGB_F) );
	{ char buf[512];
		snprintf( buf, 512,
			" [PS RGB]: Output PostScript using colours (RGB)\n"
			" [XG RGB]: Store colournames in XGraph dump\n"
			"           (Hold Mod1 to toggle TrueGray display [%3s]\n"
			"<Mod1>-r\n                                                                          ",
			(TrueGray)? "ON" : "OFF"
		);
		xtb_describe( aF(PSRGB_F), buf );
	}
	xtb_bt_new( HO_Dialog.win, "PS   trsp", HO_sdds_sl_fun, (xtb_data) &ps_transparent, aF(PSTRS_F) );
	xtb_describe( aF(PSTRS_F), 
		" [PS trsp]; Transparent: don't paint background.\n"
		" [KeyEVAL]; Dump *KEY_EVAL*s to an XGraph dump\n"
	);

    xtb_to_new(HO_Dialog.win, "Title Font:", XTB_TOP_LEFT, &dialogFont.font, &dialog_greekFont.font, aF(TFFAMLBL_F));
    xtb_ti_new(HO_Dialog.win, "", MFNAME, MAXCHBUF, df_fun, (xtb_data) &titleFont, aF(TFFAM_F));
    info->tf_family = AF(TFFAM_F).win;
    xtb_to_new(HO_Dialog.win, "Size:", XTB_TOP_LEFT, &dialogFont.font, &dialog_greekFont.font, aF(TFSIZLBL_F));
    xtb_ti_new(HO_Dialog.win, "", D_FS, MAXCHBUF, dfn_fun, (xtb_data) &tf_size, aF(TFSIZ_F));
    info->tf_size = AF(TFSIZ_F).win;

    xtb_to_new(HO_Dialog.win, "Legend Font:", XTB_TOP_LEFT, &dialogFont.font, &dialog_greekFont.font, aF(LEFFAMLBL_F));
    xtb_ti_new(HO_Dialog.win, "", MFNAME, MAXCHBUF, df_fun, (xtb_data) &legendFont, aF(LEFFAM_F));
    info->lef_family = AF(LEFFAM_F).win;
    xtb_to_new(HO_Dialog.win, "Size:", XTB_TOP_LEFT, &dialogFont.font, &dialog_greekFont.font, aF(LEFSIZLBL_F));
    xtb_ti_new(HO_Dialog.win, "", D_FS, MAXCHBUF, dfn_fun, (xtb_data) &lef_size, aF(LEFSIZ_F));
    info->lef_size = AF(LEFSIZ_F).win;

    xtb_to_new(HO_Dialog.win, "Label Font:", XTB_TOP_LEFT, &dialogFont.font, &dialog_greekFont.font, aF(LAFFAMLBL_F));
    xtb_ti_new(HO_Dialog.win, "", MFNAME, MAXCHBUF, df_fun, (xtb_data) &labelFont, aF(LAFFAM_F));
    info->laf_family = AF(LAFFAM_F).win;
    xtb_to_new(HO_Dialog.win, "Size:", XTB_TOP_LEFT, &dialogFont.font, &dialog_greekFont.font, aF(LAFSIZLBL_F));
    xtb_ti_new(HO_Dialog.win, "", D_FS, MAXCHBUF, dfn_fun, (xtb_data) &laf_size, aF(LAFSIZ_F));
    info->laf_size = AF(LAFSIZ_F).win;

    xtb_to_new(HO_Dialog.win, "Axis  Font:", XTB_TOP_LEFT, &dialogFont.font, &dialog_greekFont.font, aF(AFFAMLBL_F));
    xtb_ti_new(HO_Dialog.win, "", MFNAME, MAXCHBUF, df_fun, (xtb_data) &axisFont, aF(AFFAM_F));
    info->af_family = AF(AFFAM_F).win;
    xtb_to_new(HO_Dialog.win, "Size:", XTB_TOP_LEFT, &dialogFont.font, &dialog_greekFont.font, aF(AFSIZLBL_F));
    xtb_ti_new(HO_Dialog.win, "", D_FS, MAXCHBUF, dfn_fun, (xtb_data) &af_size, aF(AFSIZ_F));
    info->af_size = AF(AFSIZ_F).win;

    xtb_bt_new(HO_Dialog.win, "Ok [RETURN (closes graph)]", HO_ok_fun, (xtb_data) info, aF(OK_F));
	xtb_describe( aF(OK_F),
		"Clicking this button prints, and closes dialogue\n"
		"Hitting <Enter> (on keypad) prints and leaves dialogue\n"
		"Hitting <Return> prints and closes the dialogue and graph window\n"
		"Holding <Mod1> closes only the dialogue after a dump\n"
		"Leave the file name blank to obtain a sensible default\n"
	);
    xtb_bt_new(HO_Dialog.win, "Cancel [^D]", can_fun, (xtb_data) 0, aF(CAN_F));

	xtb_bt_new(HO_Dialog.win, "About", copyright_function, (xtb_data) info, aF(ABOUT_F));
	xtb_describe( aF(ABOUT_F), COPYRIGHT );

    xtb_bt_new(HO_Dialog.win, "Redraw [^R]", redraw_fun, (xtb_data) 0, aF(REDRAW_F));
	xtb_describe( aF(REDRAW_F), "Redraw the graph window belonging to this dialogue,\npossibly showing new settings\n");
    xtb_bt_new(HO_Dialog.win, "Settings [^S]", settings_fun, (xtb_data) 0, aF(SETTINGS_F));
	xtb_describe( aF(SETTINGS_F), "Close this dialogue, and open the settings dialogue\n");

    xtb_bt_new(HO_Dialog.win, "Rewrite", rewrite_fun, (xtb_data) 0, aF(REWRITE_F));
	xtb_describe( aF(REWRITE_F), "Rewrite each set to the file associated with it\n"
		" appending to files already rewritten in case multiple sets share the same file.\n"
		" Sets are rewritten using the current settings (including DumpProcessed!), except for XGStrip.\n"
		" A 1sec pause is imposed between the rewrite of each new file to store some\n"
		" of the set-order in the file-creation date\n"
	);

      /* Dividing bar */
    max_width = 0;
    for (i = 0;  i < ((int) BAR_F);  i++) {
		if (AF(i).width > max_width) max_width = AF(i).width;
    }
    xtb_bk_new(HO_Dialog.win, max_width - BAR_SLACK, 1, aF(BAR_F));

    /* Set device specific info */
    (void) HO_dev_fun(info->choices, -1,xtb_br_get(info->choices),(xtb_data) info);
    (void) fd_fun(info->fod, -1, xtb_br_get(info->fod), (xtb_data) info);

	HO_info= info;

    /* 
     * Now place elements - could make this one expression but pcc
     * is too wimpy.
     */
	xtb_fmt_tabset= 0;
	 /* 20010721: Let us suppose that all tabset fields were correctly initialised to 0
	  \ (by the system: we don't want to use tabs in this dialog). Thus, we can do the
	  \ obligatory frames selection with the following command:
	  */
	xtb_select_frames_tabset( LAST_F, ho_af, 0, NULL );

	cntrl= xtb_vert(XTB_CENTER, D_VPAD/2, D_INT/2,
			xtb_hort( XTB_CENTER, D_HPAD, D_INT,
				xtb_w( aF(ODEVLBL_F)), xtb_w( aF(ODEVROW_F)), NE
			),
			xtb_hort( XTB_CENTER, D_HPAD, D_INT,
				xtb_w( aF(DISPLBL_F)), xtb_w( aF(DISPROW_F)), xtb_w( aF(PRESTIME_F)), NE
			),
			xtb_hort( XTB_CENTER, D_HPAD, D_INT,
				xtb_w( aF(FDLBL_F)),
				xtb_vert( XTB_CENTER, D_VPAD/2, D_INT/2,
					xtb_w( aF(DIR_F)), xtb_w( aF(FDINP_F)), NE
				), NE
			),
			NE
	);

	mindim= xtb_vert(XTB_CENTER, D_HPAD, D_INT,
				xtb_hort(XTB_CENTER, D_HPAD, D_INT,
					xtb_vert(XTB_RIGHT, D_VPAD, AF(MWDIMI_F).height,
						xtb_w( aF(MDIMLBL_F)),
						xtb_w( aF(PSMBILBL_F)),
						xtb_w( aF(PSPOSLBL_F)),
						NE),
					xtb_vert(XTB_LEFT, D_VPAD, D_INT,
						xtb_hort(XTB_CENTER, D_HPAD, D_INT,
							xtb_w(aF(MWDIMI_F)),
							xtb_w(aF(MHDIMI_F)),
							xtb_w(aF(XWR_F)),
							xtb_w(aF(PSSA_F)),
							xtb_w(aF(XGAWD_F)),
							xtb_w(aF(UHPTC_F)),
						NE),
						xtb_hort(XTB_CENTER, D_HPAD, D_INT,
							xtb_w(aF(PSMB_F)),
							xtb_w(aF(PSMI_F)),
							xtb_w( aF(DONE_F)), xtb_w( aF(PAGE_XGPS_F)), xtb_w( aF(SETS_F)),
						NE),
						xtb_hort(XTB_CENTER, D_HPAD, D_INT,
							xtb_w( aF(PSSC_F)),
/* 							xtb_vert( XTB_LEFT, D_VPAD, D_INT,	*/
								xtb_w( aF(PSPOSX_F)),
/* 									xtb_w(aF(PS_LOFF_F)), NE),	*/
/* 							xtb_vert( XTB_LEFT, D_VPAD, D_INT,	*/
								xtb_w( aF(PSPOSY_F)),
/* 									xtb_w(aF(PS_BOFF_F)), NE),	*/
							xtb_w( aF(PSORIENT_F)),
							xtb_vert( XTB_CENTER, D_VPAD/2, D_INT/2,
								xtb_hort( XTB_LEFT, D_HPAD/2, D_INT/2,
									xtb_w( aF(PSSP_F)), xtb_w( aF(PSSetPage_F)), NE),
								xtb_hort( XTB_LEFT, D_HPAD/2, D_INT/2,
									xtb_w( aF(PSEPS_F)), xtb_w( aF(PSDSC_F)), NE),
							NE),
						NE),
					NE),
				NE),
				xtb_vert(XTB_CENTER, D_VPAD, D_INT,
					xtb_hort(XTB_CENTER, D_HPAD, D_INT,
						xtb_hort(XTB_CENTER, D_HPAD, 0, xtb_w( aF(UXLF_F) ), xtb_w(aF(UGSTW_F)), NE),
						xtb_w( aF(SPAX_F)), xtb_w( aF(SPAY_F)), xtb_w( aF(PSPC_F)), xtb_w( aF(SPSS_F)),
						xtb_w( aF(XGINI_F)), xtb_w( aF(XGDPFRESH_F)), xtb_w( aF(XGDBIN_F)),
						xtb_w( aF(XGDASC_F)), xtb_w( aF(XGDHEX_F)),
					NE),
					xtb_hort(XTB_CENTER, D_HPAD, D_INT,
						xtb_w( aF(XGSB_F)), xtb_w( aF(XGSTR_F)), xtb_w( aF(XGDAV_F)), xtb_w( aF(XGDPR_F)),
						xtb_w( aF(XGDPEN_F)), xtb_w( aF(XGSLDSC_F)),
						xtb_w( aF(PSMP_F)), xtb_w( aF(PSSM_F)), xtb_w( aF(PSRGB_F)), xtb_w( aF(PSTRS_F)),
					NE),
				NE),
			NE);

    font_area = xtb_vert(XTB_RIGHT, D_VPAD/2, D_INT/2,
				  xtb_hort(XTB_CENTER, D_HPAD, D_INT,
					   xtb_w(aF(TFFAMLBL_F)),
					   xtb_w(aF(TFFAM_F)),
					   xtb_w(aF(TFSIZLBL_F)),
					   xtb_w(aF(TFSIZ_F)), NE),
				  xtb_hort(XTB_CENTER, D_HPAD, D_INT,
					   xtb_w(aF(LEFFAMLBL_F)),
					   xtb_w(aF(LEFFAM_F)),
					   xtb_w(aF(LEFSIZLBL_F)),
					   xtb_w(aF(LEFSIZ_F)), NE),
				  xtb_hort(XTB_CENTER, D_HPAD, D_INT,
					   xtb_w(aF(LAFFAMLBL_F)),
					   xtb_w(aF(LAFFAM_F)),
					   xtb_w(aF(LAFSIZLBL_F)),
					   xtb_w(aF(LAFSIZ_F)), NE),
				  xtb_hort(XTB_CENTER, D_HPAD, D_INT,
					   xtb_w(aF(AFFAMLBL_F)),
					   xtb_w(aF(AFFAM_F)),
					   xtb_w(aF(AFSIZLBL_F)),
					   xtb_w(aF(AFSIZ_F)), NE),
		      NE);

    def = xtb_fmt_do( xtb_vert(XTB_CENTER, D_VPAD/2, D_INT/2,
			xtb_hort( XTB_CENTER_J, 0, 0,
				xtb_vert( XTB_CENTER, D_HPAD/2, D_INT/2,
					xtb_w(aF(TITLE_F)),
					cntrl,
					xtb_w(aF(BAR_F)),
					mindim,
					font_area,
					NE
				),
				NE
			),
			xtb_hort(XTB_JUST, D_HPAD, D_INT,
				 xtb_w(aF(OK_F)), xtb_w(aF(CAN_F)), xtb_w( aF(ABOUT_F)), xtb_w(aF(REDRAW_F)),
				 xtb_w(aF(REWRITE_F)), xtb_w(aF(SETTINGS_F)),
				 NE
			),
			NE
		),
		&frame->width, &frame->height
	);
    xtb_mv_frames(LAST_F, ho_af);
    xtb_fmt_free(def);

    frame->width += ( D_BRDR);
    frame->height += ( D_BRDR);

    /* Make window large enough to contain the info */
    XResizeWindow(disp, HO_Dialog.win, frame->width, frame->height);
    hints.flags = USSize|USPosition;	/* PSize;	*/
    hints.width = frame->width;
    hints.height = frame->height;
    XSetNormalHints(disp, HO_Dialog.win, &hints);

	XSelectInput(disp, HO_Dialog.win,
		VisibilityChangeMask|ExposureMask|StructureNotifyMask|ButtonPressMask|ButtonReleaseMask|KeyPressMask|EnterWindowMask|LeaveWindowMask
	);

    diag_cursor = XCreateFontCursor(disp, XC_spraycan );
    fg_color.pixel = normPixel;
    XQueryColor(disp, cmap, &fg_color);
    bg_color.pixel = bgPixel;
    XQueryColor(disp, cmap, &bg_color);
    XRecolorCursor(disp, diag_cursor, &fg_color, &bg_color);
    XDefineCursor(disp, HO_Dialog.win, diag_cursor);
    frame->width += (2 * D_BRDR);
    frame->height += (2 * D_BRDR);
    *HO_okbtn = AF(OK_F);
    *HO_canbtn = AF(CAN_F);
    *HO_redrawbtn = AF(REDRAW_F);

	HO_Dialog.frames= LAST_F;
	HO_Dialog.framelist= &local_ho_af;

	xtb_init(disp, screen, normPixel, bgPixel, dialogFont.font, dialog_greekFont.font, False );
}



#define SH_W	5
#define SH_H	5

int destroy_it= 0;
extern xtb_frame SD_Dialog, sd_af[];

int Handle_HO_Event( XEvent *evt, int *handled, xtb_hret *xtb_return, int *level, int handle_others )
{
  char keys[4]= "\0\0\0\0";
  KeySym keysymbuffer[4]= {0,0,0,0};
  int rd= theWin_Info->redraw, nbytes, keysyms;

	if( Exitting ){
		return(1);
	}
	if( HO_Dialog.destroy_it ){
		_CloseHO_Dialog( &HO_Dialog, True );
		return(0);
	}

	if( debugFlag ){
		fprintf( StdErr, "Handle_HO_Event(%d,\"%s\") #%lu handled=%d, handle_others=%d\n",
			level, event_name(evt->type), evt->xany.serial,
			*handled, handle_others
		);
		fflush( StdErr );
	}
	if( evt->xany.type== KeyPress || evt->xany.type== KeyRelease || evt->xany.type== ButtonPress ||
		evt->xany.type== ButtonRelease || evt->xany.type== MotionNotify
	){
		xtb_modifier_state= xtb_Mod2toMod1( 0x00FF & evt->xbutton.state );
		xtb_button_state= 0xFF00 & evt->xbutton.state;
	}
	else{
		xtb_modifier_state= 0;
		xtb_button_state= 0;
	}
	switch( evt->type){
		case ClientMessage:{
			if( HO_Dialog.win== evt->xclient.window && evt->xclient.data.l[0]== wm_delete_window &&
				strcmp( XGetAtomName(disp, evt->xclient.message_type), "WM_PROTOCOLS")== 0
			){
				_CloseHO_Dialog( &HO_Dialog, True );
			}
			break;
		}
		case UnmapNotify:
			if( evt->xany.window== HO_Dialog.win ){
				HO_Dialog.mapped= 0;
			}
			break;
		case MapNotify:
			if( evt->xany.window== HO_Dialog.win ){
				HO_Dialog.mapped= 1;
			}
			break;
		case ConfigureNotify:
			if( evt->xany.window== HO_Dialog.win ){
			  XConfigureEvent *e= (XConfigureEvent*) evt;
			  int width= e->width, height= e->height;
			  Window dummy;
				if( HO_Dialog.width!= width+ 2* D_BRDR || HO_Dialog.height!= height+ 2* D_BRDR ){
					XResizeWindow(disp, HO_Dialog.win, HO_Dialog.width- 2* D_BRDR, HO_Dialog.height- 2* D_BRDR);
					xtb_error_box( theWin_Info->window,
						"Sorry, Dialog-sizes are fixed, depending\n"
						"on area required for the buttons etc.\n",
						"Note"
					);
				}
				XTranslateCoordinates( disp, HO_Dialog.win, RootWindow(disp, screen),
						  0, 0, &e->x, &e->y, &dummy
				);
				if( debugFlag ){
					fprintf( StdErr, "HO_Dialog %dx%d+%d+%d ConfigureEvent %dx%d+%d+%d-%d\n",
						HO_Dialog.width, HO_Dialog.height, HO_Dialog.x_loc, HO_Dialog.y_loc,
						e->width, e->height, e->x, e->y, WM_TBAR
					);
				}
				e->y-= WM_TBAR;
				if( HO_Dialog.x_loc!= e->x || HO_Dialog.y_loc!= e->y ){
					pw_centre_on_X= e->x+ width/2+ e->border_width;
					pw_centre_on_Y= e->y+ height/2+ e->border_width;
					HO_Dialog.x_loc= e->x;
					HO_Dialog.y_loc= e->y;
				}
				*handled= 1;
			}
			  /* fall through to Expose handler	*/
		case Expose:
			if( evt->xany.window== HO_Dialog.win ){
			  XPoint line[3];
			  GC lineGC;
			  extern GC xtb_set_gc();
			  extern unsigned long xtb_light_pix, xtb_back_pix, xtb_norm_pix;

				line[0].x= 0;
				line[0].y= (short) HO_Dialog.height- 2* D_BRDR- 1;
				line[1].x= (short) HO_Dialog.width- 2* D_BRDR- 1;
				line[1].y= (short) HO_Dialog.height- 2* D_BRDR- 1;
				line[2].x= (short) HO_Dialog.width- 2* D_BRDR- 1;
				line[2].y= 0;
				lineGC= xtb_set_gc( HO_Dialog.win, xtb_back_pix, xtb_light_pix, dialogFont.font->fid);
				XSetLineAttributes( disp, lineGC, 2, LineSolid, CapButt, JoinMiter);
				XDrawLines(disp, HO_Dialog.win, lineGC, line, 3, CoordModeOrigin);

				line[0].x= line[1].x= line[1].y= line[2].y= 1;
				line[0].y= (short) HO_Dialog.height- 2* D_BRDR- 1;
				line[2].x= (short) HO_Dialog.width- 2* D_BRDR- 1;
				lineGC= xtb_set_gc( HO_Dialog.win, xtb_norm_pix, xtb_light_pix, dialogFont.font->fid);
				XSetLineAttributes( disp, lineGC, 2, LineSolid, CapButt, JoinMiter);
				XDrawLines(disp, HO_Dialog.win, lineGC, line, 3, CoordModeOrigin);

				if( evt->xexpose.send_event && evt->type== Expose ){
				  int i;
				  Window wi= evt->xexpose.window;
				  int w= evt->xexpose.width, h= evt->xexpose.height;
					if( debugFlag ){
						fprintf( StdErr,
							"Handle_HO_Event(): resending event (#%ld, %s, s_e=%d) for window %ld\n",
							evt->xany.serial, event_name(evt->type), evt->xany.send_event, evt->xany.window
						);
					}
					for( i= 0; i< LAST_F; i++ ){
						evt->xexpose.window= AF(i).win;
						evt->xexpose.width= AF(i).width;
						evt->xexpose.height= AF(i).height;
						XSendEvent( disp, evt->xexpose.window, True, ExposureMask, evt );
						xtb_dispatch(disp, HO_Dialog.win, LAST_F, ho_af, evt);
					}
					evt->xexpose.window= wi;
					evt->xexpose.width= w;
					evt->xexpose.height= h;
				}
				*handled= 1;
			}
			break;
		case ButtonPress:
			if( evt->xany.window== HO_Dialog.win && CheckMask(evt->xbutton.state,ControlMask) ){
				xtb_error_box( theWin_Info->window,
					"Controls and generates various forms of output.\n"
					"Control-Click on the fields to pop-up a short description\n",
					"Print/Save Dialog"
				);
				*handled= 1;
			}
			break;
		case ButtonRelease:
			if( evt->xany.window== HO_Dialog.win ){
				XSetInputFocus( disp, PointerRoot, RevertToParent, CurrentTime);
			}
			break;
		case KeyPress:{
		  xtb_frame *frame= xtb_lookup_frame( evt->xany.window);
			if( evt->xany.window== HO_Dialog.win || (frame && frame->parent== HO_Dialog.win) ){
				nbytes = XLookupString(&evt->xkey, keys, 4,
							   (KeySym *) 0, (XComposeStatus *) 0
				);
				keysymbuffer[0]= XLookupKeysym( (XKeyPressedEvent*) &evt->xkey, 0);
				sprintf( ps_comment, "%s\n%% Handle_HO_Event(): %s; mask=%s; keys[%d]",
					ps_comment, XKeysymToString(keysymbuffer[0]), xtb_modifiers_string(evt->xbutton.state),
					nbytes
				);
				STRINGCHECK( ps_comment, sizeof(ps_comment));
				if( keysymbuffer[0]!= NoSymbol ){
				  int flag;
				  xtb_data linfo;
					for( keysyms= 1; keysyms< nbytes && keysyms< 4; keysyms++){
						keysymbuffer[keysyms]= XLookupKeysym( (XKeyPressedEvent*) &evt->xkey, 0);
					}
					if( keysymbuffer[0]== XK_Return && !CheckMask(xtb_modifier_state, ControlMask|Mod1Mask) ){
						  /* unset the shiftmask, because this can cause stray windows to keep hanging around
						   \ on some X servers (this happened to me, so don't argue :))
						   */
						xtb_modifier_state&= ~ShiftMask;
						  /* Print and close graph	*/
						HO_ok_fun( HO_okbtn.win, 0, ok_info);
						  /* This is for some window managers that otherwise hog the system for much longer: */
						sleep(1);
						XSync( disp, True );
						if( Close_OK && !ok_info->errnr ){
							HO_Dialog.mapped= -1;
							destroy_it= 1;
							CloseHO_Dialog(&HO_Dialog);
							return(1);
						}
						*handled= 1;
					}
					else if( keysymbuffer[0]== XK_KP_Enter && !CheckMask(xtb_modifier_state, ControlMask|Mod1Mask) ){
					  /* print and leave dialogue	*/
						HO_ok_fun( HO_okbtn.win, 0, ok_info);
						XSync( disp, True );
						*handled= 1;
					}
					else if( keysymbuffer[0]== 'p' && CheckExclMask(xtb_modifier_state, ShiftMask|Mod1Mask) ){
					  int state= xtb_br_get(HO_d_info->choices);
					  /* Postscript output	*/
						xtb_br_set( AF(ODEVROW_F).win, PS_DEVICE );
						HO_dev_fun(HO_d_info->choices, state, PS_DEVICE, HO_d_info );
						*handled= 1;
					}
					else if( keysymbuffer[0]== 'x' && CheckExclMask(xtb_modifier_state, ShiftMask|Mod1Mask) ){
					  int state= xtb_br_get(HO_d_info->choices);
					  /* XGraph output */
						xtb_br_set( AF(ODEVROW_F).win, XGRAPH_DEVICE );
						HO_dev_fun(HO_d_info->choices, state, XGRAPH_DEVICE, HO_d_info );
						*handled= 1;
					}
					else if( keysymbuffer[0]== 't' && CheckExclMask(xtb_modifier_state, ShiftMask|Mod1Mask) ){
					  int state= xtb_br_get(HO_d_info->fod);
					  /* To Device */
						xtb_br_set( HO_d_info->fod, 0 );
						fd_fun( HO_d_info->fod, state, 0, HO_d_info );
						*handled= 1;
					}
					else if( keysymbuffer[0]== 'a' && CheckExclMask(xtb_modifier_state, ShiftMask|Mod1Mask) ){
					  int state= xtb_br_get(HO_d_info->fod);
					  /* Append File */
						xtb_br_set( HO_d_info->fod, 1 );
						fd_fun( HO_d_info->fod, state, 1, HO_d_info );
						*handled= 1;
					}
					else if( keysymbuffer[0]== 'n' && CheckExclMask(xtb_modifier_state, ShiftMask|Mod1Mask) ){
					  int state= xtb_br_get(HO_d_info->fod);
					  /* New File */
						xtb_br_set( HO_d_info->fod, 2 );
						fd_fun( HO_d_info->fod, state, 2, HO_d_info );
						*handled= 1;
					}
					else if( keysymbuffer[0]== 'v' && CheckExclMask(xtb_modifier_state, ShiftMask|Mod1Mask) ){
					  int state= xtb_br_get(HO_d_info->fod);
					  /* PS Preview */
						xtb_br_set( HO_d_info->fod, 3 );
						fd_fun( HO_d_info->fod, state, 3, HO_d_info );
						*handled= 1;
					}
					else if( keysymbuffer[0]== 'b' && CheckExclMask(xtb_modifier_state, Mod1Mask) ){
					  /* Binary output */
						flag= xtb_bt_get( AF(XGDBIN_F).win, &linfo );
						xtb_modifier_state&= ~Mod1Mask;
						HO_sdds_sl_fun( AF(XGDBIN_F).win, flag, linfo );
						xtb_modifier_state|= Mod1Mask;
						*handled= 1;
					}
					else if( keysymbuffer[0]== 'g' && CheckExclMask(xtb_modifier_state, Mod1Mask) ){
/* 						flag= xtb_bt_get( AF(UGSTW_F).win, &linfo );	*/
/* 						HO_sdds_sl_fun( AF(UGSTW_F).win, flag, linfo );	*/
						if( use_gsTextWidth ){
							gsTextWidth_Batch();
						}
						*handled= 1;
					}
					else if( keysymbuffer[0]== 'h' && CheckExclMask(xtb_modifier_state, Mod1Mask) ){
					  /* DHex output */
						flag= xtb_bt_get( AF(XGDHEX_F).win, &linfo );
						HO_sdds_sl_fun( AF(XGDHEX_F).win, flag, linfo );
						*handled= 1;
					}
					else if( keysymbuffer[0]== 's' && CheckExclMask(xtb_modifier_state, Mod1Mask) ){
					  /* XGStrip */
						flag= xtb_bt_get( AF(XGSTR_F).win, &linfo );
						HO_sdds_sl_fun( AF(XGSTR_F).win, flag, linfo );
						*handled= 1;
					}
					else if( keysymbuffer[0]== 'a' && CheckExclMask(xtb_modifier_state, Mod1Mask) ){
					  /* DumpAv */
						flag= xtb_bt_get( AF(XGDAV_F).win, &linfo );
						HO_sdds_sl_fun( AF(XGDAV_F).win, flag, linfo );
						*handled= 1;
					}
					else if( keysymbuffer[0]== 'p' && CheckExclMask(xtb_modifier_state, Mod1Mask) ){
					  /* DumpProc */
						flag= xtb_bt_get( AF(XGDPR_F).win, &linfo );
						HO_sdds_sl_fun( AF(XGDPR_F).win, flag, linfo );
						*handled= 1;
					}
					else if( keysymbuffer[0]== 'r' && CheckExclMask(xtb_modifier_state, Mod1Mask) ){
					  /* PS/XG RGB */
					  int xms= xtb_modifier_state;
						xtb_modifier_state&= ~Mod1Mask;
						flag= xtb_bt_get( AF(PSRGB_F).win, &linfo );
						HO_sdds_sl_fun( AF(PSRGB_F).win, flag, linfo );
						xtb_modifier_state= xms;
						*handled= 1;
					}
					else if( keysymbuffer[0]== XK_F4 && CheckMask( xtb_modifier_state, ShiftMask) ){
						XSetInputFocus( disp, PointerRoot, RevertToParent, CurrentTime);
						*handled= 1;
					}
					if( CheckMask( xtb_modifier_state, ControlMask|Mod1Mask) ){
						if( keysymbuffer[0]>= 0 && keysymbuffer[0]< 256 ){
							if( CheckMask( xtb_modifier_state, ShiftMask) ){
								keysymbuffer[0]= toupper( keysymbuffer[0] );
							}
							HO_cycle_focus_button( keysymbuffer[0] );
							*handled= 1;
						}
					}
				}
			}
			else{
				nbytes= 0;
			}
			  /* 20000229: why-o-why did I put this here?
			*handled= 0;
			   */
			if( nbytes && !*handled ){
				  /* see comment in dialog_s.c	*/
				if( keys[0]== 0x04 ){
					can_fun( HO_canbtn.win, 0, cinfo);
					*handled= 1;
					return(1);
				}
				else if( keys[0]== 0x12 ){
					redraw_fun( HO_redrawbtn.win, 0, rinfo);
					*handled= 1;
				}
				else if( keys[0]== 0x13 /* && evt->xany.window== HO_Dialog.win */ ){
					do_settings= 1;
					*handled= 0;
					return(1);
				}
				else if( keys[0]== '\t' ){
				  Window w= get_text_box(1)->win;
				  int dummy;
					XGetInputFocus( disp, &w, &dummy );
					goto_next_text_box( w );
				}
				else if( keys[0]== ' ' ){
					theWin_Info->halt= 1;
				}
			}
			break;
		}
	}
	if( !*handled ){
		*xtb_return= xtb_dispatch(disp, HO_Dialog.win, LAST_F, &ho_af[0], evt);
		  /* handle_others: reliq from an era where ho_dialog() still had its
		   \ own event-handling loop. Now all events are processed in _Handle_An_Event()
		   \ in xgraph.c, albeit the dialogs have their own handler-routines called
		   \ by the main handler.
			*/
		if( HO_Dialog.destroy_it ){
			_CloseHO_Dialog( &HO_Dialog, True );
			return(0);
		}
		if( handle_others ){
			if( *xtb_return!= XTB_HANDLED && *xtb_return!= XTB_STOP ){
				if( SD_Dialog.mapped> 0 ){
				  extern int Handle_SD_Event();
					Handle_SD_Event( evt, handled, xtb_return, level, 0 );
					if( Exitting ){
						return(1);
					}
				}
			}
			if( *xtb_return!= XTB_HANDLED && *xtb_return!= XTB_STOP ){
				_Handle_An_Event( evt, *level, 0, "ho_dialog" );
				if( Exitting ){
					return(1);
				}
			}
		}
	}
	if( theWin_Info ){
		if( rd!= theWin_Info->redraw ){
			xtb_bt_set( HO_redrawbtn.win, theWin_Info->redraw, NULL);
		}
		xtb_bt_set( AF(DONE_F).win, theWin_Info->printed, NULL);
	}
	if( HO_d_info->printOK ){
		destroy_it= (HO_d_info->printOK== -1)? 1 : 0;
		HO_d_info->printOK= 0;
		HO_ok_fun( HO_okbtn.win, 0, ok_info);
		CloseHO_Dialog(&HO_Dialog);
		return(1);
	}
	return(0);
}

extern LocalWin *thePrintWin_Info;

void CloseHO_Dialog( xtb_frame *dial)
{
	_CloseHO_Dialog( dial, False );
}

void _CloseHO_Dialog( xtb_frame *dial, Boolean destroy)
{ XEvent evt;
  extern Window thePrintWindow;
  LocalWin *win_info= theWin_Info;
	if( disp && dial ){
		if( dial->destroy_it ){
			dial->destroy_it= 0;
			destroy= True;
		}
		if( destroy ){
		  int i;
		  xtb_data info;
			for( i= 0; i< LAST_F; i++ ){
				if( ho_af[i].destroy ){
					(*(ho_af[i].destroy))( ho_af[i].win, &info );
					ho_af[i].win= 0;
					xfree( ho_af[i].framelist );
				}
			}
			xtb_unregister( dial->win, NULL );
			XDestroyWindow( disp, dial->win );
			dial->win= 0;
			dial->frames= 0;
			dial->framelist= NULL;
			xfree( HO_info );
		}
		else{
			XUnmapWindow( disp, dial->win );
		}
		while( XEventsQueued( disp, QueuedAfterFlush)> 0){
			XNextEvent( disp, &evt );
		}
		if( !Exitting && theWin_Info ){
			theWin_Info->HO_Dialog= NULL;
			if( destroy_it ){
				if( theWin_Info->delete_it!= -1 ){
					theWin_Info->delete_it= 1;
					theWin_Info->redraw= 1;
					redraw_fun( HO_redrawbtn.win, 0, NULL);
				}
			}
			else if( theWin_Info->redraw ){
				theWin_Info->halt= 0;
				theWin_Info->draw_count= 0;
				RedrawNow( theWin_Info );
			}
		}
		dial->mapped= 0;
		theWin_Info= NULL;
		thePrintWindow= 0;
		if( thePrintWin_Info ){
			thePrintWin_Info->HO_Dialog= NULL;
			thePrintWin_Info= NULL;
		}
		XSync( disp, False);
		  /* If user pushed the Settings button, we call the main routine
		   \ for firing up the Settings Dialog. For that purpose, we saved
		   \ a local copy of theWin_Info.
		   */
		if( do_settings && win_info && win_info->window ){
		  extern int DoSettings();
			win_info->pw_placing= PW_CENTRE_ON;
			DoSettings( win_info->window, win_info);
		}
	}
}

int ho_dialog( Window theWindow, LocalWin *win_info, char *prog, xtb_data cookie, char *title, char *in_title)
/*
 * Asks the user about hardcopy devices.  A table of hardcopy
 * device names and function pointers to their initialization
 * functions is assumed to be in the global `hard_devices' and
 * `hard_count'.  Returns a non-zero status if the dialog was
 * sucessfully posted.  Calls do_hardcopy in xgraph to actually
 * output information.
 */
{ static Window dummy;
	static int level= 0;
	Window parent= win_info->window, root_win, win_win;
    XWindowAttributes winInfo;
    XSizeHints hints;
	int win_x, win_y, mask, reexpose= False;
	extern int print_immediate;
	extern double *do_gsTextWidth_Batch;
	extern Window ascanf_window;

	do_settings= 0;

	if( level || !setNumber ){
		return(0);
	}
	level++;

	theWin_Info= win_info;
	win_info->HO_Dialog= &HO_Dialog;
	ascanf_window= win_info->window;

	if( !Odisp ){
		Odisp= XGstrdup( "To Device" );
	}
    if( !HO_Dialog.win) {
		make_HO_dialog(RootWindow(disp, screen), parent, prog, cookie,
				&HO_okbtn, &HO_canbtn, &HO_redrawbtn, &HO_Dialog, title, in_title
		);
		(void) xtb_bt_get(HO_okbtn.win, (xtb_data *) &ok_info);
		(void) xtb_bt_get(HO_canbtn.win, (xtb_data *) &cinfo);
		(void) xtb_bt_get(HO_redrawbtn.win, (xtb_data *) &rinfo);
    }
	else{
	  int old= -1, i;
	  /* Change the button information */
		if( !title)
			XStoreName(disp, HO_Dialog.win, "Hardcopy Dialog");
		else
			XStoreName(disp, HO_Dialog.win, title);
		xtb_to_set( AF(TITLE_F).win, in_title);
		{ char buf[1024], *c= getcwd( buf, 1024);
			if( !c ){
				c= serror();
			}
			xtb_ti_set( AF(DIR_F).win, c, NULL);
		}
		device_nr = -1;
		for (i = 0;  i < hard_count;  i++) {
			if (strcmp(Odevice, theWin_Info->hard_devices[i].dev_name) == 0) {
				device_nr = i;
			}
		}
		if( device_nr>= 0 ){
			xtb_br_set( AF(ODEVROW_F).win, device_nr );
			HO_dev_fun( AF(ODEVROW_F).win, old, device_nr, HO_d_info );
		}
		fod_spot = -1;
		for (i = 0;  i < fod_num;  i++) {
			if (strcmp(Odisp, fodstrs[i]) == 0) {
				fod_spot = i;
			}
		}
		if( fod_spot>= 0 ){
			xtb_br_set( AF(DISPROW_F).win, fod_spot );
			fd_fun( AF(DISPROW_F).win, old, fod_spot, HO_d_info );
		}
		(void) xtb_bt_get(HO_okbtn.win, (xtb_data *) &ok_info);
		(void) xtb_bt_get(HO_canbtn.win, (xtb_data *) &cinfo);
		(void) xtb_bt_get(HO_redrawbtn.win, (xtb_data *) &rinfo);
		ok_info->prog = prog;
		ok_info->cookie = cookie;
		XRaiseWindow( disp, HO_Dialog.win);
		reexpose= True;
    }
	HO_Dialog.parent= parent;

	xtb_bt_set( AF(UXLF_F).win, use_X11Font_length, NULL);
	xtb_bt_set( AF(UGSTW_F).win, use_gsTextWidth, NULL);
	xtb_bt_set( AF(SPAX_F).win, scale_plot_area_x, NULL);
	xtb_bt_set( AF(SPAY_F).win, scale_plot_area_y, NULL);
	xtb_bt_set( AF(PSPC_F).win, PS_PrintComment, NULL);
	xtb_bt_set( AF(SPSS_F).win, Sort_Sheet, NULL);
	xtb_bt_set( AF(XGSB_F).win, XG_SaveBounds, NULL);
	xtb_bt_set( AF(XGDAV_F).win, theWin_Info->dump_average_values, NULL);
	xtb_bt_set( AF(XGDPR_F).win, theWin_Info->DumpProcessed, NULL);
	xtb_bt_set( AF(XGDPFRESH_F).win, DProcFresh, NULL);
	xtb_bt_set( AF(XGSLDSC_F).win, splits_disconnect, NULL);
	xtb_bt_set2( AF(XGINI_F).win, Init_XG_Dump, XG_Really_Incomplete, NULL);
	xtb_bt_set( AF(XGDBIN_F).win, theWin_Info->DumpBinary, NULL);
	xtb_bt_set( AF(XGDASC_F).win, theWin_Info->DumpAsAscanf, NULL);
	xtb_bt_set( AF(XGDHEX_F).win, DumpDHex, NULL);
	xtb_bt_set( AF(XGDPEN_F).win, DumpPens, NULL);
	xtb_bt_set( AF(PSMP_F).win, ps_mpage, NULL);
	xtb_bt_set( AF(PSSM_F).win, ps_show_margins, NULL);
	xtb_bt_set( AF(XWR_F).win, (theWin_Info->dev_info.resized==1), NULL);
	xtb_ti_set( AF(PSSC_F).win, d2str( theWin_Info->ps_scale, NULL, NULL), NULL);
	xtb_br_set( AF(PSPOSX_F).win, theWin_Info->ps_xpos);
/* 	xtb_ti_set( AF(PS_LOFF_F).win, d2str( theWin_Info->ps_l_offset, NULL, NULL), NULL);	*/
	xtb_br_set( AF(PSPOSY_F).win, theWin_Info->ps_ypos);
/* 	xtb_ti_set( AF(PS_BOFF_F).win, d2str( theWin_Info->ps_b_offset, NULL, NULL), NULL);	*/
	xtb_br_set( AF(PSORIENT_F).win, theWin_Info->print_orientation);
	xtb_bt_set( AF(PSSP_F).win, showpage, NULL);
	xtb_bt_set( AF(PSEPS_F).win, psEPS, NULL);
	xtb_bt_set( AF(PSDSC_F).win, psDSC, NULL);
	xtb_bt_set( AF(PSSetPage_F).win, psSetPage, NULL);
	xtb_bt_set( AF(DONE_F).win, theWin_Info->printed, NULL);
	xtb_bt_set( AF(PRESTIME_F).win, XG_preserve_filetime, NULL);

    XGetWindowAttributes(disp, parent, &winInfo);
	if( !HO_Dialog.mapped ){
		switch( win_info->pw_placing ){
			case PW_PARENT:
			default:
				XTranslateCoordinates( disp, parent, RootWindow(disp, screen),
						  0, 0, &winInfo.x, &winInfo.y, &dummy
				);
				pw_centre_on_X = winInfo.x + winInfo.width/2;
				pw_centre_on_Y = winInfo.y + winInfo.height/2;
				break;
			case PW_MOUSE:
			  /* dialog comes up with filename box under mouse:	*/
				XQueryPointer( disp, HO_Dialog.win, &root_win, &win_win,
					&pw_centre_on_X, &pw_centre_on_Y, &win_x, &win_y, &mask
				);
				break;
			case PW_CENTRE_ON:
				break;
		}
		hints.x = pw_centre_on_X - HO_Dialog.width/2;
		hints.y = pw_centre_on_Y - HO_Dialog.height/2;
		CLIP( hints.x, 0, DisplayWidth(disp, screen) - HO_Dialog.width );
		CLIP( hints.y, 0, DisplayHeight(disp, screen) - HO_Dialog.height );
		XMoveWindow( disp, HO_Dialog.win, (int) hints.x, (int) hints.y);
		hints.flags = USPosition;
		XSetNormalHints(disp, HO_Dialog.win, &hints);
		XRaiseWindow(disp, HO_Dialog.win);
		XMapWindow(disp, HO_Dialog.win);
	}
	HO_Dialog.mapped= 1;

		if( use_gsTextWidth && *do_gsTextWidth_Batch ){
		  TextRelated textrel= theWin_Info->textrel;
			theWin_Info->textrel.gs_batch= True;
			HO_ok_fun( HO_okbtn.win, 0, ok_info);
			theWin_Info->textrel= textrel;
		}

	if( print_immediate< 0 ){
		HO_d_info->printOK= (print_immediate==-1)? -1 : 1;
	}
	else{
		  /* RJB: move the pointer so as to activate the first
		   * text_box.
		   */
		XWarpPointer( disp, HO_Dialog.win, get_text_box(0)->win, 0,0,0,0,
			(int) get_text_box(0)->width/2, (int) get_text_box(0)->height/2
		);
		  /* reset the destroy_it flag...	*/
		destroy_it= 0;
		if( reexpose ){
		  XEvent evt;
			evt.type= Expose;
			evt.xexpose.display= disp;
			evt.xexpose.x= 0;
			evt.xexpose.y= 0;
			evt.xexpose.width= HO_Dialog.width;
			evt.xexpose.height= HO_Dialog.height;
			evt.xexpose.window= HO_Dialog.win;
			evt.xexpose.count= 0;
			XSendEvent( disp, HO_Dialog.win, 0, ExposureMask, &evt);
		}
	}
	print_immediate= 0;
	level--;
	{ XEvent evt;
	  int handled= 0;
	  xtb_hret xtb_return;
	  int Level= level;
		XNextEvent( disp, &evt);
		if( Handle_HO_Event( &evt, &handled, &xtb_return, &Level, 0) &&
			thePrintWin_Info
		){
			CloseHO_Dialog( thePrintWin_Info->HO_Dialog );
		}
	}
	return(0);
}


void _do_message(char *mesg, int err)
/*
 * This posts a dialog that contains lines of text and a continue
 * button.  The text may be multiple lines.  The dialog is remade
 * each time.
 */
{  char title[256];
   extern char *today();
   extern LocalWin *ActiveWin;
	Boing( 5);
	if( err){
		if( disp== 0 ){
		  static char Mesg[MAXCHBUF];
		  char c= 0;
			if( strlen(mesg)> MAXCHBUF-64 ){
				c= mesg[MAXCHBUF-65];
				mesg[MAXCHBUF-65]= '\0';
			}
#if defined(__CYGWIN__) || defined(linux) || defined(__APPLE__)
			sprintf( Mesg, "%s %s", mesg, strerror(errno) );
#else
			sprintf( Mesg, "%s %s", mesg, (err> 0 && err< sys_nerr)? sys_errlist[err] : "" );
#endif
			STRINGCHECK( Mesg, sizeof(Mesg));
			if( c ){
				mesg[MAXCHBUF-65]= c;
			}
			fputs( Mesg, StdErr );
		}
		else{
			sprintf( title, "XGraph Error (%s)", today() );
			STRINGCHECK( title, sizeof(title) );
			XG_error_box( &ActiveWin, title, mesg, NULL );
		}
	}
	else{
		if( disp== 0){
			fputs( mesg, StdErr);
		}
		else{
			sprintf( title, "XGraph Message (%s)", today() );
			STRINGCHECK( title, sizeof(title) );
			XG_error_box( &ActiveWin, title, mesg, NULL );
		}
	}
}

void do_error(err)
char *err;
{
	_do_message( err, (errno)? errno : -1 );
}

void do_message( mesg)
char *mesg;
{
	_do_message( mesg, 0);
}


void set_HO_printit_win()
{
	if( HO_printit_win && theWin_Info ){
		xtb_bt_set( HO_printit_win, theWin_Info->printed, NULL);
	}
}

xtb_hret SimpleFileDialog_h( Window win, int bval, xtb_data info )
{
	return( SimpleFileDialog( xtb_input_dialog_inputfield, win, bval, info, True ) );
}

xtb_hret SimpleFileDialog2_h( Window win, int bval, xtb_data info )
{
	return( SimpleFileDialog( (Window) info, win, bval, info, True ) );
}

