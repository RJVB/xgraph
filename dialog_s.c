/*
vim:ts=4:sw=4:
 * Xgraph Dialog Boxes
 *
 * Dialog for changing the settings.
 */

/* define SLIDER for a demo of a vertical sliderule widget	*/
/* #define SLIDER	*/

#include "config.h"
IDENTIFY( "Settings Dialog code" );

#include <stdio.h>
#include <math.h>

#include "xgout.h"
#include "xgraph.h"
#include "ascanf.h"
#include "hard_devices.h"
#include "xtb/xtb.h"
#include "new_ps.h"
#include <X11/Xutil.h>
#include <X11/keysym.h>
#include <X11/cursorfont.h>
#include <X11/Xutil.h>

#include <strings.h>
#ifndef STRING
#	define STRING(name)	STR(name)
#endif

#ifndef MIN
#	define MIN(a,b)	(((a)>(b))?(b):(a))
#endif
#ifndef MAX
#	define MAX(a,b)	(((a)<(b))?(b):(a))
#endif
#ifndef CLIP
#	define CLIP(var,low,high)	if((var)<(low)){\
		(var)=(low);\
	}else if((var)>(high)){\
		(var)=(high);}
#	define CLIP_EXPR(var,expr,low,high)	if(((var)=(expr))<(low)){\
		(var)=(low);\
	}else if((var)>(high)){\
		(var)=(high);}
#endif
#define ABS(x)		(((x)<0)?-(x):(x))

#include "NaN.h"
#include <float.h>
#include "mathdef.h"

extern int errno;

#ifdef DEBUG
int *the_errno= &errno;
#endif

#include "fdecl.h"

#include "copyright.h"

int SD_strlen(const char *s)
{
	return((s)? strlen(s) : 0);
}

void do_error();

extern void make_err_box();
extern void del_err_box();

int dont_set_changeables= 0;

int WdebugFlag;
extern int debugFlag, debugLevel, HardSynchro, Synchro_State;
int dataSynchro_State;
extern int data_silent_process;

extern Window ascanf_window;

#include "xfree.h"

extern Pixmap dotMap ;
static LocalWin *theWin_Info;
extern LocalWin *ActiveWin;

extern double *param_scratch;
extern int param_scratch_len;

#define D_VPAD	3
#define D_HPAD	2
#define D_INT	2
#define D_BRDR	2
#define D_INP	50
#define D_DSP	10
#define D_FS	10
#define D_SN	5

typedef struct d_info {
    char *prog;			/* Program name              */
    xtb_data cookie;
	Window errb;
	Window sn_number;
	Window sn_lineWidth, sn_linestyle, sn_draw_set, sn_reverse_draw_set, sn_show_legend,
		sn_no_legend, sn_no_title;
	Window sn_legend;
} d_info;

#define	TAB			'\t'
#define	BACKSPACE	0010
#define DELETE		0177
#define	CONTROL_P	0x1b	/* actually ESC	*/
#define CONTROL_U	0025
#define CONTROL_W	0027
#define CONTROL_X	0030

char word_sep[]= ";:,./ \t-_][(){}";

/* Indices for frames made in make_SD_dialog */
enum SD_frames_defn {
    TITLE_F,
	WDEBUG_F, DEBUG_F, DEBUGL_F, SYNC_F, DSYNC_F, FRACP_F, D3STRF_F, DSILENT_F,
	SNERRB_F, OERRBROW_F, SUAE_F, SNNLB_F,
	SNLBL_F,
	SNNRLBL_F, SNNRSL_F, SNNR_F, SNNRM_F, SDDR_F,
	SNLWLBL_F, SNLW_F, SNLS_F, SNSMSz_F, SNSMS_F, SNMSLBL_F, SNELWLBL_F, SNELW_F, SNELS_F, SNEBW_F,
	SNBPLBL_F, SNBPB_F, SNBPW_F, SNBPT_F,
	SNPLILBL_F, SNPLI_F, SNAdI_F,
		SNCOLS_F,
		SNXC_F, SNYC_F, SNEC_F, SNVC_F,
		SNNC_F,
		SNFNS_F, SNFN_F, SNINF_F,
		SNNL_F, SNBF_F, SNMF_F, SNPM_F,
		SNVMF_F, SNVFM_F, SNVRM_F,
		ADH_F, RFC_F, RFRS_F, SNDS_F, SNMS_F, SNHL_F, SNRDS_F, SNDAS_F, SNSL_F, SNSLLS_F, SNLP_F, SNILP_F, SNXP_F, SNYP_F, SNYV_F,
		SNOUL_F, SNL_F, SNIL_F, SOL_F, SNNP_F, SNT_F, SNOL_F, SNSSU_F, SNSSD_F,
		SNSF_F, SNSLL_F, SNAXF_F, SOAG_F, SNBBF_F, SNHTKF_F, SNVTKF_F, SNZLF_F, SNSE_F, OWM_F, SNSRAWD_F, SNFLT_F, SNSPADD_F,
		SNSSARROW_F, SNSEARROW_F, SNSARRORN_F, SNEARRORN_F,
	DPNAS_F, SNpLEG_F, SNLEG_F, SNLEGEDIT_F, SNATA_F, SNATP_F, SNATR_F, SNATD_F, SNATM_F, SNATH_F, SNATN_F, SNATS_F, SNLEGFUN_F,
	AXMIDIG_F, AXSTR_F, BOUNDS_F, XFIT_F, YFIT_F, PBFIT_F, FITAD_F, ASPECT_F, XSYMM_F, YSYMM_F,
	SNLAV_F, SNEXA_F, SNEYA_F, SNVCXA_F, SNVCXG_F, SNVCXL_F, SNVCYA_F, SNSAVCI_F, SNVCIA_F, XMIN_F, YMIN_F,
/* 	BOUNDSSEP_F,	*/
	NOBB_F,
	XMAX_F, YMAX_F,
	USER_F, SWAPPURE_F, PROCB_F, RAWD_F, TRAX_F,
	SNIFLBL_F, SNXIF_F, SNYIF_F,
	SNS2LBL_F, SNS2X_F, SNS2Y_F,
	SNBTLBL_F, SNBTX_F, SNBTY_F,
	SNLZLBL_F, SNLZX_F, SNLZXV_F, SNLZXMI_F, SNLZXMA_F, SNLZXS_F, SNLZY_F, SNLZYV_F, SNLZYMI_F, SNLZYMA_F, SNLZYS_F,
	LBLX1_F, LBLY1_F, LBLX2_F, LBLY2_F, LSFN_F, LBLSL_F, VTTYPE_F, LBLVL_F,
	SNTAYF_F,
	SNTPF_F, SNTPB_F, SNTPO_F,
	SNTLX_F, SNTLY_F, /* SNTLX2_F, SNTLY2_F,	*/ SNTSX_F, SNTSY_F, SNTPX_F, SNTPY_F,
	KEY1_F, KEY2_F, /* KEY3_F, KEY4_F,	*/
#ifdef SLIDER
	SLIDE_F, SLIDE2_F,
#endif
	TABSELECT_F, VBAR_F, TBAR_F,
    OK_F, REDRAW_F, AUTOREDRAW_F, UPDATE_F, PHIST_F, PARGS_F, HELP_F, VARIABLES_F,
/* 	PRINT_F,	*/
	QUIT_F, BAR_F, LAST_F
} SD_frames;

int autoredraw= 0;

/* RJB: array of text_boxes that can be activated sequentially by hitting a
 * tab
 */
static int text_box[]= {
	XMIN_F, YMIN_F, XMAX_F, YMAX_F,
	SNXIF_F, SNYIF_F,
	SNLZXV_F, SNLZXS_F, SNLZYV_F, SNLZYS_F,
	SNNR_F, SNLW_F, SNLS_F, SNSMSz_F, SNSMS_F, SNELW_F, SNELS_F, SNEBW_F, SNPLI_F, SNFN_F, SNAdI_F, SNFN_F,
	SNLEG_F,
	LBLX1_F, LBLY1_F, LBLX2_F, LBLY2_F, LBLSL_F, LBLVL_F,
	SNXC_F, SNYC_F, SNEC_F, SNVC_F, SNFN_F
};
static int text_boxes= sizeof(text_box)/sizeof(int);

#define LEGENDFUNCTIONS	8
char *LegendFTypes[LEGENDFUNCTIONS] = { "Legnd", "Title", "Proc", "Assoc", "Colour", "HL_Colr", "Descr", "Labls" };

#define showTitle	LegendFunctionNR;
#define MARKERSIZEDESCR	" Marker size or scale in Msze mode:\n Negative for axes-linked, scaling size;\n NaN for global setting (-PSm=%g)\n%cggggggggggggggggggggggggggggggggggggggggggggggg)\n"

int data_sn_number= 0, data_sn_linestyle, data_sn_markstyle, data_sn_markSize,
	data_sn_elinestyle, data_sn_plot_interval, data_sn_adorn_interval, legend_changed= 0,
	LegendFunctionNr= 0, do_update_SD_size= 0, pLegFunNr= 1;
double data_sn_lineWidth, data_sn_elineWidth;
double data_start_arrow_orn, data_end_arrow_orn;
char data_sn_number_buf[D_SN], data_sn_lineWidth_buf[D_SN], data_sn_linestyle_buf[D_SN], data_sn_markstyle_buf[D_SN],
	data_sn_elineWidth_buf[D_SN], data_sn_plot_interval_buf[D_SN], data_sn_adorn_interval_buf[D_SN],
	data_sn_elinestyle_buf[D_SN], *data_legend_buf, powXFlag_buf[D_SN], powYFlag_buf[D_SN],
	data_start_arrow_orn_buf[D_INP], data_end_arrow_orn_buf[D_INP],
	fileName_buf[LEGEND];
int CleanSheet= 0, dlb_len= LEGEND;
int SD_LMaxBufSize= MAXBUFSIZE;
extern int AlwaysDrawHighlighted, Raw_NewSets,
	barBase_set, barWidth_set, barType,
	ebarWidth_set;
int error_type2num[ERROR_TYPES]= { 0, 1, 2, 4, INTENSE_FLAG, MSIZE_FLAG, EREGION_FLAG, 3 };
int error_num2type[ERROR_TYPES]= { 0, 1, 2, EREGION_FLAG, 3, 4, INTENSE_FLAG, MSIZE_FLAG };

extern char *XGstrdup(), *cleanup();

extern char *titleText2;
extern int no_pens, no_ulabels, no_legend, no_legend_box, no_intensity_legend, intensity_legend_placed,
	legend_always_visible, use_average_error, no_title, logXFlag, logYFlag, sqrtXFlag, sqrtYFlag, process_bounds,
	transform_axes, polarFlag, absYFlag, overwrite_legend, overwrite_AxGrid, xcol, ycol, ecol, lcol, Ncol, error_type, no_errors,
	exact_X_axis, exact_Y_axis, ValCat_X_axis, ValCat_Y_axis, ValCat_X_levels, ValCat_X_grid, show_all_ValCat_I, ValCat_I_axis;
static int raw_display= 1, set_raw_display, points_added, set_sarrow, set_earrow;
extern double powXFlag, powYFlag, radix, radix_offset;
extern double powAFlag;
extern double Xincr_factor, Yincr_factor, Xscale2, Yscale2, Xbias_thres, Ybias_thres;
int logXFlag2, logYFlag2;

#define AF(ix)	sd_af[(int) (ix)]
#define aF(ix)	&AF(ix)
xtb_frame AF(LAST_F);
int sd_last_f= LAST_F;
xtb_frame SD_Dialog = { (Window) 0, 0, 0, 0, 0 };
struct d_info *SD_info;
static xtb_frame okbtn, redrawbtn;
static d_info *ok_info, *rinfo;

char d3str_format[16]= "%.20g";
int d3str_format_changed= 0;

char SD_ascanf_separator_buf[2];
extern char ascanf_separator;

extern char *parse_codes(char *);
extern char log_zero_sym_x[MAXBUFSIZE], log_zero_sym_y[MAXBUFSIZE];	/* symbol to indicate log_zero	*/
extern int lz_sym_x, lz_sym_y;	/* log_zero symbols set?	*/
extern int log_zero_x_mFlag, log_zero_y_mFlag;
extern double log10_zero_x, log10_zero_y, 
	log_zero_x, log_zero_y,	/* substitute 0.0 for these values when using log axis	*/
	_log_zero_x, _log_zero_y;	/* transformed values	*/
extern char *AxisValueFormat;
extern int AxisValueMinDigits;

int apply_to_all= 0, apply_to_prev= 0, apply_to_rest= 0, apply_to_marked= 0, apply_to_hlt= 0, apply_to_new= 0, apply_to_drawn= 0,
	_apply_to_marked, _apply_to_hlt, _apply_to_new, _apply_to_drawn, apply_to_src= 0, _apply_to_src;

static xtb_hret SD_redraw_fun();

static xtb_hret SD_cycle_focus_button( int firstchar )
{  static xtb_frame *current= NULL, *brcurrent= NULL;
   static int currentN= 0, brcurrentN= -1;
   xtb_frame *hit;
	if( (hit= xtb_find_next_named_button( sd_af, LAST_F, &current, &currentN, &brcurrent, &brcurrentN, NULL, firstchar )) ){
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
	dont_set_changeables= 1;
	return( XTB_HANDLED);
}

#define FRTEST(w,fr)	( (w)==AF(fr).win && AF(fr).mapped )

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

static xtb_hret goto_next_text_box( Window *win)
{  int i, hit= 0;
#ifdef TAB_WARPS
   int x_loc, y_loc;
#endif
	if( CheckMask( xtb_modifier_state, ControlMask) ){
		if( FRTEST(*win, SNLEG_F) ){
			hit= SNNR_F;
		}
		else if( AF(SNLEG_F).mapped ){
			hit= SNLEG_F;
		}
	}
	else{
		if( FRTEST(*win, SNLW_F) ){
			hit= (data_sn_number>=0 && data_sn_number< setNumber)? SNLS_F : SNLEG_F;
		}
		else if( FRTEST(*win, SNLS_F) ){
			hit= (data_sn_number>=0 && data_sn_number< setNumber)? SNSMSz_F : SNLEG_F;
		}
		else if( FRTEST(*win, SNSMSz_F) ){
			hit= (data_sn_number>=0 && data_sn_number< setNumber)? SNSMS_F : SNLEG_F;
		}
		else if( FRTEST(*win, SNSMS_F) ){
			hit= (data_sn_number>=0 && data_sn_number< setNumber)? SNELW_F : SNLEG_F;
		}
		else if( FRTEST(*win, SNELW_F) ){
			hit= (data_sn_number>=0 && data_sn_number< setNumber)? SNELS_F : SNEBW_F;
		}
		else if( FRTEST(*win, SNELW_F) ){
			hit= (data_sn_number>=0 && data_sn_number< setNumber)? SNEBW_F : SNLEG_F;
		}
		else if( FRTEST(*win, SNELS_F) ){
			hit= (data_sn_number>=0 && data_sn_number< setNumber)? SNPLI_F : SNLEG_F;
		}
		else if( FRTEST(*win, SNPLI_F) ){
			hit= SNAdI_F;
		}
		else if( FRTEST(*win, SNAdI_F) ){
			hit= SNLEG_F;
		}
		else if( FRTEST(*win, SNFNS_F) || FRTEST(*win, SNFN_F) ){
			hit= SNLEG_F;
		}
		else if( (FRTEST(*win, SNNR_F) || FRTEST(*win, SNNRSL_F)) && (data_sn_number< 0 || data_sn_number>= setNumber) ){
			hit= SNLEG_F;
		}
	}
	if( hit && AF(hit).mapped ){
		*win= AF(hit).win;
#ifdef TAB_WARPS
		x_loc= AF(hit).width/2;
		y_loc= AF(hit).height/2;
		XWarpPointer( disp, None, *win, 0, 0, 0, 0, x_loc, y_loc);
#else
		XSetInputFocus( disp, *win, RevertToParent, CurrentTime);
#endif
	}
	else{
		for( i= 0, hit= 0; !hit && i< text_boxes; i++){
			if( get_text_box(i)->win== *win && get_text_box(i)->mapped ){
				hit= i+1;
			}
		}
		{
			if( hit== text_boxes){
				hit= 0;
				while( hit< text_boxes && !get_text_box(i)->mapped ){
					hit+= 1;
				}
			}
			if( hit< text_boxes && get_text_box(hit)->mapped ){
				*win= get_text_box(hit)->win;
#ifdef TAB_WARPS
				x_loc= get_text_box(hit)->width/2;
				y_loc= get_text_box(hit)->height/2;
				XWarpPointer( disp, None, *win, 0, 0, 0, 0, x_loc, y_loc);
#else
				XSetInputFocus( disp, *win, RevertToParent, CurrentTime);
#endif
			}
		}
	}
	dont_set_changeables= 1;
	return( XTB_HANDLED);
}

static xtb_hret goto_prev_text_box( Window win)
{  int i, hit;
#ifdef TAB_WARPS
   int x_loc, y_loc;
#endif
	for( i= 0, hit= 0; !hit && i< text_boxes; i++){
		if( get_text_box(i)->win== win && get_text_box(i)->mapped ){
			hit= i+1;
		}
	}
	if( hit){
		hit-= 2;
		if( hit== -1){
			hit= text_boxes-1;
			while( hit> 0 && !get_text_box(hit)->mapped ){
				hit-= 1;
			}
		}
		if( hit>= 0 && get_text_box(hit)->mapped ){
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
	else
		Boing(5);
	return( XTB_HANDLED);
}

/*ARGSUSED*/
static xtb_hret SD_df_fun( Window win, int ch, char *text, xtb_data val)
// Window win;			/* Widget window   */
// int ch;				/* Typed character */
// char *text;			/* Copy of text    */
// xtb_data val;			/* User info       */
{ char Text[MAXCHBUF];
  int changed= 0, accept;

	if( !theWin_Info ){
		return( XTB_STOP );
	}

    if ((ch == BACKSPACE) || (ch == DELETE)) {
		if( !( (ch== DELETE)? xtb_ti_dch_right(win) : xtb_ti_dch(win)) )
			Boing(5);
		return( SD_df_fun( win, 0, text, val) );
    }
	else if ((ch == CONTROL_U) || (ch == CONTROL_X)) {
		(void) xtb_ti_set(win, "", (xtb_data) 0);
		return( SD_df_fun( win, 0, text, val) );
    }
	else if( ((ch== TAB || ch== XK_Tab) && !xtb_clipboard_copy) ){
		// 20101113: added the missing text argument!!
		SD_df_fun( win, 0, text, val );
		return( (CheckMask(xtb_modifier_state, ShiftMask))? goto_prev_text_box(win) : goto_next_text_box( &win) );
	}
	else if( ch== CONTROL_P){
		// 20101113: added the missing text argument!!
		SD_df_fun( win, 0, text, val );
		return( goto_prev_text_box( win) );
	}
	else if( ch== CONTROL_W){
	  char *str;
		if( *text)
			str= &text[ strlen(text)-1 ];
		else{
			Boing( 5);
			return( SD_df_fun( win, 0, text, val) );
		}
		if( index( word_sep, *str) ){
			if( !xtb_ti_dch(win) ){
				Boing( 5);
			}
			return( SD_df_fun( win, 0, text, val) );
		}
		while( *str && !index( word_sep, *str) ){
			if( !xtb_ti_dch(win) ){
				Boing( 5);
				return( SD_df_fun( win, 0, text, val) );
			}
			str--;
		}
	}
	else if( ch && ch!= XK_Meta_L && ch!= XK_Meta_R && ch!= XK_Alt_L && ch!= XK_Alt_R &&
		ch!= XK_Super_L && ch!= XK_Super_R && ch!= XK_Hyper_L && ch!= XK_Hyper_R &&
		ch!= XK_Down && ch!= XK_Up
	){
	  /* Insert if printable - ascii dependent */
		if( /* (ch < ' ') || (ch >= DELETE) || */ !xtb_ti_ins(win, ch)) {
			Boing( 5);
		}
    }
	xtb_ti_get( win, Text, (xtb_data) NULL );
	  /* 981111: accepting ch==0 as "accept" causes accept when e.g. a character is deleted!	*/
	if( (accept= (/* ch== 0 || */ ch== XK_Down || ch== XK_Up || ch== 0x12 )) ){
		xtb_ti_set( win, Text, (xtb_data) NULL );
		if( val== log_zero_sym_x ){
			if( strcmp( theWin_Info->log_zero_sym_x, Text) ){
				changed+= 1;
				strcpy( theWin_Info->log_zero_sym_x, Text);
				theWin_Info->lz_sym_x= 1;
			}
			if( !SD_strlen(theWin_Info->log_zero_sym_x) ){
				theWin_Info->lz_sym_x= 0;
			}
			xtb_bt_set( AF(SNLZX_F).win, theWin_Info->lz_sym_x, NULL);
		}
		else if( val== log_zero_sym_y ){
			if( strcmp( theWin_Info->log_zero_sym_y, Text) ){
				changed+= 1;
				strcpy( theWin_Info->log_zero_sym_y, Text);
				theWin_Info->lz_sym_y= 1;
			}
			if( !SD_strlen(theWin_Info->log_zero_sym_y) ){
				theWin_Info->lz_sym_y= 0;
			}
			xtb_bt_set( AF(SNLZY_F).win, theWin_Info->lz_sym_y, NULL);
		}
		else if( val== d3str_format ){
		  extern ascanf_Function *ascanf_d3str_format;
			if( strcmp( d3str_format, Text) ){
				strcpy( d3str_format, Text);
				d3str_format_changed+= 1;
				set_changeables(0,True);
				xfree(ascanf_d3str_format->usage);
				ascanf_d3str_format->usage= strdup(d3str_format);
			}
		}
		else if( val== SD_ascanf_separator_buf ){
			SD_ascanf_separator_buf[0]= Text[0];
		}
		else if( val== &AxisValueFormat ){
			if( strlen(Text) ){
				if( XGstrcmp( AxisValueFormat, Text) ){
					xfree( AxisValueFormat );
					AxisValueFormat= strdup( Text );
					changed+= 1;
				}
			}
			else{
				xfree( AxisValueFormat );
			}
		}
	}
	if( changed ){
		theWin_Info->redraw= 1;
		if( theWin_Info->redrawn== -3 ){
			theWin_Info->redrawn= 0;
		}
		theWin_Info->printed= 0;
	}
    return XTB_HANDLED;
}

extern int Isdigit( unsigned int ch );

extern double dcmp( double b, double a, double prec);

UserLabel *GetULabelNr( int nr)
{  UserLabel *ret= NULL;
	if( nr>= 0 && nr< theWin_Info->ulabels ){
	  UserLabel *ul= theWin_Info->ulabel;
		while( ul && nr>= 0 ){
			nr-= 1;
			ret= ul;
			ul= ul->next;
		}
	}
	return( ret );
}

char *GetLabelNr( int nr)
{  UserLabel *ret= GetULabelNr( nr);
	if( ret ){
		return( ret->label );
	}
	else{
		return(NULL);
	}
}

static int set_xmin, set_ymin, set_xmax, set_ymax;
static int set_lblx1, set_lbly1;
static int set_lblx2, set_lbly2, set_setlink, fileNumber;
static int fit_x, fit_y, fit_pB, aspect;
extern int fit_after_draw;

extern int XSymmetric, YSymmetric;

extern double vectorLength;

#define DFN_LEN	13,64
char *d3str(char *buf,char *format, double val)
{  int ugi= use_greek_inf;
   char *ret;
/* 	use_greek_inf= 0;	*/
	ret= d2str(val,format,buf);
	use_greek_inf= ugi;
	return( ret );
}

extern double cus_log10X(), cus_log10Y(), Reform_X(), Reform_Y(), Trans_X(), Trans_Y();

static int apply_ok( int i )
{ int ok= 1;
	if( i< 0 || i> setNumber ){
		return( 1 );
	}
	if( _apply_to_drawn ){
		if( !draw_set( theWin_Info, i ) ){
			ok= 0;
		}
		else{
			return( 1 );
		}
	}
	if( _apply_to_marked ){
		if( theWin_Info->mark_set[i]<= 0 ){
			ok= 0;
		}
		else{
			return( 1 );
		}
	}
	if( _apply_to_hlt ){
		if( theWin_Info->legend_line[i].highlight<= 0 ){
			ok= 0;
		}
		else{
			return( 1 );
		}
	}
	if( _apply_to_new ){
		if( AllSets[i].points_added<= 0 ){
			ok= 0;
		}
		else{
			return( 1 );
		}
	}
	if( _apply_to_src ){
		if( AllSets[i].set_linked ){
			ok= 0;
		}
		else{
			return( 1 );
		}
	}
	return( ok );
}

int Parse_vectorPars( char *buffer, DataSet *this_set, int global_copy, int *Changed, char *caller )
{ double pars[2+MAX_VECPARS], *vp;
  int n= 2+MAX_VECPARS, cc, *changed= ((Changed)? Changed : &cc), *vt;
  extern int vectorType;
  extern double vectorPars[MAX_VECPARS];
	fascanf( &n, buffer, pars, NULL, NULL, NULL, NULL);
	if( n> 1 ){
		CLIP_EXPR( vectorType, (int) pars[0], 0, MAX_VECTYPES );
		if( this_set && this_set->vectorType!= vectorType ){
			this_set->vectorType= vectorType;
			*changed+= 1;
		}
		if( this_set ){
			if( this_set->vectorLength!= pars[1] ){
				this_set->vectorLength= pars[1];
				*changed+= 1;
			}
			vt= &(this_set->vectorType);
			vp= this_set->vectorPars;
		}
		else{
			vectorLength= pars[1];
			vt= &vectorType;
			vp= vectorPars;
		}
		if( n> 2 ){
			switch( *vt ){
				case 1:
				case 3:
				case 4:
					if( pars[2]< 0 ){
						pars[2]= -pars[2];
					}
					if( vp[0]!= (pars[2]= (pars[2]< 1)? pars[2] : 1/ pars[2]) ){
						vp[0]= pars[2];
						*changed+= 1;
					}
					break;
				default:
					if( caller ){
						fprintf( StdErr, "%s %s: warning: 3rd and above parameter ignored for this type (%d)\n",
							caller, buffer, *vt
						);
					}
					break;
			}
		}
		if( n> 3 ){
			switch( *vt ){
				case 1:
				case 3:
				case 4:
					if( vp[1]!= pars[3] ){
						vp[1]= pars[3];
						*changed+= 1;
					}
					break;
				default:
					if( caller ){
						fprintf( StdErr, "%s %s: warning: 4th and above parameter ignored for this type (%d)\n",
							caller, buffer, *vt
						);
					}
			}
		}
	}
	if( global_copy && vp!= vectorPars ){
		memcpy( vectorPars, vp, sizeof(vectorPars) );
	}
	return(n);
}

/*ARGSUSED*/
static xtb_hret SD_dfn_fun( Window win, int ch, char *text, xtb_data val)
/*
 * This is the handler function for the text widget for
 * specifing a number.  It supports simple
 * line editing operations.
 */
{ double value;
  char number[MAXCHBUF];
  int changed= 0, accept, all= 0, dsn= 0;
  static char level= 0;

	if( !theWin_Info ){
		return( XTB_STOP );
	}

    if ((ch == BACKSPACE) || (ch == DELETE)) {
		if( !( (ch== DELETE)? xtb_ti_dch_right(win) : xtb_ti_dch(win)) )
			Boing( 5);
		return XTB_HANDLED;
    }
	else if ((ch == CONTROL_U) || (ch == CONTROL_X)) {
		xtb_ti_set(win, "", (xtb_data) 0);
		return XTB_HANDLED;
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
	  int n= 1;
		xtb_ti_get( win, number, (xtb_data) NULL );
		if( !(val== &vectorLength && (data_sn_number>= 0 && data_sn_number< setNumber)) ){
			if( fascanf( &n, number, &value, NULL, NULL, NULL, NULL) ){
				if( CheckMask(xtb_modifier_state, ControlMask|ShiftMask) ){
					value*= (value> 0)? 2.0 : 0.5;
				}
				else if( CheckMask(xtb_modifier_state, ShiftMask) ){
					value*= (value> 0)? sqrt(2.0) : sqrt(0.5);
				}
				else{
					value+= 1.0;
				}
				d3str( number, d3str_format, value);
				xtb_ti_set( win, number, (xtb_data) 0);
			}
			else{
				Boing(1);
			}
		}
		else{
		  /* this is not supported for composite values	*/
			Boing(10);
			ch= 0;
		}
	}
	else if( ch== XK_Down ){
	  int n= 1;
		xtb_ti_get( win, number, (xtb_data) NULL );
		if( !(val== &vectorLength && (data_sn_number>= 0 && data_sn_number< setNumber)) ){
			if( fascanf( &n, number, &value, NULL, NULL, NULL, NULL) ){
				if( CheckMask(xtb_modifier_state, ControlMask|ShiftMask) ){
					value*= (value> 0)? 0.5 : 2.0;
				}
				else if( CheckMask(xtb_modifier_state, ShiftMask) ){
					value*= (value> 0)? sqrt(0.5) : sqrt(2.0);
				}
				else{
					value-= 1.0;
				}
				d3str( number, d3str_format, value);
				xtb_ti_set( win, number, (xtb_data) 0);
			}
			else{
				Boing(1);
			}
		}
		else{
		  /* this is not supported for composite values	*/
			Boing(10);
			ch= 0;
		}
	}
	else if( ch!= ' ' && ch!= TAB &&
		ch!= XK_Meta_L && ch!= XK_Meta_R && ch!= XK_Alt_L && ch!= XK_Alt_R &&
		ch!= XK_Super_L && ch!= XK_Super_R && ch!= XK_Hyper_L && ch!= XK_Hyper_R
	){
	  /* Insert if valid for a number */
		if( !xtb_ti_ins(win, ch) )
		{
			Boing( 5);
		}
    }
	if( (accept= ( ch== ' ' || ch== XK_Down || ch== XK_Up || ch== 0x12 || (ch== TAB && !xtb_clipboard_copy) )) ){
	  int n= 1, nr, N, i, ard= autoredraw;
	  Boolean aaa= False;
		if( !(level && (ch== ' ' || ch== 0x12 || (ch== TAB && !xtb_clipboard_copy))) ){
			xtb_ti_get( win, number, (xtb_data) NULL );
		}
		else if( level ){
			ch= ' ';
			strcpy( number, text);
		}
		level+= 1;
		if( !(val== &vectorLength && (data_sn_number>= 0 && data_sn_number< setNumber)) ){
			if( fascanf( &n, number, &value, NULL, NULL, NULL, NULL)<= 0 ){
				value= 0.0;
			}
			d3str( number, d3str_format, value);
			xtb_ti_set( win, number, (xtb_data) 0);
		}

		if( apply_to_marked || apply_to_hlt || apply_to_drawn || apply_to_new || apply_to_src ){
			_apply_to_drawn= apply_to_drawn;
			apply_to_drawn= 0;
			_apply_to_marked= apply_to_marked;
			apply_to_marked= 0;
			_apply_to_hlt= apply_to_hlt;
			apply_to_hlt= 0;
			_apply_to_new= apply_to_new;
			apply_to_new= 0;
			_apply_to_src= apply_to_src;
			apply_to_src= 0;
			apply_to_all= True;
			aaa= True;
		}
		if( apply_to_all ){
			nr= (apply_to_rest)? data_sn_number : 0;
			N= (apply_to_prev)? MIN(data_sn_number+1,setNumber) : setNumber;
			apply_to_all= 0;
			apply_to_rest= 0;
			apply_to_prev= 0;
			apply_to_src= 0;
			all= 1;
			dsn= data_sn_number;
			xtb_bt_set( AF(SNATA_F).win, 0, NULL );
			xtb_bt_set( AF(SNATP_F).win, 0, NULL );
			xtb_bt_set( AF(SNATR_F).win, 0, NULL );
			autoredraw= False;
		}
		else{
			nr= data_sn_number;
			N= nr+ 1;
		}

		if( val== &log_zero_x ){
		  double rvalue;
			if( value> 0 &&
					!dcmp( (rvalue=
						Reform_X( theWin_Info, cus_log10X(theWin_Info, value), 0 )/ theWin_Info->Xscale)/ value,
						1.0, -100.0
					)
			){
				if( theWin_Info->log_zero_x!= value ){
					theWin_Info->log_zero_x= value;
					theWin_Info->log_zero_x_mFlag= 0;
					changed+= 1;
				}
			}
			else if( !value ){
				if( theWin_Info->log_zero_x!= value ){
					changed+= 1;
					theWin_Info->log_zero_x= value;
					theWin_Info->log_zero_x_mFlag= 0;
					Boing(1);
				}
			}
			else{
				Boing(5);
				Boing(5);
				d3str( number, d3str_format, theWin_Info->log_zero_x);
				xtb_ti_set( win, number, NULL);
			}
		}
		else if( val== &log_zero_y ){
		  double rvalue;
			if( value> 0 &&
					!dcmp( (rvalue=
						Reform_Y( theWin_Info, cus_log10Y(theWin_Info, value), 0 )/ theWin_Info->Yscale)/ value,
						1.0, -100.0
					)
			){
				if( theWin_Info->log_zero_y!= value ){
					changed+= 1;
					theWin_Info->log_zero_y= value;
					theWin_Info->log_zero_y_mFlag= 0;
				}
			}
			else if( !value ){
				if( theWin_Info->log_zero_y!= value ){
					changed+= 1;
					theWin_Info->log_zero_y= value;
					theWin_Info->log_zero_y_mFlag= 0;
					Boing(1);
				}
			}
			else{
				Boing(5);
				Boing(5);
				d3str( number, d3str_format, theWin_Info->log_zero_y);
				xtb_ti_set( win, number, NULL);
			}
		}
		else if( val== &radix ){
			if( !value ){
				Boing(5);
				value= 2* M_PI;
				d3str( number, d3str_format, value);
				xtb_ti_set( win, number, NULL);
			}
			if( theWin_Info->radix!= value ){
			 extern double Gonio_Base();
				theWin_Info->radix= value;
				Gonio_Base( theWin_Info, theWin_Info->radix, theWin_Info->radix_offset);
				sprintf( number, "%lf", theWin_Info->radix );
				xtb_ti_set( win, number, NULL);
				if( theWin_Info->polarFlag ){
					SD_redraw_fun( 0, 0, rinfo);
					  /* 990715: sync false. 990716: restored	*/
					XG_XSync( disp, True );
					if( !theWin_Info ){
						return( XTB_STOP );
					}
				}
				else{
					changed+= 1;
				}
			}
		}
		else if( val== &radix_offset ){
			if( theWin_Info->radix_offset!= value ){
			 extern double Gonio_Base();
				theWin_Info->radix_offset= value;
				Gonio_Base( theWin_Info, theWin_Info->radix, theWin_Info->radix_offset);
				sprintf( number, "%lf", theWin_Info->radix_offset );
				xtb_ti_set( win, number, NULL);
				if( theWin_Info->polarFlag ){
					SD_redraw_fun( 0, 0, rinfo);
					  /* 990715: sync made "false" 990716: restored	*/
					XG_XSync( disp, True );
					if( !theWin_Info ){
						return( XTB_STOP );
					}
				}
				else{
					changed+= 1;
				}
			}
		}
		else if( val== &powAFlag ){
			if( !value ){
				Boing(5);
				value= 1.0;
				d3str( number, d3str_format, value);
				xtb_ti_set( win, number, NULL);
			}
			if( theWin_Info->powAFlag!= value ){
				theWin_Info->powAFlag= value;
				changed+= 1;
			}
		}
		else if( val== &powXFlag ){
			if( !value ){
				Boing(5);
				value= (theWin_Info->sqrtXFlag> 0)? 0.5 : 1.0;
				d3str( number, d3str_format, value);
				xtb_ti_set( win, number, NULL);
			}
			if( theWin_Info->powXFlag!= value ){
				theWin_Info->powXFlag= value;
				changed+= 1;
			}
		}
		else if( val== &powYFlag ){
			if( !value ){
				Boing(5);
				value= (theWin_Info->sqrtYFlag> 0)? 0.5 : 1.0;
				d3str( number, d3str_format, value);
				xtb_ti_set( win, number, NULL);
			}
			if( theWin_Info->powYFlag!= value ){
				theWin_Info->powYFlag= value;
				changed+= 1;
			}
		}
		else if( val== &Xincr_factor ){
		  double *v= (theWin_Info->ValCat_X_axis)? &theWin_Info->ValCat_X_incr : &theWin_Info->Xincr_factor;
			if( value<= 0 ){
				Boing(5);
/* 				value= theWin_Info->Xincr_factor;	*/
/* 				d3str( number, d3str_format, value);	*/
/* 				xtb_ti_set( win, number, NULL);	*/
			}
			if( *v!= value ){
				*v= value;
				theWin_Info->axis_stuff.__XLabelLength= 0;
				theWin_Info->axis_stuff.__YLabelLength= 0;
				theWin_Info->axis_stuff.XLabelLength= 0;
				theWin_Info->axis_stuff.YLabelLength= 0;
				changed+= 1;
			}
		}
		else if( val== &Yincr_factor ){
		  double *v= (theWin_Info->ValCat_Y_axis)? &theWin_Info->ValCat_Y_incr : &theWin_Info->Yincr_factor;
			if( value<= 0 ){
				Boing(5);
/* 				value= theWin_Info->Yincr_factor;	*/
/* 				d3str( number, d3str_format, value);	*/
/* 				xtb_ti_set( win, number, NULL);	*/
			}
			if( *v!= value ){
				*v= value;
				theWin_Info->axis_stuff.__XLabelLength= 0;
				theWin_Info->axis_stuff.__YLabelLength= 0;
				theWin_Info->axis_stuff.XLabelLength= 0;
				theWin_Info->axis_stuff.YLabelLength= 0;
				changed+= 1;
			}
		}
		else if( val== &Xbias_thres ){
			if( theWin_Info->Xbias_thres!= value ){
				theWin_Info->Xbias_thres= value;
				changed+= 1;
			}
		}
		else if( val== &Ybias_thres ){
			if( theWin_Info->Ybias_thres!= value ){
				theWin_Info->Ybias_thres= value;
				changed+= 1;
			}
		}
		else if( val== &Xscale2 ){
			if( !value ){
				Boing(5);
				value= theWin_Info->Xscale;
				d3str( number, d3str_format, value);
				xtb_ti_set( win, number, NULL);
			}
			if( theWin_Info->Xscale!= fabs(value) ){
				theWin_Info->_Xscale= value;
				theWin_Info->Xscale= fabs(value);
				changed+= 1;
			}
		}
		else if( val== &Yscale2 ){
			value= fabs(value);
			if( theWin_Info->Yscale!= value ){
				theWin_Info->Yscale= value;
				changed+= 1;
			}
		}
		else if( val== &ValCat_X_levels ){
			if( theWin_Info->ValCat_X_levels!= (int) value ){
				theWin_Info->ValCat_X_levels= (int) value;
				theWin_Info->axis_stuff.__XLabelLength= 0;
				theWin_Info->axis_stuff.__YLabelLength= 0;
				theWin_Info->axis_stuff.XLabelLength= 0;
				theWin_Info->axis_stuff.YLabelLength= 0;
				changed+= 1;
			}
		}
		else if( val== &AxisValueMinDigits ){
		  int x= (int) value;
			if( x!= AxisValueMinDigits ){
				AxisValueMinDigits= x;
				changed+= 1;
			}
		}
		else if( val== &debugLevel ){
			if( value ){
				debugLevel= (int) value;
			}
			else{
				debugLevel= 0;
			}
		}
		else if( val== &Synchro_State ){
			if( value ){
				Synchro_State= (int) value;
			}
			else{
				Synchro_State= 0;
			}
		}
		else if( val== &set_xmin ){
		  double *v= (theWin_Info->win_geo.nobb_coordinates)? &theWin_Info->win_geo.nobb_loX :
		  		&theWin_Info->win_geo.bounds._loX;
			if( *v!= value ){
				if( theWin_Info->win_geo.nobb_coordinates ){
					if( NaN(value) ){
						theWin_Info->win_geo.nobb_range_X= 0;
					}
					else{
						theWin_Info->win_geo.nobb_range_X= 1;
					}
				}
				*v= value;
				if( !theWin_Info->win_geo.nobb_coordinates && !CheckMask(xtb_modifier_state, Mod1Mask) )
					theWin_Info->win_geo.user_coordinates= 1;
				changed= 2;
				theWin_Info->fit_xbounds= 0;
				theWin_Info->aspect= 0;
				theWin_Info->x_symmetric= 0;
				theWin_Info->y_symmetric= 0;
				xtb_bt_set( AF(XFIT_F).win, 0, NULL);
				xtb_bt_set( AF(PBFIT_F).win, 0, NULL);
				xtb_bt_set( AF(ASPECT_F).win, 0, NULL);
			}
		}
		else if( val== &set_ymin ){
		  double *v= (theWin_Info->win_geo.nobb_coordinates)? &theWin_Info->win_geo.nobb_loY :
		  		&theWin_Info->win_geo.bounds._loY;
			if( *v!= value ){
				if( theWin_Info->win_geo.nobb_coordinates ){
					if( NaN(value) ){
						theWin_Info->win_geo.nobb_range_Y= 0;
					}
					else{
						theWin_Info->win_geo.nobb_range_Y= 1;
					}
				}
				*v= value;
				if( !theWin_Info->win_geo.nobb_coordinates && !CheckMask(xtb_modifier_state, Mod1Mask) )
					theWin_Info->win_geo.user_coordinates= 1;
				changed= 2;
				theWin_Info->fit_ybounds= 0;
				theWin_Info->aspect= 0;
				theWin_Info->x_symmetric= 0;
				theWin_Info->y_symmetric= 0;
				xtb_bt_set( AF(YFIT_F).win, 0, NULL);
				xtb_bt_set( AF(ASPECT_F).win, 0, NULL);
			}
		}
		else if( val== &set_xmax ){
		  double *v= (theWin_Info->win_geo.nobb_coordinates)? &theWin_Info->win_geo.nobb_hiX :
		  		&theWin_Info->win_geo.bounds._hiX;
			if( *v!= value ){
				if( theWin_Info->win_geo.nobb_coordinates ){
					if( NaN(value) ){
						theWin_Info->win_geo.nobb_range_X= 0;
					}
					else{
						theWin_Info->win_geo.nobb_range_X= 1;
					}
				}
				*v= value;
				if( !theWin_Info->win_geo.nobb_coordinates && !CheckMask(xtb_modifier_state, Mod1Mask) )
					theWin_Info->win_geo.user_coordinates= 1;
				changed= 2;
				theWin_Info->fit_xbounds= 0;
				theWin_Info->aspect= 0;
				theWin_Info->x_symmetric= 0;
				theWin_Info->y_symmetric= 0;
				xtb_bt_set( AF(XFIT_F).win, 0, NULL);
				xtb_bt_set( AF(PBFIT_F).win, 0, NULL);
				xtb_bt_set( AF(ASPECT_F).win, 0, NULL);
			}
		}
		else if( val== &set_ymax ){
		  double *v= (theWin_Info->win_geo.nobb_coordinates)? &theWin_Info->win_geo.nobb_hiY :
		  		&theWin_Info->win_geo.bounds._hiY;
			if( *v!= value ){
				if( theWin_Info->win_geo.nobb_coordinates ){
					if( NaN(value) ){
						theWin_Info->win_geo.nobb_range_Y= 0;
					}
					else{
						theWin_Info->win_geo.nobb_range_Y= 1;
					}
				}
				*v= value;
				if( !theWin_Info->win_geo.nobb_coordinates && !CheckMask(xtb_modifier_state, Mod1Mask) )
					theWin_Info->win_geo.user_coordinates= 1;
				changed= 2;
				theWin_Info->fit_ybounds= 0;
				theWin_Info->aspect= 0;
				theWin_Info->x_symmetric= 0;
				theWin_Info->y_symmetric= 0;
				xtb_bt_set( AF(YFIT_F).win, 0, NULL);
				xtb_bt_set( AF(ASPECT_F).win, 0, NULL);
			}
		}
		else if( val== &set_lblx1 ){
			if( data_sn_number>= setNumber && data_sn_number< setNumber+ theWin_Info->ulabels ){
			  UserLabel *ul= GetULabelNr( data_sn_number- setNumber);
				if( ul->x1!= value ){
					ul->x1= value;
					changed+= 1;
				}
			}
			else if( data_sn_number>= 0 && data_sn_number< setNumber ){
				if( AllSets[data_sn_number].show_legend ){
					if( theWin_Info->_legend_ulx!= value ){
						theWin_Info->_legend_ulx= value;
						changed+= 1;
					}
				}
				else if( theWin_Info->error_type[data_sn_number]== INTENSE_FLAG ){
					if( theWin_Info->IntensityLegend._legend_ulx!= value ){
						theWin_Info->IntensityLegend._legend_ulx= value;
						changed+= 1;
					}
				}
				else{
					Boing(10);
				}
			}
			else if( data_sn_number== -2 || data_sn_number== -1 ){
			  double *x;
				if( data_sn_number== -2 ){
					x= &theWin_Info->xname_x;
				}
				else{
					x= &theWin_Info->yname_x;
				}
				if( *x!= value ){
					*x= value;
					changed+= 1;
				}
			}
			else{
				Boing(5);
			}
		}
		else if( val== &set_lbly1 ){
			if( data_sn_number>= setNumber && data_sn_number< setNumber+ theWin_Info->ulabels ){
			  UserLabel *ul= GetULabelNr( data_sn_number- setNumber);
				if( ul->y1!= value ){
					ul->y1= value;
					changed+= 1;
				}
			}
			else if( data_sn_number>= 0 && data_sn_number< setNumber ){
				if( AllSets[data_sn_number].show_legend ){
					if( theWin_Info->_legend_uly!= value ){
						theWin_Info->_legend_uly= value;
						changed+= 1;
					}
				}
				else if( theWin_Info->error_type[data_sn_number]== INTENSE_FLAG ){
					if( theWin_Info->IntensityLegend._legend_uly!= value ){
						theWin_Info->IntensityLegend._legend_uly= value;
						changed+= 1;
					}
				}
				else{
					Boing(10);
				}
			}
			else if( data_sn_number== -2 || data_sn_number== -1 ){
			  double *y;
				if( data_sn_number== -2 ){
					y= &theWin_Info->xname_y;
				}
				else{
					y= &theWin_Info->yname_y;
				}
				if( *y!= value ){
					*y= value;
					changed+= 1;
				}
			}
			else{
				Boing(5);
			}
		}
		else if( val== &set_lblx2 ){
			if( data_sn_number>= setNumber && data_sn_number< setNumber+ theWin_Info->ulabels ){
			  UserLabel *ul= GetULabelNr( data_sn_number- setNumber);
				if( ul->x2!= value ){
					ul->x2= value;
					changed+= 1;
				}
			}
			else if( data_sn_number>= 0 && data_sn_number< setNumber ){
				for( i= nr; i< N; i++ ){
					if( i>= 0 && apply_ok(i) ){
						if( all && i!= dsn ){
							data_sn_number= i;
							set_changeables(2,True);
							SD_dfn_fun( win, ch, number, val );
						}
						else{
						  double oval= value;
							if( value< -1 ){
								value= -1;
							}
							else if( AllSets[i].numPoints>= 0 && value>= AllSets[data_sn_number].numPoints ){
								value= AllSets[i].numPoints- 1;
							}
							if( AllSets[i].error_point!= (int) oval ){
								AllSets[i].error_point= (int) oval;
								changed+= 1;
							}
						}
					}
				}
			}
			else{
				Boing(5);
			}
		}
		else if( val== &set_lbly2 ){
			if( data_sn_number>= setNumber && data_sn_number< setNumber+ theWin_Info->ulabels ){
			  UserLabel *ul= GetULabelNr( data_sn_number- setNumber);
				if( ul->y2!= value ){
					ul->y2= value;
					changed+= 1;
				}
			}
			else if( data_sn_number>= 0 && data_sn_number< setNumber ){
				for( i= nr; i< N; i++ ){
					if( i>= 0 && apply_ok(i) ){
						if( all && i!= dsn ){
							data_sn_number= i;
							set_changeables(2,True);
							SD_dfn_fun( win, ch, number, val );
						}
						else{
							if( value< 0 ){
								value= 0;
							}
							if( AllSets[i].NumObs!= (int) value ){
								AllSets[i].NumObs= (int) value;
								changed+= 1;
							}
						}
					}
				}
			}
			else{
				Boing(5);
			}
		}
		else if( val== &fileNumber ){
			if( data_sn_number>= setNumber && data_sn_number< setNumber+ theWin_Info->ulabels ){
			  UserLabel *ul= GetULabelNr( data_sn_number- setNumber);
				if( ul->vertical!= (int) value ){
					ul->vertical= (int) value;
					changed+= 1;
				}
			}
			if( data_sn_number>= 0 && data_sn_number< setNumber ){
				for( i= nr; i< N; i++ ){
					if( i>= 0 && apply_ok(i) ){
						if( all && i!= dsn ){
							data_sn_number= i;
							set_changeables(2,True);
							SD_dfn_fun( win, ch, number, val );
						}
						else{
							if( value>= 0 ){
								if( theWin_Info->fileNumber[i]!= (int) value ){
									AllSets[i].fileNumber= (int) value;
									theWin_Info->fileNumber[i]= (int) value;
									changed+= 1;
								}
							}
						}
					}
				}
			}
			else{
				Boing(5);
			}
		}
		else if( val== &set_setlink ){
			if( data_sn_number>= setNumber && data_sn_number< setNumber+ theWin_Info->ulabels ){
			  UserLabel *ul= GetULabelNr( data_sn_number- setNumber);
				if( ul->set_link!= (int) value ){
					ul->set_link= (int) value;
					changed+= 1;
				}
			}
			else if( data_sn_number>= 0 && data_sn_number< setNumber ){
				for( i= nr; i< N; i++ ){
					if( i>= 0 && apply_ok(i) ){
						if( all && i!= dsn ){
							data_sn_number= i;
							set_changeables(2,True);
							SD_dfn_fun( win, ch, number, val );
						}
						else if( AllSets[i].set_link>= 0 || AllSets[i].numPoints<= 0 ){
							if( AllSets[i].set_link!= (int) value ){
								LinkSet2( &AllSets[i], (int) value );
								changed+= 1;
							}
						}
						else{
							Boing(15);
						}
					}
				}
			}
			else{
				Boing(5);
			}
		}
		else if( val== &vectorLength ){
			if( data_sn_number>= setNumber && data_sn_number< setNumber+ theWin_Info->ulabels ){
			  UserLabel *ul= GetULabelNr( data_sn_number- setNumber);
				if( ul->pnt_nr!= (int) value ){
					if( ul->set_link>= 0 && ul->set_link< setNumber && value>= 0 ){
					  int NP= AllSets[ul->set_link].numPoints;
						CLIP_EXPR( ul->pnt_nr, (int) value, 0, NP-1 );
						update_LinkedLabel( theWin_Info, ul, &AllSets[ul->set_link], ul->pnt_nr, ul->short_flag );
						set_changeables(2,True);
					}
					else{
						ul->pnt_nr= (int) value;
					}
					changed+= 1;
				}
			}
			else if( data_sn_number>= 0 && data_sn_number< setNumber ){
				for( i= nr; i< N; i++ ){
					if( i>= 0 && apply_ok(i) ){
						if( all && i!= dsn ){
							data_sn_number= i;
							set_changeables(2,True);
							SD_dfn_fun( win, ch, number, val );
						}
						else{
						  int ch= 0;
							if( !Parse_vectorPars( number, &AllSets[i], False, &ch, NULL ) ){
								Boing(10);
							}
							changed+= ch;
						}
					}
				}
			}
			else{
				Boing(5);
			}
		}
		else if( val== &xcol ){
			if( data_sn_number>= 0 && data_sn_number< setNumber ){
				for( i= nr; i< N; i++ ){
					if( i>= 0 && apply_ok(i) ){
						if( all && i!= dsn ){
							data_sn_number= i;
							set_changeables(2,True);
							SD_dfn_fun( win, ch, number, val );
						}
						else{
							if( value< 0 ){
								value= 0;
							}
							if( theWin_Info->xcol[i]!= (int) value ){
								CLIP_EXPR( theWin_Info->xcol[i], (int) value, 0, AllSets[i].ncols- 1 );
								changed+= 1;
							}
						}
					}
				}
			}
			else{
				Boing(5);
			}
		}
		else if( val== &ycol ){
			if( data_sn_number>= 0 && data_sn_number< setNumber ){
				for( i= nr; i< N; i++ ){
					if( i>= 0 && apply_ok(i) ){
						if( all && i!= dsn ){
							data_sn_number= i;
							set_changeables(2,True);
							SD_dfn_fun( win, ch, number, val );
						}
						else{
							if( value< 0 ){
								value= 0;
							}
							if( theWin_Info->ycol[i]!= (int) value ){
								CLIP_EXPR( theWin_Info->ycol[i], (int) value, 0, AllSets[i].ncols- 1 );
								changed+= 1;
							}
						}
					}
				}
			}
			else{
				Boing(5);
			}
		}
		else if( val== &ecol ){
			if( data_sn_number>= 0 && data_sn_number< setNumber ){
				for( i= nr; i< N; i++ ){
					if( i>= 0 && apply_ok(i) ){
						if( all && i!= dsn ){
							data_sn_number= i;
							set_changeables(2,True);
							SD_dfn_fun( win, ch, number, val );
						}
						else{
							if( value< 0 ){
								value= 0;
							}
							if( theWin_Info->ecol[i]!= (int) value ){
								CLIP_EXPR( theWin_Info->ecol[i], (int) value, 0, AllSets[i].ncols- 1 );
								changed+= 1;
							}
						}
					}
				}
			}
			else{
				Boing(5);
			}
		}
		else if( val== &lcol ){
			if( data_sn_number>= 0 && data_sn_number< setNumber ){
				for( i= nr; i< N; i++ ){
					if( i>= 0 && apply_ok(i) ){
						if( all && i!= dsn ){
							data_sn_number= i;
							set_changeables(2,True);
							SD_dfn_fun( win, ch, number, val );
						}
						else{
							if( value< 0 ){
								value= 0;
							}
							if( theWin_Info->lcol[i]!= (int) value ){
								CLIP_EXPR( theWin_Info->lcol[i], (int) value, 0, AllSets[i].ncols- 1 );
								changed+= 1;
							}
						}
					}
				}
			}
			else{
				Boing(5);
			}
		}
		else if( val== &Ncol ){
			for( i= nr; i< N; i++ ){
				if( i>= 0 && i< setNumber ){
					if( i>= 0 && apply_ok(i) ){
						if( all && i!= dsn ){
							data_sn_number= i;
							set_changeables(2,True);
							SD_dfn_fun( win, ch, number, val );
						}
						else{
							if( AllSets[i].Ncol!= (int) value ){
								CLIP_EXPR( AllSets[i].Ncol, (int) value, value, AllSets[i].ncols- 1 );
								changed+= 1;
							}
						}
					}
				}
				else{
					Boing(5);
				}
			}
		}
		else if( val== &data_sn_lineWidth ){
			  /* set option that can be changed for all sets:
			   \ nr and N will be initialised accordingly (data_sn_number, data_sn_number+1
			   \ for 1 set, 0, setNumber for all.)
			   */
			for( i= nr; i< N; i++ ){
				if( i>= 0 && apply_ok(i) && i< setNumber ){
					if( all && i!= dsn ){
					  /* If changing the value for all sets, set data_sn_number to the current
					   \ set, update the Dialog's fields for that set, and call ourselves again
					   \ with the same keystroke to cause the same operation to be applied to
					   \ the current set. This should of course not be done when the current set
					   \ is the initial set (the value <dsn> of data_sn_number when we first got here).
					   */
						data_sn_number= i;
						set_changeables(2,True);
						SD_dfn_fun( win, ch, number, val );
					}
					else{
					  /* This does the actual modification. We come here either directly (level==1) when doing
					   \ just 1 set, or when doing set # <dsn>, or through an extra call (level==2) to ourselves
					   \ in the other cases.
					   */
						if( AllSets[i].lineWidth!= value ){
							data_sn_lineWidth= value;
							changed+= 1;
							AllSets[i].lineWidth= data_sn_lineWidth;
						}
					}
				}
				else if( i< setNumber + theWin_Info->ulabels ){
				  UserLabel *ul= GetULabelNr(i-setNumber);
					if( ul && ul->lineWidth!= value ){
						data_sn_lineWidth= value;
						changed+= 1;
						ul->lineWidth= data_sn_lineWidth;
					}
				}
			}
		}
		else if( val== &data_sn_linestyle ){
			for( i= nr; i< N; i++ ){
				if( i>= 0 && apply_ok(i) && i< setNumber ){
					if( all && i!= dsn ){
						data_sn_number= i;
						set_changeables(2,True);
						SD_dfn_fun( win, ch, number, val );
					}
					else{
						if( value< 0 ){
							value= 0;
						}
						if( AllSets[i].linestyle!= (int) value ){
							data_sn_linestyle= (int) value;
							changed+= 1;
							AllSets[i].linestyle= data_sn_linestyle;
						}
					}
				}
			}
		}
		else if( val== &data_sn_markSize ){
			for( i= nr; i< N; i++ ){
				if( i>= 0 && apply_ok(i) && i< setNumber ){
					if( all && i!= dsn ){
						data_sn_number= i;
						set_changeables(2,True);
						SD_dfn_fun( win, ch, number, val );
					}
					else{
						if( AllSets[i].markSize!= value ){
							changed+= 1;
							AllSets[i].markSize= value;
						}
					}
				}
			}
		}
		else if( val== &data_sn_markstyle ){
			for( i= nr; i< N; i++ ){
				if( i>= 0 && apply_ok(i) && i< setNumber ){
					if( all && i!= dsn ){
						data_sn_number= i;
						set_changeables(2,True);
						SD_dfn_fun( win, ch, number, val );
					}
					else{
						if( value> 0 && AllSets[i].markstyle!= -value ){
							data_sn_markstyle= (int) value;
							changed+= 1;
							AllSets[i].markstyle= -1 * abs(data_sn_markstyle);
						}
					}
				}
			}
		}
		else if( val== &data_sn_elineWidth ){
			for( i= nr; i< N; i++ ){
				if( i>= 0 && apply_ok(i) && i< setNumber ){
					if( all && i!= dsn ){
						data_sn_number= i;
						set_changeables(2,True);
						SD_dfn_fun( win, ch, number, val );
					}
					else{
						if( AllSets[i].elineWidth!= value ){
							data_sn_elineWidth= value;
							changed+= 1;
							AllSets[i].elineWidth= data_sn_elineWidth;
						}
					}
				}
			}
		}
		else if( val== &data_sn_elinestyle ){
			for( i= nr; i< N; i++ ){
				if( i>= 0 && apply_ok(i) && i< setNumber ){
					if( all && i!= dsn ){
						data_sn_number= i;
						set_changeables(2,True);
						SD_dfn_fun( win, ch, number, val );
					}
					else{
						if( AllSets[i].elinestyle!= (int) value ){
							data_sn_elinestyle= (int) value;
							changed+= 1;
							AllSets[i].elinestyle= data_sn_elinestyle;
						}
					}
				}
			}
		}
		else if( val== &ebarWidth_set ){
			for( i= nr; i< N; i++ ){
				if( i>= 0 && apply_ok(i) && i< setNumber ){
					if( all && i!= dsn ){
						data_sn_number= i;
						set_changeables(2,True);
						SD_dfn_fun( win, ch, number, val );
					}
					else{
						if( NaN(value) ){
							if( AllSets[i].ebarWidth_set ){
								AllSets[i].ebarWidth_set= False;
								changed+= 1;
							}
						}
						else if( AllSets[i].ebarWidth!= value || !AllSets[i].ebarWidth_set ){
							changed+= 1;
							AllSets[i].ebarWidth= value;
							AllSets[i].ebarWidth_set= True;
						}
					}
				}
			}
		}
		else if( val== &barBase_set ){
			for( i= nr; i< N; i++ ){
				if( i>= 0 && apply_ok(i) && i< setNumber ){
					if( all && i!= dsn ){
						data_sn_number= i;
						set_changeables(2,True);
						SD_dfn_fun( win, ch, number, val );
					}
					else{
						if( NaN(value) ){
							if( AllSets[i].barBase_set ){
								AllSets[i].barBase_set= False;
								AllSets[i].barBase= barBase;
								changed+= 1;
							}
						}
						else if( AllSets[i].barBase!= value || !AllSets[i].barBase_set ){
							changed+= 1;
							AllSets[i].barBase= value;
							AllSets[i].barBase_set= True;
						}
					}
				}
			}
		}
		else if( val== &barWidth_set ){
			for( i= nr; i< N; i++ ){
				if( i>= 0 && apply_ok(i) && i< setNumber ){
					if( all && i!= dsn ){
						data_sn_number= i;
						set_changeables(2,True);
						SD_dfn_fun( win, ch, number, val );
					}
					else{
						if( NaN(value) ){
							if( AllSets[i].barWidth_set ){
								AllSets[i].barWidth_set= False;
								AllSets[i].barWidth= barWidth;
								changed+= 1;
							}
						}
						else if( AllSets[i].barWidth!= value || !AllSets[i].barWidth_set ){
							changed+= 1;
							AllSets[i].barWidth= value;
							AllSets[i].barWidth_set= True;
						}
					}
				}
			}
		}
		else if( val== &barType ){
			for( i= nr; i< N; i++ ){
				if( i>= 0 && apply_ok(i) && i< setNumber ){
					if( all && i!= dsn ){
						data_sn_number= i;
						set_changeables(2,True);
						SD_dfn_fun( win, ch, number, val );
					}
					else{
						CLIP( value, 0, BARTYPES-1);
						if( AllSets[i].barType!= (int) value ){
							changed+= 1;
							AllSets[i].barType= (int) value;
						}
					}
				}
			}
		}
		else if( val== &data_sn_plot_interval ){
			for( i= nr; i< N; i++ ){
				if( i>= 0 && apply_ok(i) && i< setNumber ){
					if( all && i!= dsn ){
						data_sn_number= i;
						set_changeables(2,True);
						SD_dfn_fun( win, ch, number, val );
					}
					else{
						if( AllSets[i].plot_interval!= (int) value ){
							data_sn_plot_interval= (int) value;
							changed+= 1;
							AllSets[i].plot_interval= data_sn_plot_interval;
						}
					}
				}
			}
		}
		else if( val== &data_sn_adorn_interval ){
			for( i= nr; i< N; i++ ){
				if( i>= 0 && apply_ok(i) && i< setNumber ){
					if( all && i!= dsn ){
						data_sn_number= i;
						set_changeables(2,True);
						SD_dfn_fun( win, ch, number, val );
					}
					else{
						if( AllSets[i].adorn_interval!= (int) value ){
							data_sn_adorn_interval= (int) value;
							changed+= 1;
							AllSets[i].adorn_interval= data_sn_adorn_interval;
						}
					}
				}
			}
		}
		else if( val== &data_start_arrow_orn ){
			for( i= nr; i< N; i++ ){
				if( i>= 0 && apply_ok(i) && i< setNumber ){
					if( all && i!= dsn ){
						data_sn_number= i;
						set_changeables(2,True);
						SD_dfn_fun( win, ch, number, val );
					}
					else{
						if( AllSets[i].sarrow_orn!= value ){
							data_start_arrow_orn= value;
							changed+= 1;
							if( !NaN(value) ){
								AllSets[i].sarrow_orn= value;
								AllSets[i].sarrow_orn_set= True;
							}
							else{
								AllSets[i].sarrow_orn_set= False;
							}
						}
					}
				}
			}
		}
		else if( val== &data_end_arrow_orn ){
			for( i= nr; i< N; i++ ){
				if( i>= 0 && apply_ok(i) && i< setNumber ){
					if( all && i!= dsn ){
						data_sn_number= i;
						set_changeables(2,True);
						SD_dfn_fun( win, ch, number, val );
					}
					else{
						if( AllSets[i].earrow_orn!= value ){
							data_end_arrow_orn= value;
							changed+= 1;
							if( !NaN(value) ){
								AllSets[i].earrow_orn= value;
								AllSets[i].earrow_orn_set= True;
							}
							else{
								AllSets[i].earrow_orn_set= False;
							}
						}
					}
				}
			}
		}
		else{
			Boing(5);
		}
		if( aaa ){
			_apply_to_drawn= 0;
			_apply_to_marked= 0;
			_apply_to_hlt= 0;
			_apply_to_new= 0;
			_apply_to_src= 0;
			xtb_bt_set( AF(SNATD_F).win, 0, NULL );
			xtb_bt_set( AF(SNATM_F).win, 0, NULL );
			xtb_bt_set( AF(SNATH_F).win, 0, NULL );
			xtb_bt_set( AF(SNATN_F).win, 0, NULL );
			xtb_bt_set( AF(SNATS_F).win, 0, NULL );
		}
		autoredraw= ard;
		level-= 1;
	}
	if( changed ){
		theWin_Info->redraw= 1;
		if( theWin_Info->redrawn== -3 ){
			theWin_Info->redrawn= 0;
		}
		theWin_Info->printed= 0;
		set_changeables(2,True);
		if( changed== 2 ){
			xtb_bt_set( AF(USER_F).win, theWin_Info->win_geo.user_coordinates, NULL);
		}
	}
	if( all ){
		data_sn_number= dsn;
		if( !changed ){
			set_changeables(2,True);
		}
	}
	if( ((ch== TAB || ch== XK_Tab) && !xtb_clipboard_copy) ){
		return( (CheckMask(xtb_modifier_state, ShiftMask))? goto_prev_text_box(win) : goto_next_text_box( &win) );
	}
	else{
		return XTB_HANDLED;
	}
}

#define SNNR_MINVAL	-22
#define SNLEG_CHARACTERS	"-uUtTdDpPaAbBxXyYnNRrFfgGIiEecCoOlL"

char *Data_fileName()
{ char *name= "";
	switch( data_sn_number ){
		case -22:
			name= "*COLUMNLABELS*";
			break;
		case -21:
			name= "*READ* text";
			break;
		case -20:
			name= "*ARGUMENTS*";
			break;
		case -19:
			name= "*ENTER_RAW_AFTER*";
			break;
		case -18:
			name= "*LEAVE_RAW_AFTER*";
			break;
		case -17:
			name= "*DUMP_BEFORE*";
			break;
		case -16:
			name= "*DUMP_AFTER*";
			break;
		case -15:
			name= "*DRAW_BEFORE*";
			break;
		case -14:
			name= "*DRAW_AFTER*";
			break;
		case -13:
			name= "*READ_FILE*";
			break;
		case -12:
			name= "*EVAL*";
			break;
		case -11:
		  /* If this changes, remember to update LegendorTitle()!	*/
			name= "*TRANSFORM_X*";
			break;
		case -10:
		  /* If this changes, remember to update LegendorTitle()!	*/
			name= "*TRANSFORM_Y*";
			break;
		case -9:
		  /* If this changes, remember to update LegendorTitle()!	*/
			name= "*DATA_INIT*";
			break;
		case -8:
		  /* If this changes, remember to update LegendorTitle()!	*/
			name= "*DATA_BEFORE*";
			break;
		case -7:
		  /* If this changes, remember to update LegendorTitle()!	*/
			name= "*DATA_PROCESS*";
			break;
		case -6:
		  /* If this changes, remember to update LegendorTitle()!	*/
			name= "*DATA_AFTER*";
			break;
		case -5:
		  /* If this changes, remember to update LegendorTitle()!	*/
			name= "*DATA_FINISH*";
			break;
		case -4:
			name= "-T <alt.title>";
			break;
		case -3:
			name= "-t <global title>";
			break;
		case -2:
			name= "X Label";
			break;
		case -1:
			name= "Y Label";
			break;
		default:
			if( data_sn_number< setNumber ){
				if( AllSets[data_sn_number].numPoints>= 0 ){
					if( theWin_Info->labels_in_legend ){
						name= AllSets[data_sn_number].YUnits;
					}
					else{
						name= AllSets[data_sn_number].fileName;
						if( name ){
							name= rindex( name, '/' );
						}
						if( !name ){
							name= AllSets[data_sn_number].fileName;
						}
					}
					if( !name ){
						name= "";
					}
				}
			}
			else{
				name= "User Label";
			}
			break;
	}
	return( name);
}


static int fileName_len, fileName_width, legend_len, legend_width;
#define MIN_LEGEND_LENGTH	MAX(MFNAME+D_FS,16)
static char *fileName_max, *setName_max, setName_max_dummy[MIN_LEGEND_LENGTH+12];

extern int StringCheck(char *, int, char *, int);

#define STRINGCHECK(s,max)	StringCheck(s,max,__FILE__,__LINE__)

char *clabels_string_buf= NULL;
int csb_len= 0;

char *LegendorTitle(int data_sn_number, int mode)
{ static char *buf= NULL;
  static int buflen= 0;

	if( !buflen ){
		if( (buf= calloc( 256, sizeof(char))) ){
			buflen= 256;
		}
		else{
			fprintf( StdErr, "LegendorTitle(): can't allocate 256 element textbuffer (%s)\n", serror() );
			return(NULL);
		}
	}

	if( data_sn_number< 0 ){
		switch( mode ){
			case 6:
				if( data_sn_number>= -11 && data_sn_number<= -10 && theWin_Info->transform.description ){
					return( theWin_Info->transform.description );
				}
				else if( data_sn_number>= -9 && data_sn_number<= -5 && theWin_Info->process.description ){
					return( theWin_Info->process.description );
				}
				else{
					return( "" );
				}
				break;
			default:
				return( "" );
				break;
		}
	}
	else if( data_sn_number>= setNumber ){
	  UserLabel *ul= GetULabelNr( data_sn_number- setNumber );
		switch( mode ){
			case 4:{
			  extern char *ULabel_pixelCName2();
				return( ULabel_pixelCName2( ul, NULL) );
				break;
			}
			default:
				return( ul->label );
				break;
		}
	}
	else{
		switch( mode ){
			case 1:
				if( AllSets[data_sn_number].titleText ){
					return( AllSets[data_sn_number].titleText);
				}
				else{
					return( "" );
				}
				break;
			case 2:
				if( AllSets[data_sn_number].process.set_process_len ){
					return( AllSets[data_sn_number].process.set_process);
				}
				else{
					return( "" );
				}
				break;
			case 3:
				if( AllSets[data_sn_number].numAssociations> 0 && AllSets[data_sn_number].Associations ){
				  int i, l= 0, nl;
				  char *c;
				  double *ass= AllSets[data_sn_number].Associations;
#ifdef OLD_ASSOC_DISPLAY
				  int naL= 0;
#else
				  int naL= 4+ log10((double)AllSets[data_sn_number].numAssociations)+ 2;
#endif
					for( i= 0; i< AllSets[data_sn_number].numAssociations; i++ ){
						c= d2str( ass[i], d3str_format, NULL);
						if( (nl= l+ strlen(c)+ 2+ naL)>= buflen- 1 ){
							nl+= 2;
							if( (buf= realloc( buf, nl* sizeof(char))) ){
								buflen= nl;
							}
							else{
								fprintf( StdErr,
									"LegendorTitle(): can't expand %d element textbuffer to %d elements (%s)\n",
										buflen, nl, serror()
								);
								return(NULL);
							}
						}
						{
							if( l ){
								l= sprintf( buf, "%s, %s", buf, c );
							}
							else{
#ifdef OLD_ASSOC_DISPLAY
								l= sprintf( buf, "[%d] %s", AllSets[data_sn_number].numAssociations, c );
#else
								l= sprintf( buf, "%s", c );
#endif
							}
						}
					}
#ifndef OLD_ASSOC_DISPLAY
					l= sprintf( buf, "%s #[%d]", buf, AllSets[data_sn_number].numAssociations );
#endif
					STRINGCHECK(buf, buflen );
					return( buf );
				}
				else{
					return( "" );
				}
				break;
			case 4:
				if( AllSets[data_sn_number].pixvalue< 0 ){
					return( AllSets[data_sn_number].pixelCName );
				}
				else{
					sprintf( buf, "default [%s]", AllAttrs[AllSets[data_sn_number].pixvalue].pixelCName );
					STRINGCHECK( buf, buflen );
					return( buf );
				}
				break;
			case 5:
				if( theWin_Info->legend_line[data_sn_number].pixvalue< 0 ){
					return( theWin_Info->legend_line[data_sn_number].pixelCName );
				}
				else{
					sprintf( buf, "default [%s]", highlightCName );
					STRINGCHECK( buf, buflen );
					return( buf );
				}
				break;
			case 6:
				if( AllSets[data_sn_number].process.description && AllSets[data_sn_number].process.description ){
					return( AllSets[data_sn_number].process.description);
				}
				else{
					return( "" );
				}
				break;
			case 7:{
			  Sinc sinc;
				Sinc_string_behaviour( &sinc, NULL, 0,0, SString_Dynamic );
				if( AllSets[data_sn_number].ColumnLabels ){
					Sprint_SetLabelsList( &sinc, AllSets[data_sn_number].ColumnLabels, NULL, "\n" );
					if( sinc.sinc.string ){
						strcpalloc( &clabels_string_buf, &csb_len, sinc.sinc.string );
						xfree( sinc.sinc.string );
						return( clabels_string_buf );
					}
					else{
						return( "" );
					}
				}
				else{
					return( "" );
				}
				break;
			}
			case 0:
			default:
				if( AllSets[data_sn_number].setName ){
					return( AllSets[data_sn_number].setName);
				}
				else{
					return( "" );
				}
				break;
		}
	}
}

int SD_dynamic_resize= False;

int find_fileName_max_AND_legend_len()
{  int dsn= data_sn_number, w;
   char *sm= theWin_Info->XUnits;
	data_sn_number= SNNR_MINVAL;
	fileName_max= Data_fileName();
	fileName_len= SD_strlen( fileName_max);
	fileName_width= xtb_TextWidth( fileName_max, dialogFont.font, dialog_greekFont.font);

	if( SD_dynamic_resize ){
		legend_width= xtb_TextWidth( theWin_Info->XUnits, dialogFont.font, dialog_greekFont.font);
		legend_len= SD_strlen(theWin_Info->XUnits);
		if( SD_strlen( theWin_Info->tr_XUnits) > legend_len ){
			legend_len= SD_strlen(theWin_Info->tr_XUnits);
			sm= theWin_Info->tr_XUnits;
		}
		if( (w= xtb_TextWidth( theWin_Info->tr_XUnits, dialogFont.font, dialog_greekFont.font)) > legend_width ){
			legend_width= w;
		}
		if( (w= xtb_TextWidth( theWin_Info->YUnits, dialogFont.font, dialog_greekFont.font)) > legend_width ){
			legend_width= w;
		}
		if( (w= xtb_TextWidth( theWin_Info->tr_YUnits, dialogFont.font, dialog_greekFont.font)) > legend_width ){
			legend_width= w;
		}
		if( SD_strlen( theWin_Info->YUnits) > legend_len ){
			legend_len= SD_strlen(theWin_Info->YUnits);
			sm= theWin_Info->YUnits;
		}
		if( SD_strlen( theWin_Info->tr_YUnits) > legend_len ){
			legend_len= SD_strlen(theWin_Info->tr_YUnits);
			sm= theWin_Info->tr_YUnits;
		}
		data_sn_number= 0;
	/* 	setName_max= AllSets[0].setName;	*/
		setName_max= (data_sn_number< setNumber)? LegendorTitle(data_sn_number, LegendFunctionNr) :
				(LegendFunctionNr==0)? GetLabelNr( data_sn_number- setNumber) : "";
		for( data_sn_number= SNNR_MINVAL; data_sn_number< setNumber+ theWin_Info->ulabels; data_sn_number++ ){
		  char *s;
			if( data_sn_number>= 0 && (data_sn_number>= setNumber || AllSets[data_sn_number].numPoints> 0) ){
				s= (data_sn_number< setNumber)? LegendorTitle(data_sn_number, LegendFunctionNr) :
						(LegendFunctionNr==0)? GetLabelNr( data_sn_number- setNumber) : "";
				if( (w= xtb_TextWidth( s, dialogFont.font, dialog_greekFont.font))> legend_width ){
					legend_width= w;
					setName_max= s;
				}
				if( SD_strlen( s)> legend_len ){
					legend_len= SD_strlen( s );
					sm= s;
				}
			}
			if( (w= xtb_TextWidth(Data_fileName(), dialogFont.font, dialog_greekFont.font))> fileName_width ){
				fileName_max= Data_fileName();
				fileName_width= w;
				fileName_len= SD_strlen( fileName_max );
			}
		}
	}
	else{
		legend_len= 0;
	}
	if( legend_len< MIN_LEGEND_LENGTH ){
		legend_len= MIN_LEGEND_LENGTH;
		memset( setName_max_dummy, (int) '8', MIN_LEGEND_LENGTH+10);
		setName_max_dummy[MIN_LEGEND_LENGTH+10]= '\0';
		setName_max= setName_max_dummy;
	}
	legend_len+= 10;
	if( setName_max!= setName_max_dummy ){
	  static char *c;
	  /* append 10 '8's to setName_max	*/
		xfree(c);
		if( (c= calloc( 1,  SD_strlen(setName_max)+ 12 ))  ){
		  int len= SD_strlen(setName_max);
			strcpy( c, setName_max);
			memset( &c[len], '8', 10 );
			c[len+10]= '\0';
			setName_max= c;
		}
	}
	legend_width= xtb_TextWidth( setName_max, dialogFont.font, dialog_greekFont.font );
	if( debugFlag ){
		fprintf( StdErr, "max legend: [%s],%d(-10)\n\tmax legendwidth: [%s],%d(-%d)\n\tmax fileName: %d, width= %d\n",
			sm, legend_len,
			setName_max, legend_width, 10*XFontWidth(dialogFont.font),
			fileName_len, fileName_width
		);
		fflush( StdErr );
	}
	data_sn_number= dsn;
	return(0);
}

int show_legend_Value()
{
	if( data_sn_number< 0 || data_sn_number> setNumber ){
		return( 0 );
	}
	else{
		return( AllSets[data_sn_number].show_legend );
	}
}

int show_llines_Value()
{
	if( data_sn_number< 0 || data_sn_number> setNumber ){
		return( 0 );
	}
	else{
		return( AllSets[data_sn_number].show_llines );
	}
}

int raw_display_Value()
{
	if( data_sn_number< 0 || data_sn_number> setNumber ){
		return( 0 );
	}
	else{
		return( AllSets[data_sn_number].raw_display );
	}
}

int points_added_Value()
{
	if( data_sn_number< 0 || data_sn_number> setNumber ){
		return( 0 );
	}
	else{
		return( AllSets[data_sn_number].points_added );
	}
}

int start_arrow_Value()
{
	if( data_sn_number< 0 || data_sn_number> setNumber ){
		return( 0 );
	}
	else{
		return( AllSets[data_sn_number].arrows & 0x01 );
	}
}

int end_arrow_Value()
{
	if( data_sn_number< 0 || data_sn_number> setNumber ){
		return( 0 );
	}
	else{
		return( AllSets[data_sn_number].arrows & 0x02 );
	}
}

int floating_Value()
{
	if( data_sn_number< 0 || data_sn_number> setNumber ){
		return( 0 );
	}
	else{
		return( AllSets[data_sn_number].floating );
	}
}

int noLines_Value()
{
	if( data_sn_number< 0 || data_sn_number> setNumber ){
		return( 0 );
	}
	else{
		return( AllSets[data_sn_number].noLines );
	}
}

int barFlag_Value()
{
	if( data_sn_number< 0 || data_sn_number> setNumber ){
		return( 0 );
	}
	else{
		return( AllSets[data_sn_number].barFlag );
	}
}

int markFlag_Value()
{
	if( data_sn_number< 0 || data_sn_number> setNumber ){
		return( 0 );
	}
	else{
		return( AllSets[data_sn_number].markFlag );
	}
}

int use_error_Value()
{
	if( data_sn_number< 0 || data_sn_number> setNumber ){
		return( theWin_Info->use_errors );
	}
	else{
		return( AllSets[data_sn_number].use_error );
	}
}

int overwrite_marks_Value()
{
	if( data_sn_number< 0 || data_sn_number> setNumber ){
		return( 0 );
	}
	else{
		return( AllSets[data_sn_number].overwrite_marks );
	}
}

int pixelMarks_Value()
{
	if( data_sn_number< 0 || data_sn_number> setNumber ){
		return( 0 );
	}
	else{
		return( AllSets[data_sn_number].pixelMarks );
	}
}

int draw_set_Value()
{
	if( data_sn_number< 0 || data_sn_number> setNumber ){
		return( 0 );
	}
	else{
		return( theWin_Info->draw_set[data_sn_number]> 0 );
	}
}

int mark_set_Value()
{
	if( data_sn_number< 0 || data_sn_number> setNumber ){
		return( 0 );
	}
	else{
		return( theWin_Info->mark_set[data_sn_number]> 0 );
	}
}

int highlight_Value()
{
	if( data_sn_number< 0 || data_sn_number> setNumber ){
		return( 0 );
	}
	else{
		return( theWin_Info->legend_line[data_sn_number].highlight> 0 );
	}
}

extern XSegment *Xsegs, *lowYsegs, *highYsegs;		/* Point space for X */
#define LYsegs	lowYsegs
#define HYsegs	highYsegs
extern XSegment_error *XsegsE;
extern long XsegsSize, YsegsSize, XsegsESize;

int new_legend, new_set, bounds_changed;

extern int GetColor( char *name, Pixel *pix);
extern char *ParsedColourName;
#define StoreCName(name)	xfree(name);name=XGstrdup(ParsedColourName)

xtb_hret SD_snl_fun(Window win, int ch, char *text, xtb_data Val)
/*
 * This is the handler function for the text widget for
 * editing the selected legend.  It supports simple
 * line editing operations.
 */
{
#ifdef TAB_WARPS
   int x_loc, y_loc;
#endif
   int new_file;
   char *fileName;
   void *val= Val;

	if( !theWin_Info ){
		return( XTB_STOP );
	}

    if ((ch == BACKSPACE) || (ch == DELETE)) {
		if( !( (ch== DELETE)? xtb_ti_dch_right(win) : xtb_ti_dch(win)) )
			Boing( 5);
		else
			legend_changed= 1;
    }
	else if ((ch == CONTROL_U) || (ch == CONTROL_X)) {
		(void) xtb_ti_set(win, "", (xtb_data) 0);
		legend_changed= 1;
    }
	else if( ((ch== TAB || ch== XK_Tab) && !xtb_clipboard_copy) || ch== XK_Down || ch== XK_Up ){
		if( val== &new_legend ){
			strncpy( data_legend_buf, text, SD_LMaxBufSize);
		}
		else if( val== &new_set ){
			strncpy( fileName_buf, text, LEGEND-1);
		}
		if( data_sn_number>= SNNR_MINVAL && data_sn_number< setNumber+ theWin_Info->ulabels ){
			if( legend_changed){
			  int redraw= theWin_Info->redraw;
			  char *fname= data_legend_buf;
			  char Fname[LEGEND+256];
			  FILE *strm= NULL;
				theWin_Info->redraw= 1;
				if( theWin_Info->redrawn== -3 ){
					theWin_Info->redrawn= 0;
				}
				theWin_Info->printed= 0;
				if( data_sn_number== -20 && val== &new_legend ){
				  LocalWin *aw= ActiveWin;
					ActiveWin= theWin_Info;
					ParseArgsString2( data_legend_buf, setNumber );
					ActiveWin= aw;
					theWin_Info->printed= 0;
				}

				  /* 20001220: I added a cleanup() of data_legend_buf before passing it to strcpalloc() and new_process_...()
				   \ in order to do a correct parsing of '#x' etc. in strings.
				   */
				
				else if( data_sn_number== -19 && val== &new_legend ){
					strcpalloc( &theWin_Info->process.enter_raw_after, &theWin_Info->process.enter_raw_after_allen, cleanup(data_legend_buf) );
					new_process_enter_raw_after( theWin_Info );
					theWin_Info->printed= 0;
				}
				else if( data_sn_number== -18 && val== &new_legend ){
					strcpalloc( &theWin_Info->process.leave_raw_after, &theWin_Info->process.leave_raw_after_allen, cleanup(data_legend_buf) );
					new_process_leave_raw_after( theWin_Info );
					theWin_Info->printed= 0;
				}
				else if( data_sn_number== -17 && val== &new_legend ){
					strcpalloc( &theWin_Info->process.dump_before, &theWin_Info->process.dump_before_allen, cleanup(data_legend_buf) );
					new_process_dump_before( theWin_Info );
					theWin_Info->printed= 0;
				}
				else if( data_sn_number== -16 && val== &new_legend ){
					strcpalloc( &theWin_Info->process.dump_after, &theWin_Info->process.dump_after_allen, cleanup(data_legend_buf) );
					new_process_dump_after( theWin_Info );
					theWin_Info->printed= 0;
				}
				else if( data_sn_number== -15 && val== &new_legend ){
					strcpalloc( &theWin_Info->process.draw_before, &theWin_Info->process.draw_before_allen, cleanup(data_legend_buf) );
					new_process_draw_before( theWin_Info );
					theWin_Info->printed= 0;
				}
				else if( data_sn_number== -14 && val== &new_legend ){
					strcpalloc( &theWin_Info->process.draw_after, &theWin_Info->process.draw_after_allen, cleanup(data_legend_buf) );
					new_process_draw_after( theWin_Info );
					theWin_Info->printed= 0;
				}
				else if( data_sn_number== -21 && val== &new_legend ){
					xtb_bt_set( AF(OK_F).win, 1, NULL );
					ParseInputString( theWin_Info, data_legend_buf );
					xtb_bt_set( AF(OK_F).win, 0, NULL );
				}
				else if( data_sn_number== -13 && val== &new_legend ){
				  extern char *tildeExpand();
					errno= 0;
					if( fname[0]== '|' ){
					  extern Boolean PIPE_error;
						fname++;
						tildeExpand( Fname, fname );
						PIPE_error= False;
						strm= popen( Fname, "r");
					}
					else{
						strm = fopen( tildeExpand( Fname, fname), "r");
					}
/* include_file:;	*/
					if( !strm){
						xtb_error_box( theWin_Info->window, "Can't open file\n", Fname );
						theWin_Info->redraw= redraw;
					}
					else{
						if( CleanSheet ){
						  int idx;
						  extern double *ascanf_TotalSets;
						  extern int maxitems;
						  /* "erase" old datasets.	*/
							for( idx= 0; idx< setNumber; idx++ ){
								  /* This should be enough: setting numPoints to 0 will
								   \ cause ReadData + children to re-initialise the sets.
								   */
								AllSets[idx].numPoints= 0;
							}
							setNumber= 0;
							*ascanf_TotalSets= 0;
							maxitems= 0;
							titleText[0]= '\0';
							titleText2[0]= '\0';
							theWin_Info->XUnits[0]= '\0';
							theWin_Info->YUnits[0]= '\0';
							theWin_Info->tr_XUnits[0]= '\0';
							theWin_Info->tr_YUnits[0]= '\0';
						}
						xtb_bt_set( AF(OK_F).win, 1, NULL );
						IncludeFile( theWin_Info, strm, Fname, True, NULL );
						xtb_bt_set( AF(OK_F).win, 0, NULL );
						if( fname== data_legend_buf ){
							fclose(strm);
						}
						else{
							pclose( strm);
						}
					}
				}
				else if( data_sn_number== -12 && val== &new_legend ){
				  double val= 0.0;
				  char as= ascanf_separator;
					ascanf_window= theWin_Info->window;
					ActiveWin= theWin_Info;
					ascanf_separator= SD_ascanf_separator_buf[0];
					new_param_now( data_legend_buf, &val, -1 );
					ascanf_separator= as;
					if( !SD_Dialog.mapped || !theWin_Info ){
					  /* While this can theoretically also arrive in the other situations,
					   \ it is more likely to happen when user kills us (= tells us to quit)
					   \ in the middle of a lengthy EVAL command.
					   */
						return( XTB_HANDLED );
					}
				}
				else if( data_sn_number>= -11 && data_sn_number<= -10 && LegendFunctionNr== 6 ){
					xfree( theWin_Info->transform.description );
					theWin_Info->transform.description= XGstrdup( cleanup(data_legend_buf) );
				}
				else if( data_sn_number== -11 && val== &new_legend ){
					strcpalloc( &theWin_Info->transform.x_process, &theWin_Info->transform.x_allen, cleanup(data_legend_buf) );
					new_transform_x_process( theWin_Info );
					theWin_Info->printed= 0;
				}
				else if( data_sn_number== -10 && val== &new_legend ){
					strcpalloc( &theWin_Info->transform.y_process, &theWin_Info->transform.y_allen, cleanup(data_legend_buf) );
					new_transform_y_process( theWin_Info );
					theWin_Info->printed= 0;
				}
				else if( data_sn_number>= -9 && data_sn_number<= -5 && LegendFunctionNr== 6 ){
					xfree( theWin_Info->process.description );
					theWin_Info->process.description= XGstrdup( cleanup(data_legend_buf) );
				}
				else if( data_sn_number== -9 && val== &new_legend ){
					strcpalloc( &theWin_Info->process.data_init, &theWin_Info->process.data_init_allen, cleanup(data_legend_buf) );
					new_process_data_init( theWin_Info );
					theWin_Info->printed= 0;
				}
				else if( data_sn_number== -8 && val== &new_legend ){
					strcpalloc( &theWin_Info->process.data_before, &theWin_Info->process.data_before_allen, cleanup(data_legend_buf) );
					new_process_data_before( theWin_Info );
					theWin_Info->printed= 0;
				}
				else if( data_sn_number== -7 && val== &new_legend ){
					strcpalloc( &theWin_Info->process.data_process, &theWin_Info->process.data_process_allen, cleanup(data_legend_buf) );
					new_process_data_process( theWin_Info );
					theWin_Info->printed= 0;
				}
				else if( data_sn_number== -6 && val== &new_legend ){
					strcpalloc( &theWin_Info->process.data_after, &theWin_Info->process.data_after_allen, cleanup(data_legend_buf) );
					new_process_data_after( theWin_Info );
					theWin_Info->printed= 0;
				}
				else if( data_sn_number== -5 && val== &new_legend ){
					strcpalloc( &theWin_Info->process.data_finish, &theWin_Info->process.data_finish_allen, cleanup(data_legend_buf) );
					new_process_data_finish( theWin_Info );
					theWin_Info->printed= 0;
				}
				else if( data_sn_number== -4 && val== &new_legend ){
					xfree(titleText2);
					titleText2= XGstrdup(data_legend_buf);
					theWin_Info->printed= 0;
				}
				else if( data_sn_number== -3 && val== &new_legend ){
					strncpy( titleText, data_legend_buf, MAXBUFSIZE);
					titleText[MAXBUFSIZE]= '\0';
					theWin_Info->printed= 0;
				}
				else if( data_sn_number== -2 && val== &new_legend ){
					strncpy( raw_XLABEL(theWin_Info), data_legend_buf, MAXBUFSIZE-1);
					theWin_Info->printed= 0;
				}
				else if( data_sn_number== -1 && val== &new_legend ){
					strncpy( raw_YLABEL(theWin_Info), data_legend_buf, MAXBUFSIZE-1);
					theWin_Info->printed= 0;
				}
				else if( data_sn_number== -22 && val== &new_legend ){
				  char *command= concat( "*COLUMNLABELS* new\n", command, "\n", NULL );
					if( command ){
						ParseInputString( theWin_Info, command );
						xfree( command );
						theWin_Info->printed= 0;
					}
				}
				else if( data_sn_number< setNumber ){
					if( val== &new_legend ){
						switch( LegendFunctionNr ){
						  extern char *String_ParseVarNames();
							case 1:
								xfree( AllSets[data_sn_number].titleText );
								AllSets[data_sn_number].titleText= XGstrdup( data_legend_buf);
								AllSets[data_sn_number].titleChanged= 1;
								break;
							case 2:
								xfree( AllSets[data_sn_number].process.set_process );
								AllSets[data_sn_number].process.set_process= XGstrdup( data_legend_buf);
								new_process_set_process( theWin_Info, &AllSets[data_sn_number] );
								break;
#ifndef OLD_ASSOC_DISPLAY
							case 3:{
							  int N, i;
							  double dum;
							  char *c= index( data_legend_buf, '#' );
							  DataSet *this_set= &AllSets[data_sn_number];
								if( c ){
									*c= '\0';
								}
								N= new_param_now( data_legend_buf, &dum, -1 );
								if( N> this_set->allocAssociations ){
									this_set->Associations= (double*) realloc( this_set->Associations,
										sizeof(double)* (this_set->allocAssociations= N)
									);
								}
								if( this_set->Associations ){
									this_set->numAssociations= N;
									for( i= 0; i< N; i++ ){
										this_set->Associations[i]= param_scratch[i];
									}
									if( this_set->allocAssociations> ASCANF_MAX_ARGS ){
										Ascanf_AllocMem( this_set->allocAssociations );
									}
								}
								else{
									xtb_error_box( theWin_Info->window, "Can't reallocate set's associations!", "Error" );
								}
								break;
							}
#endif
							case 4:{
							  Pixel tp;
								ActiveWin= theWin_Info;
								if( strcasecmp( data_legend_buf, "default")== 0 ||
									(strncasecmp( data_legend_buf, "default", 7)== 0 && isspace(data_legend_buf[7]))
								){
									xfree( AllSets[data_sn_number].pixelCName );
									AllSets[data_sn_number].pixvalue= data_sn_number % MAXATTR;
									AllSets[data_sn_number].pixelValue= AllAttrs[data_sn_number % MAXATTR].pixelValue;
								}
								else if( GetColor( data_legend_buf, &tp) ){
									if( AllSets[data_sn_number].pixvalue< 0 ){
										FreeColor( &AllSets[data_sn_number].pixelValue, &AllSets[data_sn_number].pixelCName );
									}
									StoreCName( AllSets[data_sn_number].pixelCName );
									AllSets[data_sn_number].pixelValue= tp;
									AllSets[data_sn_number].pixvalue= -1;
								}
								break;
							}
							case 5:{
							  Pixel tp;
								ActiveWin= theWin_Info;
								if( strcasecmp( data_legend_buf, "default")== 0 ||
									(strncasecmp( data_legend_buf, "default", 7)== 0 && isspace(data_legend_buf[7]))
								){
									xfree( theWin_Info->legend_line[data_sn_number].pixelCName );
									theWin_Info->legend_line[data_sn_number].pixvalue= 0;
								}
								else if( GetColor( data_legend_buf, &tp) ){
									if( theWin_Info->legend_line[data_sn_number].pixvalue< 0 ){
										FreeColor( &theWin_Info->legend_line[data_sn_number].pixelValue, 
											&theWin_Info->legend_line[data_sn_number].pixelCName
										);
									}
									StoreCName( theWin_Info->legend_line[data_sn_number].pixelCName );
									theWin_Info->legend_line[data_sn_number].pixelValue= tp;
									theWin_Info->legend_line[data_sn_number].pixvalue= -1;
								}
								break;
							}
							case 6:
								xfree( AllSets[data_sn_number].process.description );
								AllSets[data_sn_number].process.description= XGstrdup( cleanup(data_legend_buf) );
								break;
							case 7:
								Free_LabelsList( AllSets[data_sn_number].ColumnLabels );
								AllSets[data_sn_number].ColumnLabels= Parse_SetLabelsList( NULL, data_legend_buf, '\t', 0, NULL );
								break;
							case 0:
								xfree( AllSets[data_sn_number].setName );
/* 								AllSets[data_sn_number].setName= String_ParseVarNames( XGstrdup( cleanup(data_legend_buf) ), 	*/
/* 									"%[", ']', True, True, "dialog_s.c::SD_snl_fun()"	*/
/* 								);	*/
								AllSets[data_sn_number].setName= XGstrdup( cleanup(data_legend_buf) );
								break;
							default:
								Boing(10);
								break;
						}
						theWin_Info->printed= 0;
					}
					else if( val== &new_set ){
						if( theWin_Info->labels_in_legend ){
							xfree( AllSets[data_sn_number].YUnits );
							AllSets[data_sn_number].YUnits= XGstrdup( parse_codes(fileName_buf) );
						}
						else{
							xfree( AllSets[data_sn_number].fileName );
							AllSets[data_sn_number].fileName= XGstrdup( parse_codes(fileName_buf) );
						}
						theWin_Info->printed= 0;
					}
				}
				else{
				  UserLabel *ul= GetULabelNr( data_sn_number- setNumber);
					if( ul ){
						switch( LegendFunctionNr ){
							default:
								strncpy( ul->label, data_legend_buf, MAXBUFSIZE );
								ul->pnt_nr= -1;
								theWin_Info->printed= 0;
								break;
							case 4:{
							  Pixel tp;
								ActiveWin= theWin_Info;
								if( strcasecmp( data_legend_buf, "default")== 0 ||
									(strncasecmp( data_legend_buf, "default", 7)== 0 && isspace(data_legend_buf[7]))
								){
									xfree( ul->pixelCName );
									ul->pixvalue= 0;
									ul->pixlinked= 0;
									ul->pixelValue= AllAttrs[ul->pixvalue].pixelValue;
									theWin_Info->printed= 0;
								}
								else if( strcasecmp( data_legend_buf, "linked")== 0 ||
									(strncasecmp( data_legend_buf, "linked", 6)== 0 && isspace(data_legend_buf[6]))
								){
									if( ul->set_link>= 0 && ul->set_link< setNumber ){
										xfree( ul->pixelCName );
										ul->pixvalue= 0;
										ul->pixlinked= 1;
									}
									else{
									  /* Don't change anything..!	*/
										xtb_error_box( theWin_Info->window, "Label isn't linked to a valid set", "Error" );
									}
									theWin_Info->printed= 0;
								}
								else if( GetColor( data_legend_buf, &tp) ){
									if( ul->pixvalue< 0 ){
										FreeColor( &ul->pixelValue, &ul->pixelCName );
									}
									StoreCName( ul->pixelCName );
									ul->pixelValue= tp;
									ul->pixvalue= -1;
									ul->pixlinked= 0;
									theWin_Info->printed= 0;
								}
								break;
							}
						}
					}
				}
				  /* see if the Dialogue box must be changed too. Finally we
				   \ can get rid of such a ridiculously large window caused
				   \ by e.g. a meta-concatenated YLabel just by shortening
				   \ the culprit.
				   */
				do_update_SD_size= 1;
			}
			  /* check lineWidth etc.	*/
			if( data_sn_number>= 0 && data_sn_number< setNumber ){
				if( val== &new_legend ){
					xtb_ti_get( AF(SNLW_F).win, data_sn_lineWidth_buf, (xtb_data *) 0);
					if( sscanf(data_sn_lineWidth_buf, "%lf", &data_sn_lineWidth) != 1) {
						do_error("Warning: can't read lineWidth\n");
					}
					else if( AllSets[data_sn_number].lineWidth!= data_sn_lineWidth ){
						theWin_Info->redraw= 1;
						theWin_Info->printed= 0;
						AllSets[data_sn_number].lineWidth= data_sn_lineWidth;
					}
					xtb_ti_get( AF(SNLS_F).win, data_sn_linestyle_buf, (xtb_data *) 0);
					if( sscanf(data_sn_linestyle_buf, "%d", &data_sn_linestyle) != 1) {
						do_error("Warning: can't read linestyle\n");
					}
					else{
						if( data_sn_linestyle< 0 ){
							data_sn_linestyle= 0;
						}
						if( AllSets[data_sn_number].linestyle!= data_sn_linestyle ){
							theWin_Info->redraw= 1;
							theWin_Info->printed= 0;
							AllSets[data_sn_number].linestyle= data_sn_linestyle;
						}
					}
					xtb_ti_get( AF(SNSMS_F).win, data_sn_markstyle_buf, (xtb_data *) 0);
					if( sscanf(data_sn_markstyle_buf, "%d", &data_sn_markstyle) != 1 || data_sn_markstyle<= 0 ){
						do_error("Warning: illegal markstyle\n");
					}
					else if( abs(AllSets[data_sn_number].markstyle)!= abs(data_sn_markstyle) ){
						theWin_Info->redraw= 1;
						theWin_Info->printed= 0;
						AllSets[data_sn_number].markstyle= -1 * abs(data_sn_markstyle);
					}
					xtb_ti_get( AF(SNELW_F).win, data_sn_elineWidth_buf, (xtb_data *) 0);
					if( sscanf(data_sn_elineWidth_buf, "%lf", &data_sn_elineWidth) != 1) {
						do_error("Warning: can't read errorbar lineWidth\n");
					}
					else if( AllSets[data_sn_number].elineWidth!= data_sn_elineWidth ){
						theWin_Info->redraw= 1;
						theWin_Info->printed= 0;
						AllSets[data_sn_number].elineWidth= data_sn_elineWidth;
					}
					xtb_ti_get( AF(SNELS_F).win, data_sn_elinestyle_buf, (xtb_data *) 0);
					if( sscanf(data_sn_elinestyle_buf, "%d", &data_sn_elinestyle) != 1) {
						do_error("Warning: can't read errorbar linestyle\n");
					}
					else{
						if( data_sn_elinestyle< 0 ){
							data_sn_elinestyle= 0;
						}
						if( AllSets[data_sn_number].elinestyle!= data_sn_elinestyle ){
							theWin_Info->redraw= 1;
							theWin_Info->printed= 0;
							AllSets[data_sn_number].elinestyle= data_sn_elinestyle;
						}
					}
					{ char bpbuf[D_SN];
					  double v;
						  /* 990506: I think this piece of code ain't really necessary.
						   \ As long as the b.pars fields are not included in the TAB
						   \ "next win procession"(?)
						   */
						xtb_ti_get( AF(SNBPB_F).win, bpbuf, NULL );
						if( sscanf( bpbuf, "%lf", &v)!= 1 ){
						}
					}
					xtb_ti_get( AF(SNPLI_F).win, data_sn_plot_interval_buf, (xtb_data *) 0);
					if( sscanf(data_sn_plot_interval_buf, "%d", &data_sn_plot_interval) != 1) {
						do_error("Warning: can't read plot_intervaln");
					}
					else if( AllSets[data_sn_number].plot_interval!= data_sn_plot_interval ){
						theWin_Info->redraw= 1;
						theWin_Info->printed= 0;
						AllSets[data_sn_number].plot_interval= data_sn_plot_interval;
					}
					xtb_ti_get( AF(SNAdI_F).win, data_sn_adorn_interval_buf, (xtb_data *) 0);
					if( sscanf(data_sn_adorn_interval_buf, "%d", &data_sn_adorn_interval) != 1) {
						do_error("Warning: can't read adorn_intervaln");
					}
					else if( AllSets[data_sn_number].adorn_interval!= data_sn_adorn_interval ){
						theWin_Info->redraw= 1;
						theWin_Info->printed= 0;
						AllSets[data_sn_number].adorn_interval= data_sn_adorn_interval;
					}
				}
			}
			else if( data_sn_number>= 0 && data_sn_number< setNumber + theWin_Info->ulabels ){
			  UserLabel *ul= GetULabelNr(data_sn_number-setNumber);
				if( ul && val== &new_legend ){
					xtb_ti_get( AF(SNLW_F).win, data_sn_lineWidth_buf, (xtb_data *) 0);
					if( sscanf(data_sn_lineWidth_buf, "%lf", &data_sn_lineWidth) != 1) {
						do_error("Warning: can't read lineWidth\n");
					}
					else if( ul->lineWidth!= data_sn_lineWidth ){
						theWin_Info->redraw= 1;
						theWin_Info->printed= 0;
						ul->lineWidth= data_sn_lineWidth;
					}
				}
			}

			if( ((ch== TAB || ch== XK_Tab) && !xtb_clipboard_copy) ){
				win= AF(SNNR_F).win;
				if( data_sn_number< setNumber+ theWin_Info->ulabels- 1){
					data_sn_number++;
				}
				else{
					data_sn_number= SNNR_MINVAL;
				}
			}
			switch( data_sn_number ){
				case -22:
					sprintf( data_sn_number_buf, "COL" );
					new_file= 0;
					break;
				case -21:
					sprintf( data_sn_number_buf, "TXT" );
					new_file= 0;
					break;
				case -20:
					sprintf( data_sn_number_buf, "ARG" );
					new_file= 0;
					break;
				case -19:
					sprintf( data_sn_number_buf, "ERA" );
					new_file= 0;
					break;
				case -18:
					sprintf( data_sn_number_buf, "LRA" );
					new_file= 0;
					break;
				case -17:
					sprintf( data_sn_number_buf, "DUB" );
					new_file= 0;
					break;
				case -16:
					sprintf( data_sn_number_buf, "DUA" );
					new_file= 0;
					break;
				case -15:
					sprintf( data_sn_number_buf, "BD" );
					new_file= 0;
					break;
				case -14:
					sprintf( data_sn_number_buf, "AD" );
					new_file= 0;
					break;
				case -13:
					sprintf( data_sn_number_buf, "RF" );
					new_file= 0;
					break;
				case -12:
					sprintf( data_sn_number_buf, "EN" );
					new_file= 0;
					break;
				case -11:
					sprintf( data_sn_number_buf, "TX" );
					new_file= 0;
					break;
				case -10:
					sprintf( data_sn_number_buf, "TY" );
					new_file= 0;
					break;
				case -9:
					sprintf( data_sn_number_buf, "DI" );
					new_file= 0;
					break;
				case -8:
					sprintf( data_sn_number_buf, "DB" );
					new_file= 0;
					break;
				case -7:
					sprintf( data_sn_number_buf, "DP" );
					new_file= 0;
					break;
				case -6:
					sprintf( data_sn_number_buf, "DA" );
					new_file= 0;
					break;
				case -5:
					sprintf( data_sn_number_buf, "DF" );
					new_file= 0;
					break;
				case -4:
					sprintf(data_sn_number_buf, "-T");
					new_file= 0;
					break;
				case -3:
					sprintf(data_sn_number_buf, "-t");
					new_file= 0;
					break;
				case -2:
					sprintf(data_sn_number_buf, "X");
					new_file= 0;
					break;
				case -1:
					sprintf(data_sn_number_buf, "Y");
					new_file= 0;
					break;
				default:
					sprintf(data_sn_number_buf, "%d", data_sn_number);
					if( data_sn_number< setNumber ){
						new_file= theWin_Info->new_file[data_sn_number];
					}
					else{
						new_file= 0;
					}
					break;
			}
			fileName= Data_fileName();
			xtb_ti_set( AF(SNNR_F).win, data_sn_number_buf, (xtb_data) 0);
/* 			xtb_bt_set( AF(SNFNS_F).win, new_file, (xtb_data) 0);	*/
/* 			xtb_ti_set( AF(SNFN_F).win, fileName, (xtb_data) 0);	*/
			set_changeables(0,True);
			if( (ch== TAB && !xtb_clipboard_copy) ){
#ifdef TAB_WARPS
				x_loc= AF(SNNR_F).width/2;
				y_loc= AF(SNNR_F).height/2;
				XWarpPointer( disp, None, win, 0, 0, 0, 0, x_loc, y_loc);
#else
				XSetInputFocus( disp, win, RevertToParent, CurrentTime);
#endif
				set_changeables(2,True);
			}
		}
		else{
			Boing( 5);
		}
	}
	else if( ch== CONTROL_P){
		win= AF(SNNR_F).win;
#ifdef TAB_WARPS
		x_loc= AF(SNNR_F).width/2;
		y_loc= AF(SNNR_F).height/2;
		XWarpPointer( disp, None, win, 0, 0, 0, 0, x_loc, y_loc);
#else
		XSetInputFocus( disp, win, RevertToParent, CurrentTime);
#endif
	}
	else if( ch== CONTROL_W){
	  char *str;
		if( *text){
			str= &text[ strlen(text)-1 ];
			legend_changed= 1;
		}
		else{
			Boing( 5);
			return( XTB_HANDLED);
		}
		if( index( word_sep, *str) ){
			if( !xtb_ti_dch(win) ){
				Boing( 5);
			}
			legend_changed= 1;
			return( XTB_HANDLED);
		}
		while( *str && !index( word_sep, *str) ){
			if( !xtb_ti_dch(win) ){
				Boing( 5);
				return( XTB_HANDLED);
			}
			legend_changed= 1;
			str--;
		}
	}
	else if( ch!= XK_Meta_L && ch!= XK_Meta_R && ch!= XK_Alt_L && ch!= XK_Alt_R &&
		ch!= XK_Super_L && ch!= XK_Super_R && ch!= XK_Hyper_L && ch!= XK_Hyper_R
	){
	  /* Insert if printable - ascii dependent */
		if( /* (ch < ' ') || (ch >= DELETE) || */ !xtb_ti_ins(win, ch)) {
			Boing( 5);
			if( debugFlag ){
				fprintf( StdErr, "SD_snl_fun(): bad char '%c' (0x%02x)\n", ch, (int) ch);
				fflush( StdErr );
			}
		}
		legend_changed= 1;
    }
    return XTB_HANDLED;
}

void Data_SN_Number( char *number )
{
	switch( data_sn_number ){
		case -22:
			strcpy( number, "COL" );
			break;
		case -21:
			strcpy( number, "TXT" );
			break;
		case -20:
			strcpy( number, "ARG" );
			break;
		case -19:
			strcpy( number, "ERA" );
			break;
		case -18:
			strcpy( number, "LRA" );
			break;
		case -17:
			strcpy( number, "DUB" );
			break;
		case -16:
			strcpy( number, "DUA" );
			break;
		case -15:
			strcpy( number, "BD" );
			break;
		case -14:
			strcpy( number, "AD" );
			break;
		case -13:
			strcpy( number, "RF" );
			break;
		case -12:
			strcpy( number, "EN" );
			break;
		case -11:
			strcpy( number, "TX" );
			break;
		case -10:
			strcpy( number, "TY" );
			break;
		case -9:
			strcpy( number, "DI" );
			break;
		case -8:
			strcpy( number, "DB" );
			break;
		case -7:
			strcpy( number, "DP" );
			break;
		case -6:
			strcpy( number, "DA" );
			break;
		case -5:
			strcpy( number, "DF" );
			break;
		case -4:
			strcpy( number, "-T");
			break;
		case -3:
			strcpy( number, "-t");
			break;
		case -2:
			strcpy( number, "X");
			break;
		case -1:
			strcpy( number, "Y");
			break;
		default:
			sprintf( number, "%d", data_sn_number );
			break;
	}
}

int Data_SN_Label( char *number )
{
	if( SD_strlen( number ) ){
		if( !strcasecmp( number, "COL" ) ){
			data_sn_number= -22;
		}
		else if( !strcasecmp( number, "TXT" ) ){
			data_sn_number= -21;
		}
		else if( !strcasecmp( number, "ARG" ) ){
			data_sn_number= -20;
		}
		else if( !strcasecmp( number, "ERA" ) ){
			data_sn_number= -19;
		}
		else if( !strcasecmp( number, "LRA" ) ){
			data_sn_number= -18;
		}
		else if( !strcasecmp( number, "DUB" ) ){
			data_sn_number= -17;
		}
		else if( !strcasecmp( number, "DUA" ) ){
			data_sn_number= -16;
		}
		else if( !strcasecmp( number, "BD" ) ){
			data_sn_number= -15;
		}
		else if( !strcasecmp( number, "AD" ) ){
			data_sn_number= -14;
		}
		else if( !strcasecmp( number, "RF" ) ){
			data_sn_number= -13;
		}
		else if( !strcasecmp( number, "EN") || !strcasecmp( number, "PN" ) ){
			data_sn_number= -12;
		}
		else if( !strcasecmp( number, "TX" ) ){
			data_sn_number= -11;
		}
		else if( !strcasecmp( number, "TY" ) ){
			data_sn_number= -10;
		}
		else if( !strcasecmp( number, "DI" ) ){
			data_sn_number= -9;
		}
		else if( !strcasecmp( number, "DB" ) ){
			data_sn_number= -8;
		}
		else if( !strcasecmp( number, "DP" ) ){
			data_sn_number= -7;
		}
		else if( !strcasecmp( number, "DA" ) ){
			data_sn_number= -6;
		}
		else if( !strcasecmp( number, "DF" ) ){
			data_sn_number= -5;
		}
		else if( !strcmp( number, "-T" ) ){
			data_sn_number= -4;
		}
		else if( !strcmp( number, "-t" ) ){
			data_sn_number= -3;
		}
		else if( index( "xX", number[0] ) ){
			data_sn_number= -2;
		}
		else if( index( "yY", number[0] ) ){
			data_sn_number= -1;
		}
		else{
			return( -1 );
		}
		return(1);
	}
	return(0);
}

char *SNN_menu[]= {
		"   COL for \"*COLUMNLABELS* new\"\n",
		"   TXT for \"*READ* Text\"\n",
		"   ARG for *ARGUMENT* (easier than as shown under RF and TXT..)\n"
		"      e.g. *ARGUMENTS* -T bla to skip the |echo and the quotes\n"
		"      or *DRAW_SET* TITLE <pat> to draw matching sets\n",
		"   ERA for *ENTER_RAW_AFTER*\n",
		"   LRA for *LEAVE_RAW_AFTER*\n",
		"   DUB for *DUMP_BEFORE*\n",
		"   DUA for *DUMP_AFTER*\n",
		"   BD for *DRAW_BEFORE*\n",
		"   AD for *DRAW_AFTER* (DB & DA were used...)\n",
		"   RF for \"Read a New File\" (*READ_FILE*)\n"
		"      Use a '|' as first character to read from a pipe\n"
		"       (e.g. |echo \"*ARGUMENTS* -T bla\"\n"
		"            to define a new \"supertitle\")\n",
		"   EN for *EVAL* (was: *PARAM_NOW*; PN) \n",
		"   TX for *TRANSFORM_X*\n",
		"   TY for *TRANSFORM_Y*\n",
		"   DI for *DATA_INIT*\n",
		"   DB for *DATA_BEFORE*\n",
		"   DP for *DATA_PROCESS*\n",
		"   DA for *DATA_AFTER*\n",
		"   DF for *DATA_FINISH*\n",
		"   -T for the alternative (-T) titletext\n",
		"   -t for the (global, -t) titletext\n",
		"   X for X axis-label\n",
		"   Y for Y axis-label\n",
		"   0\n",
		"   XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n"
};

static xtb_hret snn_fun(Window win, int ch, char *text, xtb_data val)
/*
 * This is the handler function for the number widget for
 * selecting a legend-number.  It supports simple
 * line editing operations.
 */
{
#ifdef TAB_WARPS
   int x_loc, y_loc;
#endif
   char *str, number[MAXCHBUF];

	if( !theWin_Info ){
		return( XTB_STOP );
	}

    if ((ch == BACKSPACE) || (ch == DELETE)) {
		if( !( (ch== DELETE)? xtb_ti_dch_right(win) : xtb_ti_dch(win)) )
			Boing( 5);
    }
	else if ((ch == CONTROL_U) || (ch == CONTROL_X)) {
		(void) xtb_ti_set(win, "", (xtb_data) 0);
    }
	else if( ch== CONTROL_P){
		return( goto_prev_text_box( win) );
	}
	else if( ch== CONTROL_W){
		if( *text){
			str= &text[ strlen(text)-1 ];
		}
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
	else if( ch!= TAB && ch!= XK_Down && ch != XK_Up && ch &&
		ch!= XK_Meta_L && ch!= XK_Meta_R && ch!= XK_Alt_L && ch!= XK_Alt_R &&
		ch!= XK_Super_L && ch!= XK_Super_R && ch!= XK_Hyper_L && ch!= XK_Hyper_R
	){
	  /* Insert if acceptable for function selection */
		if( !( index( SNLEG_CHARACTERS, ch) || Isdigit(ch) ) || !xtb_ti_ins(win, ch)) {
			Boing( 5);
		}
    }

	xtb_ti_get( win, number, NULL);
	if( strlen( number ) ){
		if( Data_SN_Label( number )== -1 ){
		  int d;
			if( sscanf( number, "%d", &d)!= 1 ){
				data_legend_buf[0]= '\0';
				xtb_ti_set( AF(SNLEG_F).win, data_legend_buf, (xtb_data) 0 );
				return( XTB_HANDLED);
			}
			else if( d< 0 || d>= setNumber+ theWin_Info->ulabels ){
				data_legend_buf[0]= '\0';
				xtb_ti_set( AF(SNLEG_F).win, data_legend_buf, (xtb_data) 0 );
				Boing( 5);
				return( XTB_HANDLED);
			}
			data_sn_number= d;
		}
		set_changeables(0,True);
		if( debugFlag ){
			fprintf( StdErr, "Legend \"%s\" = #%d\n", number, data_sn_number );
		}
		if( !xtb_ti_set( AF(SNLEG_F).win, data_legend_buf, (xtb_data) 0 ) ){
		  char mesg[LEGEND+32];
			sprintf( mesg, "Can't set legend #%d\n(%s)\n",
				data_sn_number, data_legend_buf
			);
			do_error( mesg);
		}
	}

	if( ((ch== TAB || ch== XK_Tab) && !xtb_clipboard_copy) ){
		(CheckMask(xtb_modifier_state, ShiftMask))? goto_prev_text_box(win) : goto_next_text_box( &win);
	}
	else if( ch== XK_Down || ch== XK_Up ){
	  /* Quick hack-in of cursor based increment/decrement.	*/
		switch( ch ){
			case XK_Down:
				if( data_sn_number> SNNR_MINVAL ){
					data_sn_number-= 1;
				}
				else{
					Boing( 5 );
				}
				break;
			case XK_Up:
				if( data_sn_number< setNumber+ theWin_Info->ulabels - 1 ){
					data_sn_number+= 1;
				}
				else{
					Boing( 5 );
				}
				break;
		}
		Data_SN_Number( number );
		xtb_ti_set( win, number, NULL);
		  /* call SELF to update all those other fields.
		   \ Pass a nullbyte as input character.
		   */
		snn_fun( win, 0, text, val);
	}

	xtb_sr_set( AF(SNNRSL_F).win, (double) data_sn_number, NULL);

    return XTB_HANDLED;
}

/* routine that handles the (extra) legend-slider	*/
xtb_hret snn_slide_f( Window win, int pos, double val, xtb_data info)
{  char buf[256];
	sprintf( buf, "%d", (int) val );
	xtb_ti_set( AF(SNNR_F).win, buf, NULL);
	snn_fun( AF(SNNR_F).win, 0, buf, NULL);
	return( XTB_HANDLED);
}

static xtb_hret sSD_dfn_fun( Window win, int bval, xtb_data info)
/* 
 \ swaps new_file field of current data_sn_number dataset
 */
{  int nr, N, i, ard= autoredraw;
   extern double newfile_incr_width;
   Boolean aaa= False;
	if( !theWin_Info ){
		return( XTB_STOP );
	}

	if( data_sn_number< setNumber && (apply_to_marked || apply_to_hlt || apply_to_drawn || apply_to_new || apply_to_src ) ){
		_apply_to_drawn= apply_to_drawn;
		apply_to_drawn= 0;
		_apply_to_marked= apply_to_marked;
		apply_to_marked= 0;
		_apply_to_hlt= apply_to_hlt;
		apply_to_hlt= 0;
		_apply_to_new= apply_to_new;
		apply_to_new= 0;
		_apply_to_src= apply_to_src;
		apply_to_src= 0;
		apply_to_all= aaa= True;
	}
	if( apply_to_all && data_sn_number< setNumber ){
		nr= (apply_to_rest)? data_sn_number : 0;
		N= (apply_to_prev)? MIN(data_sn_number+1,setNumber) : setNumber;
		apply_to_all= 0;
		apply_to_rest= 0;
		apply_to_prev= 0;
		apply_to_src= 0;
		xtb_bt_set( AF(SNATA_F).win, 0, NULL );
		xtb_bt_set( AF(SNATP_F).win, 0, NULL );
		xtb_bt_set( AF(SNATR_F).win, 0, NULL );
		autoredraw= False;
	}
	else{
		nr= data_sn_number;
		N= nr+ 1;
	}
	bval= !bval;
	for( ; nr< N; nr++ ){
	  UserLabel *ul;
		if( nr>= 0 && nr< setNumber && apply_ok(nr) ){
			xtb_sr_set( AF(SNNRSL_F).win, (double) nr, NULL);
			theWin_Info->new_file[nr]= bval;
			theWin_Info->redraw= 1;
			if( theWin_Info->redrawn== -3 ){
				theWin_Info->redrawn= 0;
			}
			theWin_Info->printed= 0;
			  /* update lineWidth's if necessary	*/
			if( bval && newfile_incr_width ){
				for( i= nr; i< setNumber; i++ ){
					AllSets[i].lineWidth+= newfile_incr_width;
					AllSets[i].fileNumber+= 1;
					theWin_Info->fileNumber[i]+= 1;
				}
			}
			else if( newfile_incr_width ){
				for( i= nr; i< setNumber; i++ ){
					if( AllSets[i].lineWidth> newfile_incr_width ){
						AllSets[i].lineWidth-= newfile_incr_width;
						AllSets[i].fileNumber-= 1;
						theWin_Info->fileNumber[i]-= 1;
					}
				}
			}
		}
		else if( (ul= GetULabelNr( nr- setNumber)) ){
			ul->nobox= bval;
			xtb_sr_set( AF(SNNRSL_F).win, (double) nr, NULL);
			theWin_Info->redraw= 1;
			if( theWin_Info->redrawn== -3 ){
				theWin_Info->redrawn= 0;
			}
			theWin_Info->printed= 0;
		}
	}
	if( aaa ){
		_apply_to_drawn= 0;
		_apply_to_marked= 0;
		_apply_to_hlt= 0;
		_apply_to_new= 0;
		_apply_to_src= 0;
		xtb_bt_set( AF(SNATD_F).win, 0, NULL );
		xtb_bt_set( AF(SNATM_F).win, 0, NULL );
		xtb_bt_set( AF(SNATH_F).win, 0, NULL );
		xtb_bt_set( AF(SNATN_F).win, 0, NULL );
		xtb_bt_set( AF(SNATS_F).win, 0, NULL );
	}
	autoredraw= ard;
	if( data_sn_number>= setNumber+ theWin_Info->ulabels ){
		Boing( 5);
	}
	else if( data_sn_number>= 0 ){
		set_changeables(0,True);
	}
	return( XTB_HANDLED );
}

static int set_draw_set= 1, set_mark_set, set_highlight, set_show_legend= 2, set_show_llines,
	set_no_legend= 4, set_no_title= 8,
	set_noLines= 32, set_floating, set_barFlag= 64, set_markFlag= 128, set_pixelMarks= 256,
	set_legend_placed= 512, set_show_errors, set_overwrite_marks= 1024, set_filename_in_legend,
	set_bbFlag, set_zeroFlag, set_xname_placed, set_yname_placed, user_coordinates, nobb_coordinates;
extern int use_X11Font_length, yname_vertical;
extern int filename_in_legend, labels_in_legend, axisFlag, bbFlag, zeroFlag, htickFlag, vtickFlag;
extern double overlap();
extern int Allow_Fractions;
extern Boolean changed_Allow_Fractions;

extern char *comment_buf;
extern int comment_size;

xtb_hret display_ascanf_variables_h(Window win, int bval, xtb_data info)
{
	display_ascanf_variables( win, CheckMask(xtb_modifier_state, Mod1Mask), True, NULL );
	xtb_bt_set( win, 0, NULL);
	return( XTB_HANDLED );
}

xtb_hret SD_set_info(Window win, int bval, xtb_data info)
{
	if( !theWin_Info ){
		return( XTB_STOP );
	}

	if( data_sn_number< setNumber ){
	  char tt[256];
	  char *infobuf= (data_sn_number>= 0)? AllSets[data_sn_number].set_info : theWin_Info->version_list;
		errno= 0;
		if( data_sn_number< 0 ){
			sprintf( tt, "Global info/version_list:\n" );
		}
		else{
			sprintf( tt, "Info for set #%d:\n", data_sn_number );
		}
		if( CheckMask( xtb_modifier_state, Mod1Mask) ){
		  int si_len= (infobuf)? strlen(infobuf) : 0;
		  _ALLOCA( ibuf, char, (int)((si_len)? 1.5* si_len : 256), ib_len);
		  char *nbuf;
		  extern xtb_hret xtb_input_dialog_parse_codes();
			if( infobuf ){
				strcpy( ibuf, infobuf);
			}
			else{
				ibuf[0]= '\0';
			}
			if( (nbuf= xtb_input_dialog( SD_Dialog.win, ibuf, 100, ib_len/sizeof(char),
					"Edit the set-info; control-click the input field for an overview\n"
					" Use the [Parse] button to translate opcodes in the text\n",
					tt,
					False,
					"Parse", xtb_input_dialog_parse_codes,
					"Edit", SimpleEdit_h,
					NULL, NULL
				))
			){
				if( data_sn_number< 0 ){
				  char *pinfo= NULL, *vinfo;
				  int l;
					  /* Obtain a string with the current process info: */
					_Dump_Arg0_Info( theWin_Info, NULL, &pinfo, False );
					  /* We must prepend a newline to it: */
					l= strlen(pinfo)* sizeof(char);
					if( (pinfo= _XGrealloc( &pinfo, (strlen(pinfo)+2)* sizeof(char), NULL, NULL)) ){
						memmove( &pinfo[1], pinfo, l );
						pinfo[0]= '\n';
					}
					  /* Now we can use it with SubstituteOpcodes to replace the desired pattern in nbuf: */
					vinfo= SubstituteOpcodes( nbuf, "\n*PINFO*\n", pinfo, NULL );
					xfree( theWin_Info->version_list );
					  /* SubstituteOpcodes will have returned nbuf if nothing was replaced (so we must strdup), or
					   \ a new pointer that we can just copy (and free elsewhere when required):
					   */
					theWin_Info->version_list= (vinfo== nbuf)? strdup(nbuf) : vinfo;
					  /* Anyhow, pinfo can be freed here: */
					xfree( pinfo );
				}
				else{
					xfree( AllSets[data_sn_number].set_info );
					AllSets[data_sn_number].set_info= strdup(nbuf);
				}
				if( nbuf!= ibuf ){
					  /* must free nbuf here; xtb_input_dialog() allocated it. */
					xfree( nbuf );
				}
			}
			GCA();
		}
		else if( infobuf ){
		  int id;
		  char *sel= NULL;
		  xtb_frame *menu= NULL;
			id= xtb_popup_menu( theWin_Info->window, infobuf, tt, &sel, &menu);

			if( sel ){
				while( *sel && isspace( (unsigned char) *sel) ){
					sel++;
				}
			}
			if( sel && *sel ){
				if( debugFlag ){
					xtb_error_box( theWin_Info->window, sel, "Copied to clipboard:" );
				}
				else{
					Boing(10);
				}
				XStoreBuffer( disp, sel, strlen(sel), 0);
				 // RJVB 20081217
				xfree(sel);
			}
			xtb_popup_delete( &menu );
		}
	}
	else{
		Boing(0);
	}
	xtb_bt_set( win, 0, NULL);
	return( XTB_HANDLED );
}

xtb_hret SD_set_bardimensions(Window win, int bval, xtb_data info)
{ char ibuf[256], *nbuf;
	if( !theWin_Info ){
		return( XTB_STOP );
	}

	errno= 0;
	sprintf( ibuf, "%s, %s, %s",
		d2str( theWin_Info->bar_legend_dimension_weight[0], d3str_format, NULL ),
		d2str( theWin_Info->bar_legend_dimension_weight[1], d3str_format, NULL ),
		d2str( theWin_Info->bar_legend_dimension_weight[2], d3str_format, NULL )
	);
	if( (nbuf= xtb_input_dialog( SD_Dialog.win, ibuf, strlen(ibuf), sizeof(ibuf)/sizeof(char),
			"Enter the weights to be applied for drawing legends of barplots;\n"
			" X,Y,M where\n"
			" X is a weight on the standard PostScript marker width (-PSm), determines the width\n"
			" Y is a weight on the legend text point size, determines the half-height (1 means twice as high as the text)\n"
			" M is a factor that determines the spacing margin in the legend\n"
			, "Callibration of barplot legend display",
			False,
			NULL, NULL, NULL, NULL, NULL, NULL
		))
	){
	  double bw[3];
	  int n= 3;
		memcpy( bw, theWin_Info->bar_legend_dimension_weight, sizeof(bw) );
		if( fascanf( &n, nbuf, bw, NULL, NULL, NULL, NULL)>= 1 ){
			memcpy( theWin_Info->bar_legend_dimension_weight, bw, sizeof(bw) );
		}
		if( nbuf!= ibuf ){
			xfree( nbuf );
		}
	}
	xtb_bt_set( win, 0, NULL);
	return( XTB_HANDLED );
}

xtb_hret SD_process_hist(Window win, int bval, xtb_data info)
{
	if( !theWin_Info ){
		return( XTB_STOP );
	}

	process_hist( win );
	xtb_bt_set( win, 0, NULL);
	return( XTB_HANDLED );
}

xtb_hret SD_option_hist(Window win, int bval, xtb_data info)
{
	if( !theWin_Info ){
		return( XTB_STOP );
	}

	option_hist( theWin_Info );
	xtb_bt_set( win, 0, NULL);
	return( XTB_HANDLED );
}

xtb_hret SD_help_fnc(Window win, int bval, xtb_data info)
{
	if( !theWin_Info ){
		return( XTB_STOP );
	}

	help_fnc( theWin_Info, CheckMask( xtb_modifier_state, Mod1Mask), True );
	xtb_bt_set( win, 0, NULL);
	return( XTB_HANDLED );
}

xtb_hret SD_selectfun(Window win, int bval, xtb_data info)
{ int id, n= 0, N= 0, idx= data_sn_number+ ABS(SNNR_MINVAL);
  char *sel= NULL, name[5], *c, **snn_menu;
  xtb_frame *menu= NULL;
	if( !theWin_Info ){
		return( XTB_STOP );
	}
	  /* 20020424: xtb_CharArray() takes a list of pairs <elements>,<text> and creates (allocates) an
	   \ array of strings out of it. Of this array (the function's return value), the 1st
	   \ element (array[0]) points to a text buffer containing all strings concatenated;
	   \ each following element points to the start in that buffer of the next text string
	   \ passed in the arguments. The first argument should be a pointer to an integer that will
	   \ return the number of entries in the returned array. If the 2nd argument is True, each
	   \ text string will be terminated, such that each array[i] element represents exactly 1
	   \ of the specified strings. When this argument is False, the strings are not terminated;
	   \ thence, array[0] points to a full concatenation of all the specified strings, array[1]
	   \ to the last N-1, and so forth.
	   \ When elements is positive, <text> is supposed to be of type char**; an array of strings.
	   \ When elements is negative, <text> is supposed to be a single string, a char*.
	   */
	snn_menu= xtb_CharArray( &N, False, sizeof(SNN_menu)/sizeof(char*), SNN_menu, 0 );
	if( (c= strstr( snn_menu[0], "\n   0\n   XXXXX")) ){
		c++;
		sprintf( c, "   0\n   %d\n", setNumber-1 );
	}
	if( N>= ABS(SNNR_MINVAL)+ 2 ){
		if( idx>= N ){
			snn_menu[ABS(SNNR_MINVAL)+1][1]= '>';
		}
		else if( idx>= 0 ){
			snn_menu[idx][1]= '>';
		}
		else{
			fprintf( StdErr, "SD_selectfun(): data_sn_number+ABS(SNNR_MINVAL)==%d+ABS(%d)<0: please report!\n",
				data_sn_number, SNNR_MINVAL
			);
		}
	}
	id= xtb_popup_menu( win, snn_menu[0], "Select one of:", &sel, &menu);

	if( sel && (sel[0]== ' ' || sel[1]== '>') ){
	  int set;
	  char *s= sel;
		c= name;
		while( isspace(*s) || *s== '>' ){
			s++;
		}
		if( isdigit(s[0]) && (set= atoi(s))>= 0 && set< setNumber ){
			if( debugFlag ){
				xtb_error_box( theWin_Info->window, s, "Setting selector to:" );
			}
			else{
				Boing(10);
			}
			xtb_ti_set( AF(SNNR_F).win, s, NULL);
			snn_fun( AF(SNNR_F).win, 0, s, NULL);
		}
		else{
			while( *s!= ' ' && n< 4 ){
				*c++ = *s++;
				n++;
			}
			*c= '\0';
			if( *s== ' ' && strncmp( &s[1], "for ", 4 )== 0 ){
				if( debugFlag ){
					xtb_error_box( theWin_Info->window, name, "Setting selector to:" );
				}
				else{
					Boing(10);
				}
				xtb_ti_set( AF(SNNR_F).win, name, NULL);
				snn_fun( AF(SNNR_F).win, 0, name, NULL);
			}
		}
		xfree( sel );
	}
	xtb_popup_delete( &menu );
	xfree( snn_menu[0] );
	xfree( snn_menu );

	xtb_bt_set( win, 0, NULL);
	return( XTB_HANDLED );
}

char *Vector_menu[]= {
		" Select one of the following types (click on the type number):\n",
		"   0: parameter: <length>; a simple line that starts in the datapoint.\n",
		"   1: parameters: <length>,<front_fraction[1/3]>,<arrow_length[5]>;\n"
		"           an arrow that passes through the datapoint.\n"
		"           negative <length> inverses direction; <front_fraction> e.g. 1/3 (or 3!; default) means\n"
		"           that the arrowtip is at 1/3 of the vectorlength from the \"centre\"point;\n"
		"           <arrow_length> e.g. 5 (default) means the arrowhead's length is approx. 1/5\n"
		"           of the vectorlength.\n",
		"   2: <placeholder>\n"
		"           Like type 0, except that the length is taken from the lcol column ($DATA{3})\n",
		"   3: parameters: <placeholder>,<front_fraction[1/3]>,<arrow_length[5]>\n"
		"           Like type 1, except that the length is taken from the lcol column ($DATA{3})\n",
		"   4: parameters: <length>,<front_fraction[1/3]>,<arrow_length[5]>\n"
		"           Like type 4, except that the arrowhead's size is based on the <length> parameter.\n",
};

static char *UL_Type_menu[]= {
		" Select one of the following label types (click on the first line of the type description):\n",
		"   HL: a simple horizontal line\n",
		"   VL: a simple vertical line\n"
};


xtb_hret SD_vanes_or_ultype_selectfun(Window win, int bval, xtb_data info)
{ int id, N= 0, idx;
  char *sel= NULL, **vector_menu;
  xtb_frame *menu= NULL;
  DataSet *this_set;
  UserLabel *ul= GetULabelNr(data_sn_number-setNumber);

	if( !theWin_Info ){
		return( XTB_STOP );
	}
	if( data_sn_number< 0 ||
		(data_sn_number>= setNumber && 
			(!ul || ul->type== UL_regular)
		)
	){
		xtb_error_box( theWin_Info->window, "Not available for non-sets and \"regular\" UserLabels!", "Warning" );
		return( XTB_HANDLED );
	}
	if( ul ){
		vector_menu= xtb_CharArray( &N, False, sizeof(UL_Type_menu)/sizeof(char*), UL_Type_menu, 0 );
		switch( ul->type ){
			case UL_hline:
				vector_menu[1][1]= '>';
				break;
			case UL_vline:
				vector_menu[2][1]= '>';
				break;
		}
		id= xtb_popup_menu( win, vector_menu[0], "Select the desired type of the selected/current label:", &sel, &menu);
		if( sel ){
		  int set;
			for( set= N-1; set> 0; set-- ){
				if( strstr( vector_menu[set], sel) ){
					switch( set ){
						case 2:
							ul->type= UL_vline;
							theWin_Info->redraw= 1;
							break;
						case 1:
							ul->type= UL_hline;
							theWin_Info->redraw= 1;
							break;
					}
					set_changeables(2,True);
					set= -1;
				}
			}
			if( set>= 0 ){
				Boing(100);
			}
		}
	}
	else{
		vector_menu= xtb_CharArray( &N, False, sizeof(Vector_menu)/sizeof(char*), Vector_menu, 0 );
		this_set= &AllSets[data_sn_number];
		idx= this_set->vectorType+ 1;
		if( N>= MAX_VECTYPES+ 1 ){
			if( idx>= 1 && idx< N ){
				vector_menu[idx][1]= '>';
			}
			else{
				fprintf( StdErr, "SD_vanes_or_ultype_selectfun(): invalid or unknown vector type %d: please report!\n",
					this_set->vectorType
				);
			}
		}
		id= xtb_popup_menu( win, vector_menu[0], "Type of vectors plotted on sets in vector mode", &sel, &menu);

		if( sel ){
		  int set;
			for( set= N-1; set> 0; set-- ){
				  /* 20020424: each vector_menu[set] points to the onset of the corresponding entry in a continuous
				   \ stringbuffer (xtb_CharArray() was called with terminate==False). Thus, if user wants to click not
				   \ on an entries 1st line, we need to work backwards, finding the first entry that contains the returned
				   \ selection. (And then stop.) Doing this forward would match all entries up to the requested entry...
				   */
				if( strstr( vector_menu[set], sel) ){
					if( this_set->vectorType!= set-1 ){
						this_set->vectorType= set-1;
						theWin_Info->redraw= 1;
					}
					set_changeables(2,True);
					set= -1;
				}
			}
			if( set>= 0 ){
				Boing(100);
			}
		}
	}

	xfree( sel );
	xtb_popup_delete( &menu );
	xfree( vector_menu[0] );
	xfree( vector_menu );

	xtb_bt_set( win, 0, NULL);
	return( XTB_HANDLED );
}

xtb_hret SD_column_selectfun(Window win, int bval, xtb_data info)
{ int id, idx, *col;
  char *sel= NULL, *column_menu, *tmesg;
  xtb_frame *menu= NULL;
  DataSet *this_set;
  LabelsList *Labels;
	if( !theWin_Info ){
		return( XTB_STOP );
	}
	if( data_sn_number< 0 || data_sn_number>= setNumber ){
		xtb_error_box( theWin_Info->window, "Not available for non-sets!", "Warning" );
		return( XTB_HANDLED );
	}
	this_set= &AllSets[data_sn_number];
	if( !(Labels= this_set->ColumnLabels) && !(Labels= theWin_Info->ColumnLabels) ){
		xtb_error_box( theWin_Info->window,
			"Sorry, no column labels have been defined, neither for the current set (*LABELS* ..)\n"
			" nor globally (through *COLUMNLABELS* ..). You will just have to enter a column\n"
			" number in the entry box.\n",
			"Notice"
		);
		return( XTB_HANDLED );
	}
	if( info== &xcol ){
		col= &theWin_Info->xcol[data_sn_number];
		tmesg= "Select a column to be used for the X values:";
	}
	else if( info== &ycol ){
		col= &theWin_Info->ycol[data_sn_number];
		tmesg= "Select a column to be used for the Y values:";
	}
	else if( info== &ecol ){
		col= &theWin_Info->ecol[data_sn_number];
		tmesg= "Select a column to be used for the Error/Orientation/Intensity/MarkerSize values:";
	}
	else if( info== &lcol ){
		col= &theWin_Info->lcol[data_sn_number];
		tmesg= "Select a column to be used for the Vector Length values:";
	}
	else if( info== &Ncol ){
		col= &this_set->Ncol;
		tmesg= "Select a column to be used for the Nobs values:";
	}
	else{
		xtb_error_box( theWin_Info->window,
			"SD_column_selectfun() was called with an invalid/unknown info argument;\n"
			" please report\n",
			"Error"
		);
		return( XTB_HANDLED );
	}

	column_menu= NULL;
	for( idx= 0; idx< this_set->ncols; idx++ ){
	  char *c= Find_LabelsListLabel( Labels, idx ), num[64];
		sprintf( num, "%d", idx );
		if( !c ){
			c= num;
		}
		if( idx== *col ){
			column_menu= concat2( column_menu, " > ", NULL );
		}
		else{
			column_menu= concat2( column_menu, "   ", NULL );
		}
		if( c== num ){
			column_menu= concat2( column_menu, c, "\n", NULL );
		}
		else{
			column_menu= concat2( column_menu, num, " : ", c, "\n", NULL );
		}
	}

	xtb_bt_set( win, 1, NULL);
	id= xtb_popup_menu( win, column_menu, tmesg, &sel, &menu);

	if( sel ){
		if( id>= 0 && id< this_set->ncols ){
			if( *col!= id ){
				*col= id;
				theWin_Info->redraw= 1;
			}
			set_changeables(2,True);
		}
		else{
			Boing(100);
		}
		xfree( sel );
	}
	xtb_popup_delete( &menu );
	xfree( column_menu );

	xtb_bt_set( win, 0, NULL);
	return( XTB_HANDLED );
}

xtb_hret SD_markerStyle_selectfun(Window win, int bval, xtb_data info)
{ int id, idx, incrm, N= 0, Nm;
  char *sel= NULL, **menutext, tmesg[]= "Select a marker style:";
  xtb_frame *menu= NULL;
  DataSet *this_set;
  char *markmenu[]= {
	  	"   blank filled rectangle -- 1",
		"   black filled rectangle -- 2",
	  	"   blank filled circle -- 3",
		"   black filled circle -- 4",
	  	"   blank filled diamond -- 5",
		"   black filled diamond -- 6",
	  	"   blank filled upwards triangle -- 7",
		"   black filled upwards triangle -- 8",
	  	"   blank filled downwards triangle -- 9",
		"   black filled downwards triangle -- 10",
	  	"   blank filled diabolo -- 11",
		"   black filled diabolo -- 12",
		"   cross (X) -- 13",
		"   plus  (+) -- 14",
		"   blank filled rectangle with a cross (X) -- 15",
		"   blank rectangle with diamond (\"star\") -- 16",
		"   blank circle with diamond (\"star\") -- 17",
		"   Higher values repeat these symbols",
  };

	if( !theWin_Info ){
		return( XTB_STOP );
	}
	if( data_sn_number< 0 || data_sn_number>= setNumber ){
		xtb_error_box( theWin_Info->window, "Not available for non-sets!", "Warning" );
		return( XTB_HANDLED );
	}
	if( (Nm= sizeof(markmenu)/sizeof(char*)-1)> internal_psMarkers ){
		xtb_error_box( theWin_Info->window,
			"There is a mismatch between the number of preconfigured markertypes,\n"
			" and the actually available postscript markers!\n",
			"Error"
		);
	}
	this_set= &AllSets[data_sn_number];
	idx= ABS( this_set->markstyle );
	if( idx ){
		idx-= 1;
	}
	incrm= idx/ Nm; idx= idx % Nm;
	menutext= xtb_CharArray( &N, False, sizeof(markmenu)/sizeof(char*), markmenu, 0 );
	menutext[idx][1]= '>';
	if( incrm ){
		menutext[Nm][1]= '>';
	}

	xtb_bt_set( win, 1, NULL);
	id= xtb_popup_menu( win, menutext[0], tmesg, &sel, &menu);

	if( sel ){
		if( id== Nm ){
			incrm= (incrm)? 0 : 1;
			id= idx;
		}
		id+= incrm* Nm+ 1;
		if( id>= 0 && this_set->markstyle!= -id ){
			data_sn_markstyle= (int) id;
			this_set->markstyle= -1 * abs(data_sn_markstyle);
			theWin_Info->redraw= 1;
			set_changeables(2,True);
		}
		else{
			Boing(100);
		}
		xfree( sel );
	}
	xtb_popup_delete( &menu );
	xfree( menutext[0] );

	xtb_bt_set( win, 0, NULL);
	return( XTB_HANDLED );
}

int Column2ValCat( LocalWin *wi, int *column, ValCategory **vcat )
{ Boolean aaa;
  int nv, nr, N, i, change= False;
	if( apply_to_marked || apply_to_hlt || apply_to_drawn || apply_to_new || apply_to_src ){
		_apply_to_drawn= apply_to_drawn;
		apply_to_drawn= 0;
		_apply_to_marked= apply_to_marked;
		apply_to_marked= 0;
		_apply_to_hlt= apply_to_hlt;
		apply_to_hlt= 0;
		_apply_to_new= apply_to_new;
		apply_to_new= 0;
		_apply_to_src= apply_to_src;
		apply_to_src= 0;
		apply_to_all= True;
		aaa= True;
	}
	if( apply_to_all ){
		nr= (apply_to_rest)? data_sn_number : 0;
		N= (apply_to_prev)? MIN(data_sn_number+1,setNumber) : setNumber;
		apply_to_all= 0;
		apply_to_rest= 0;
		apply_to_prev= 0;
		apply_to_src= 0;
		xtb_bt_set( AF(SNATA_F).win, 0, NULL );
		xtb_bt_set( AF(SNATP_F).win, 0, NULL );
		xtb_bt_set( AF(SNATR_F).win, 0, NULL );
	}
	else{
		nr= data_sn_number;
		N= nr+ 1;
	}
	if( nr>= 0 && nr< setNumber ){
	  ascanf_Function *af;
	  int ok= True, take_usage, j;
		*vcat= Free_ValCat( *vcat );
		nv= 0;
		for( i= nr; i< N && ok; i++ ){
			if( apply_ok(i) ){
			  DataSet *this_set= &AllSets[i];
			  double v;
				for( j= 0; j< this_set->numPoints && ok; j++ ){
					if( (af= parse_ascanf_address( (v= this_set->columns[column[i]][j]), 0, "Columns2ValCat()", 0, &take_usage )) &&
						take_usage
					){
						if( !(*vcat= Add_ValCat(*vcat, &nv, v, af->usage)) ){
							ok= False;
							xtb_error_box( wi->window, "Error allocating a new category", "Error" );
						}
					}
				}
			}
		}
		if( nv ){
			change= True;
		}
	}
	else{
		nv= ValCat_N( *vcat );
		xtb_error_box( wi->window,
			"Request ignored: no valid set(s) selection from which to construct categories!", "Notification" );
	}
	if( aaa ){
		_apply_to_drawn= 0;
		_apply_to_marked= 0;
		_apply_to_hlt= 0;
		_apply_to_new= 0;
		_apply_to_src= 0;
		xtb_bt_set( AF(SNATD_F).win, 0, NULL );
		xtb_bt_set( AF(SNATM_F).win, 0, NULL );
		xtb_bt_set( AF(SNATH_F).win, 0, NULL );
		xtb_bt_set( AF(SNATN_F).win, 0, NULL );
		xtb_bt_set( AF(SNATS_F).win, 0, NULL );
	}
	if( change ){
		wi->redraw= True;
		set_changeables(2, True);
	}

	return( nv );
}

static xtb_hret SD_sdds_sl_fun(Window win, int bval, xtb_data info)
/* 
 \ swaps new_file field of current data_sn_number dataset
 */
{  int i, *field= NULL, bbval= bval, extra= 0;
   short *sfield= NULL;
   int *Info= ((int*) info);
   int do_redraw= 0, new_bval;
   static int level= 0;
   Boolean mod1mask= CheckMask(xtb_modifier_state, Mod1Mask);
   Boolean shiftmask= CheckMask(xtb_modifier_state, ShiftMask);

	if( !theWin_Info ){
		return( XTB_STOP );
	}

	bval= !bval;
	new_bval= bval;

	level+= 1;

	if( Info== &no_ulabels ){
		field= &theWin_Info->no_ulabels;
		do_redraw= !mod1mask;
	}
	else if( Info== &set_no_legend ){
		field= &theWin_Info->no_legend;
		do_redraw= !mod1mask;
	}
	else if( Info== &no_intensity_legend ){
		field= &theWin_Info->no_intensity_legend;
		do_redraw= !mod1mask;
	}
	else if( Info== &legend_always_visible ){
		field= &theWin_Info->legend_always_visible;
		do_redraw= !mod1mask;
	}
	else if( Info== &exact_X_axis ){
		field= &theWin_Info->exact_X_axis;
		do_redraw= !mod1mask;
	}
	else if( Info== &exact_Y_axis ){
		field= &theWin_Info->exact_Y_axis;
		do_redraw= !mod1mask;
	}
	else if( Info== &ValCat_X_levels ){
		field= &theWin_Info->ValCat_X_levels;
		do_redraw= !mod1mask;
	}
	else if( Info== &ValCat_X_axis ){
		field= &theWin_Info->ValCat_X_axis;
		if( shiftmask ){
			if( xtb_error_box( theWin_Info->window,
					"Do you want to use the current set's X column to be used for defining the X categories?\n"
					" For this to be useful, it must consist of only (ascanf) string pointers -- if it\n"
					" doesn't, you'll end up deleting all X categories since the existing categories are\n"
					" deleted first! NB: the raw values are used!\n",
					"X categories"
				)> 0
			){ int N;
			   char msg[256];
				N= Column2ValCat( theWin_Info, theWin_Info->xcol, &theWin_Info->ValCat_X );
				sprintf( msg, "There are now %d X categories defined", N );
				xtb_error_box( theWin_Info->window, msg, "Result" );
			}
			else{
				bval= 0;
				return( XTB_HANDLED );
			}
		}
		if( !theWin_Info->ValCat_X ){
			bval= 0;
			xtb_error_box( theWin_Info->window, "Window doesn't have any X categories defined\n", "FYI" );
		}
		else{
			theWin_Info->axis_stuff.__XLabelLength= 0;
			theWin_Info->axis_stuff.__YLabelLength= 0;
			theWin_Info->axis_stuff.XLabelLength= 0;
			theWin_Info->axis_stuff.YLabelLength= 0;
		}
		do_redraw= !mod1mask;
	}
	else if( Info== &ValCat_X_grid ){
		field= &theWin_Info->ValCat_X_grid;
		do_redraw= !mod1mask;
	}
	else if( Info== &ValCat_Y_axis ){
		field= &theWin_Info->ValCat_Y_axis;
		if( shiftmask ){
			if( xtb_error_box( theWin_Info->window,
					"Do you want to use the current set's Y column to be used for defining the Y categories?\n"
					" For this to be useful, it must consist of only (ascanf) string pointers -- if it\n"
					" doesn't, you'll end up deleting all Y categories since the existing categories are\n"
					" deleted first! NB: the raw values are used!\n",
					"Y categories"
				)> 0
			){ int N;
			   char msg[256];
				N= Column2ValCat( theWin_Info, theWin_Info->ycol, &theWin_Info->ValCat_Y );
				sprintf( msg, "There are now %d Y categories defined", N );
				xtb_error_box( theWin_Info->window, msg, "Result" );
			}
			else{
				bval= 0;
				return( XTB_HANDLED );
			}
		}
		if( !theWin_Info->ValCat_Y ){
			bval= 0;
			xtb_error_box( theWin_Info->window, "Window doesn't have any Y categories defined\n", "FYI" );
		}
		else{
			theWin_Info->axis_stuff.__XLabelLength= 0;
			theWin_Info->axis_stuff.__YLabelLength= 0;
			theWin_Info->axis_stuff.XLabelLength= 0;
			theWin_Info->axis_stuff.YLabelLength= 0;
		}
		do_redraw= !mod1mask;
	}
	else if( Info== &show_all_ValCat_I ){
		field= &theWin_Info->show_all_ValCat_I;
		if( theWin_Info->ValCat_I_axis ){
			theWin_Info->IntensityLegend.legend_width= 0;
		}
		do_redraw= !mod1mask;
	}
	else if( Info== &ValCat_I_axis ){
		field= &theWin_Info->ValCat_I_axis;
		if( shiftmask ){
			if( xtb_error_box( theWin_Info->window,
					"Do you want to use the current set's E column to be used for defining the I categories?\n"
					" For this to be useful, it must consist of only (ascanf) string pointers -- if it\n"
					" doesn't, you'll end up deleting all I categories since the existing categories are\n"
					" deleted first! NB: the raw values are used!\n",
					"Intensity categories"
				)> 0
			){ int N;
			   char msg[256];
				N= Column2ValCat( theWin_Info, theWin_Info->ecol, &theWin_Info->ValCat_I );
				sprintf( msg, "There are now %d Intensity categories defined", N );
				xtb_error_box( theWin_Info->window, msg, "Result" );
			}
			else{
				bval= 0;
				return( XTB_HANDLED );
			}
		}
		if( !theWin_Info->ValCat_I ){
			bval= 0;
			xtb_error_box( theWin_Info->window, "Window doesn't have any Intensity categories defined\n", "FYI" );
		}
		else{
			theWin_Info->IntensityLegend.legend_width= 0;
		}
		do_redraw= !mod1mask;
	}
	else if( Info== &apply_to_all ){
		field= &apply_to_all;
	}
	else if( Info== &apply_to_prev ){
		field= &apply_to_prev;
	}
	else if( Info== &apply_to_rest ){
		field= &apply_to_rest;
	}
	else if( Info== &apply_to_drawn ){
		field= &apply_to_drawn;
	}
	else if( Info== &apply_to_marked ){
		field= &apply_to_marked;
	}
	else if( Info== &apply_to_hlt ){
		field= &apply_to_hlt;
	}
	else if( Info== &apply_to_new ){
		field= &apply_to_new;
	}
	else if( Info== &apply_to_src ){
		field= &apply_to_src;
	}
	else if( Info== &use_average_error ){
		field= (int*) &theWin_Info->use_average_error;
		do_redraw= !mod1mask;
	}
	else if( Info== &no_legend_box ){
		field= (int*) &theWin_Info->no_legend_box;
		do_redraw= !mod1mask;
	}
	else if( Info== &overwrite_legend ){
		field= (int*) &theWin_Info->overwrite_legend;
		do_redraw= !mod1mask;
	}
	else if( Info== &overwrite_AxGrid ){
		field= (int*) &theWin_Info->overwrite_AxGrid;
		do_redraw= !mod1mask;
	}
	else if( Info== &AlwaysDrawHighlighted ){
		field= &theWin_Info->AlwaysDrawHighlighted;
	}
	else if( Info== &CleanSheet ){
		field= &CleanSheet;
	}
	else if( Info== &Raw_NewSets ){
		field= &Raw_NewSets;
	}
	else if( Info== &user_coordinates ){
		field= &(theWin_Info->win_geo.user_coordinates);
		do_redraw= !mod1mask;
	}
	else if( Info== &nobb_coordinates ){
		field= &(theWin_Info->win_geo.nobb_coordinates);
	}
	else if( Info== &set_xname_placed ){
		field= &theWin_Info->xname_placed;
		do_redraw= !mod1mask;
	}
	else if( Info== &set_yname_placed ){
		field= &theWin_Info->yname_placed;
		do_redraw= !mod1mask;
	}
	else if( Info== &yname_vertical ){
		field= &theWin_Info->yname_vertical;
		do_redraw= !mod1mask;
	}
	else if( Info== &set_legend_placed ){
		field= &theWin_Info->legend_placed;
		do_redraw= !mod1mask;
	}
	else if( Info== &intensity_legend_placed ){
		field= &theWin_Info->IntensityLegend.legend_placed;
		do_redraw= !mod1mask;
	}
	else if( Info== &set_zeroFlag ){
		field= &theWin_Info->zeroFlag;
		do_redraw= !mod1mask;
	}
	else if( Info== &htickFlag ){
		field= &theWin_Info->htickFlag;
			*field= !*field;
		do_redraw= !mod1mask;
	}
	else if( Info== &vtickFlag ){
		field= &theWin_Info->vtickFlag;
			*field= !*field;
		do_redraw= !mod1mask;
	}
	else if( Info== &axisFlag ){
		field= &theWin_Info->axisFlag;
		if( *field!= bval ){
			theWin_Info->axis_stuff.__XLabelLength= 0;
			theWin_Info->axis_stuff.__YLabelLength= 0;
			theWin_Info->axis_stuff.XLabelLength= 0;
			theWin_Info->axis_stuff.YLabelLength= 0;
		}
		do_redraw= !mod1mask;
	}
	else if( Info== &set_bbFlag ){
		field= &theWin_Info->bbFlag;
		do_redraw= !mod1mask;
	}
	else if( Info== (int*) overlap ){
		field= &theWin_Info->show_overlap;
		do_redraw= !mod1mask;
	}
	else if( Info== &set_filename_in_legend ){
		field= &theWin_Info->filename_in_legend;
		do_redraw= !mod1mask;
	}
	else if( Info== &labels_in_legend ){
		field= &theWin_Info->labels_in_legend;
		do_update_SD_size= 1;
		do_redraw= !mod1mask;
	}
	else if( Info== &set_no_title ){
		if( !theWin_Info->no_title && CheckMask(xtb_modifier_state, Mod1Mask) ){
		  int resp;
		  extern int AllTitles, DrawColouredSetTitles;
			if( DrawColouredSetTitles ){
				resp= xtb_error_box( theWin_Info->window,
					"DataSets' title text is currently drawn reflecting the sets' colours.\n"
					" Should this feature be turned OFF (this also prevents duplicate lines)?\n",
					"Question"
				);
				if( resp> 0 ){
					DrawColouredSetTitles= False;
					theWin_Info->redraw= True;
					theWin_Info->printed= False;
				}
			}
			else{
				resp= xtb_error_box( theWin_Info->window,
					"Datasets' title text is currently drawn in the default foreground colour, and do not reflect the sets' colours.\n"
					" Should they be coloured (this causes all titles to be drawn, even duplicates)?\n",
					"Question"
				);
				if( resp> 0 ){
					DrawColouredSetTitles= True;
					theWin_Info->redraw= True;
					theWin_Info->printed= False;
				}
			}
			if( !DrawColouredSetTitles ){
				if( AllTitles ){
					resp= xtb_error_box( theWin_Info->window,
						"All titles (lines in the title) are currently being drawn, even those that repeat the previous (line).\n"
						" Should this feature be turned OFF so that no repetitions are shown?\n",
						"Question"
					);
					if( resp> 0 ){
						AllTitles= False;
						theWin_Info->redraw= True;
						theWin_Info->printed= False;
					}
				}
				else{
					resp= xtb_error_box( theWin_Info->window,
						"Currently, only titles (lines in the plot title field) are shown that do not repeat the previous (line).\n"
						" Do you want to turn ON the feature that will cause all lines to be shown?",
						"Question"
					);
					if( resp> 0 ){
						AllTitles= True;
						theWin_Info->redraw= True;
						theWin_Info->printed= False;
					}
				}
			}
			return(XTB_HANDLED);
		}
		else{
			field= &theWin_Info->no_title;
			do_redraw= !mod1mask;
		}
	}
	else if( Info== &no_pens ){
		field= &theWin_Info->no_pens;
		do_redraw= !mod1mask;
	}
	else if( Info== &lz_sym_x ){
		field= (int*) &theWin_Info->lz_sym_x;
		do_redraw= !mod1mask;
	}
	else if( Info== &lz_sym_y ){
		field= (int*) &theWin_Info->lz_sym_y;
		do_redraw= !mod1mask;
	}
	else if( Info== &log_zero_x_mFlag ){
		field= (int*) &theWin_Info->log_zero_x_mFlag;
		if( win== AF(SNLZXMI_F).win ){
			bval*= -1;
			xtb_bt_set( AF(SNLZXMA_F).win, 0, NULL );
		}
		else{
			xtb_bt_set( AF(SNLZXMI_F).win, 0, NULL );
		}
		do_redraw= !mod1mask;
	}
	else if( Info== &log_zero_y_mFlag ){
		field= (int*) &theWin_Info->log_zero_y_mFlag;
		if( win== AF(SNLZYMI_F).win ){
			bval*= -1;
			xtb_bt_set( AF(SNLZYMA_F).win, 0, NULL );
		}
		else{
			xtb_bt_set( AF(SNLZYMI_F).win, 0, NULL );
		}
		do_redraw= !mod1mask;
	}
	else if( Info== &absYFlag ){
		field= (int*) &theWin_Info->absYFlag;
		do_redraw= !mod1mask;
	}
	else if( Info== &fit_x || Info== &fit_pB ){
		field= &theWin_Info->fit_xbounds;
		if( CheckMask( xtb_modifier_state, ShiftMask) && level<= 1 ){
/* 			SD_sdds_sl_fun( win, !bval, (xtb_data) &fit_y);	*/
			theWin_Info->fit_ybounds= bval;
		}
		if( bval ){
			do_redraw= 2;
			if( Info== &fit_pB ){
				bval= 2;
			}
			if( !mod1mask ){
				theWin_Info->win_geo.user_coordinates= 0;
			}
			else{
				theWin_Info->win_geo.user_coordinates= 1;
			}
			xtb_bt_set( AF(USER_F).win, theWin_Info->win_geo.user_coordinates, NULL);
		}
		else if( theWin_Info->fit_xbounds== 2 && Info== &fit_pB ){
		  /* This means we're switching off radix fitting	*/
			bval= 1;
		}
	}
	else if( Info== &fit_y ){
		field= &theWin_Info->fit_ybounds;
		if( CheckMask( xtb_modifier_state, ShiftMask) && level<= 1 ){
/* 			SD_sdds_sl_fun( win, !bval, (xtb_data) &fit_x);	*/
			theWin_Info->fit_xbounds= bval;
		}
		if( bval ){
			do_redraw= 2;
			if( !mod1mask ){
				theWin_Info->win_geo.user_coordinates= 0;
			}
			else{
				theWin_Info->win_geo.user_coordinates= 1;
			}
			xtb_bt_set( AF(USER_F).win, theWin_Info->win_geo.user_coordinates, NULL);
		}
	}
	else if( Info== &fit_after_draw ){
		field= &theWin_Info->fit_after_draw;
		if( bval && (theWin_Info->fit_xbounds || theWin_Info->fit_ybounds) ){
			do_redraw= 2;
			if( !mod1mask ){
				theWin_Info->win_geo.user_coordinates= 0;
			}
			else{
				theWin_Info->win_geo.user_coordinates= 1;
			}
			xtb_bt_set( AF(USER_F).win, theWin_Info->win_geo.user_coordinates, NULL);
		}
	}
	else if( Info== &aspect ){
		field= (int*) &theWin_Info->aspect;
		if( theWin_Info->aspect && !bval ){
			theWin_Info->win_geo.bounds = theWin_Info->win_geo.aspect_base_bounds;
		}
		do_redraw= (mod1mask)? 0 : 2;
	}
	else if( Info== &XSymmetric ){
		field= (int*) &theWin_Info->x_symmetric;
		if( theWin_Info->aspect ){
			theWin_Info->win_geo.bounds = theWin_Info->win_geo.aspect_base_bounds;
		}
		theWin_Info->aspect= 0;
		do_redraw= (mod1mask)? 0 : 2;
	}
	else if( Info== &YSymmetric ){
		field= (int*) &theWin_Info->y_symmetric;
		if( theWin_Info->aspect ){
			theWin_Info->win_geo.bounds = theWin_Info->win_geo.aspect_base_bounds;
		}
		theWin_Info->aspect= 0;
		do_redraw= (mod1mask)? 0 : 2;
	}
	else if( Info== &process_bounds ){
		field= (int*) &theWin_Info->process_bounds;
		do_redraw= (mod1mask)? 0 : 2;
	}
	else if( Info== &transform_axes ){
		field= (int*) &theWin_Info->transform_axes;
		if( !bval ){
			if( (theWin_Info->logXFlag> 0 || theWin_Info->sqrtXFlag> 0) && SD_strlen( theWin_Info->transform.x_process ) ){
				do_error( "Unset either \"logX\" or \"powX\" first\n" );
				bval= 1;
			}
			else if( (theWin_Info->logYFlag> 0 || theWin_Info->sqrtYFlag> 0) && SD_strlen( theWin_Info->transform.y_process ) ){
				do_error( "Unset either \"logY\" or \"powY\" first\n" );
				bval= 1;
			}
			else if( (theWin_Info->polarFlag> 0) &&
				(SD_strlen( theWin_Info->transform.y_process ) || SD_strlen( theWin_Info->transform.x_process))
			){
				do_error( "Unset \"polar\" first\n" );
				bval= 1;
			}
			else{
				do_redraw= (mod1mask)? 0 : 2;
				theWin_Info->axis_stuff.__XLabelLength= 0;
				theWin_Info->axis_stuff.__YLabelLength= 0;
				theWin_Info->axis_stuff.XLabelLength= 0;
				theWin_Info->axis_stuff.YLabelLength= 0;
			}
		}
		else{
			do_redraw= (mod1mask)? 0 : 2;
			theWin_Info->axis_stuff.__XLabelLength= 0;
			theWin_Info->axis_stuff.__YLabelLength= 0;
			theWin_Info->axis_stuff.XLabelLength= 0;
			theWin_Info->axis_stuff.YLabelLength= 0;
		}
	}
	else if( Info== &raw_display ){
		field= (int*) &theWin_Info->raw_display;
		do_redraw= mod1mask;
	}
	else if( Info== &polarFlag ){
	  char buf[64];
		field= (int*) &theWin_Info->polarFlag;
		if( bval && !theWin_Info->raw_display &&
			(SD_strlen( theWin_Info->transform.y_process ) || SD_strlen( theWin_Info->transform.x_process)) &&
			!theWin_Info->transform_axes
		){
			do_error( "Set \"trax\" first to get polar plotting\n");
			bval= 0;
		}
		else{
			do_redraw= mod1mask;
		}
		if( bval ){
			d3str( buf, d3str_format, theWin_Info->powAFlag );
			xtb_ti_set( AF(SNTPX_F).win, buf, &powAFlag );
		}
		else{
			d3str( buf, d3str_format, theWin_Info->powXFlag );
			xtb_ti_set( AF(SNTPX_F).win, buf, &powXFlag );
		}
	}
	else if( Info== &logXFlag ){
		field= (int*) &theWin_Info->logXFlag;
		if( bval && !theWin_Info->raw_display && SD_strlen(theWin_Info->transform.x_process) && !theWin_Info->transform_axes ){
			do_error( "Set \"trax\" first to get logarithmic X-axis\n");
			bval= 0;
		}
		else{
			do_redraw= mod1mask;
		}
		if( bval ){
/* 			xtb_bt_set( AF(SNTLX2_F).win, 0, NULL);	*/
		}
	}
	else if( Info== &logYFlag ){
		field= (int*) &theWin_Info->logYFlag;
		if( bval && !theWin_Info->raw_display && SD_strlen(theWin_Info->transform.y_process) && !theWin_Info->transform_axes ){
			do_error( "Set \"trax\" first to get logarithmic Y-axis\n");
			bval= 0;
		}
		else{
			do_redraw= mod1mask;
		}
		if( bval ){
/* 			xtb_bt_set( AF(SNTLY2_F).win, 0, NULL);	*/
		}
	}
	else if( Info== &logXFlag2 ){
		field= &theWin_Info->logXFlag;
		if( bval && !theWin_Info->raw_display && SD_strlen(theWin_Info->transform.x_process) && !theWin_Info->transform_axes ){
			do_error( "Set \"trax\" first to get logarithmic X-axis\n");
			bval= 0;
		}
		else{
			do_redraw= mod1mask;
		}
		if( bval ){
			bval= 3;
			xtb_bt_set( AF(SNTLX_F).win, 0, NULL);
		}
	}
	else if( Info== &logYFlag2 ){
		field= &theWin_Info->logYFlag;
		if( bval && !theWin_Info->raw_display && SD_strlen(theWin_Info->transform.y_process) && !theWin_Info->transform_axes ){
			do_error( "Set \"trax\" first to get logarithmic Y-axis\n");
			bval= 0;
		}
		else{
			do_redraw= mod1mask;
		}
		if( bval ){
			bval= 3;
			xtb_bt_set( AF(SNTLY_F).win, 0, NULL);
		}
	}
	else if( Info== &sqrtXFlag ){
		field= (int*) &theWin_Info->sqrtXFlag;
		if( bval && !theWin_Info->raw_display && SD_strlen(theWin_Info->transform.x_process) && !theWin_Info->transform_axes ){
			do_error( "Set \"trax\" first to get sqrt/pow X-axis\n");
			bval= 0;
		}
		else{
			do_redraw= mod1mask;
		}
	}
	else if( Info== &sqrtYFlag ){
		field= (int*) &theWin_Info->sqrtYFlag;
		if( bval && !theWin_Info->raw_display && SD_strlen(theWin_Info->transform.y_process) && !theWin_Info->transform_axes ){
			do_error( "Set \"trax\" first to get sqrt/pow Y-axis\n");
			bval= 0;
		}
		else{
			do_redraw= mod1mask;
		}
	}
	else if( Info== &WdebugFlag ){
		field= (int*) &theWin_Info->debugFlag;
		if( *field> 0 ){
			bval= (bval)? 0 : -1;
		}
		else if( *field< 0 ){
			bval= (bval)? 1 : 0;
		}
		else{
			bval= (bval)? 1 : -1;
		}
	}
	else if( Info== &debugFlag ){
	  extern int *dbF_cache;
		field= (int*) Info;
		if( do_redraw< 0 ){
			do_redraw= 0;
		}
		if( dbF_cache ){
			*dbF_cache= bval;
		}
		if( bval ){
			Synchro_State= 0;
			X_Synchro(theWin_Info);
		}
		else{
			Synchro_State= 1;
			X_Synchro(theWin_Info);
		}
	}
	else if( Info== &Synchro_State ){
		field= (int*) Info;
		if( do_redraw< 0 ){
			do_redraw= 0;
		}
		if( bval ){
			Synchro_State= 0;
			HardSynchro= True;
			X_Synchro(theWin_Info);
		}
		else{
			Synchro_State= 1;
			HardSynchro= False;
			X_Synchro(theWin_Info);
		}
	}
	else if( Info== &dataSynchro_State ){
		field= (int*) Info;
		if( do_redraw< 0 ){
			do_redraw= 0;
		}
		if( bval ){
			theWin_Info->data_sync_draw= 1;
		}
		else{
			theWin_Info->data_sync_draw= 0;
		}
	}
	else if( Info== &data_silent_process ){
		field= (int*) Info;
		if( do_redraw< 0 ){
			do_redraw= 0;
		}
		if( bval ){
			theWin_Info->data_silent_process= 1;
		}
		else{
			theWin_Info->data_silent_process= 0;
		}
	}
	else if( Info== &Allow_Fractions ){
		field= (int*) Info;
		changed_Allow_Fractions= True;
		do_redraw= mod1mask;
	}
	if( Info== &set_draw_set || Info== &set_raw_display ){
		extra= theWin_Info->ulabels;
	}
#define MULTIPLE_SET()	((apply_to_marked || apply_to_hlt || apply_to_drawn || apply_to_new || apply_to_src) && field!= &apply_to_marked && field!= &apply_to_hlt)
/* 	if( (sfield || field) || (data_sn_number>= 0 && data_sn_number< setNumber+ extra) )	*/
	if( (sfield || field) || (data_sn_number>= 0 && data_sn_number< setNumber+ extra) ||
		MULTIPLE_SET() || apply_to_all || apply_to_prev || apply_to_rest
	)
	{ int n, ok, j= 0, ard= autoredraw, no_ard= False;
	  Boolean aaa;
		xtb_bt_set( win, bval, (xtb_data) 0);
		if( MULTIPLE_SET() ){
			_apply_to_drawn= apply_to_drawn;
			apply_to_drawn= 0;
			_apply_to_marked= apply_to_marked;
			apply_to_marked= 0;
			_apply_to_hlt= apply_to_hlt;
			apply_to_hlt= 0;
			_apply_to_new= apply_to_new;
			apply_to_new= 0;
			_apply_to_src= apply_to_src;
			apply_to_src= 0;
			apply_to_all= True;
			aaa= True;
		}
		if( apply_to_all && !(field || sfield) ){
		  /* apply_to_all only makes sense for per-set settings. All other settings
		   \ already have initialised either field or sfield.
		   */
			i= (apply_to_rest)? data_sn_number : 0;
			n= (apply_to_prev)? MIN(data_sn_number+1,setNumber) : setNumber;
			apply_to_all= 0;
			apply_to_prev= 0;
			apply_to_rest= 0;
			apply_to_src= 0;
			xtb_bt_set( AF(SNATA_F).win, 0, NULL );
			xtb_bt_set( AF(SNATP_F).win, 0, NULL );
			xtb_bt_set( AF(SNATR_F).win, 0, NULL );
			autoredraw= False;
		}
		else{
			i= data_sn_number;
			n= i+ 1;
			  /* In this case we don't change apply_to_all, so we also
			   \ don't change apply_to_marked and apply_to_hlt
			   */
			aaa= False;
			autoredraw= ard;
			apply_to_drawn= _apply_to_drawn;
			_apply_to_drawn= 0;
			apply_to_marked= _apply_to_marked;
			_apply_to_marked= 0;
			apply_to_hlt= _apply_to_hlt;
			_apply_to_hlt= 0;
			apply_to_new= _apply_to_new;
			_apply_to_new= 0;
			apply_to_src= _apply_to_src;
			_apply_to_src= 0;
		}
		for( ; i< n; i++, j++ ){
			if( i>= 0 && i< setNumber ){
				if( apply_ok(i) ){
					if( !(field || sfield) ){
						ok= 0;
					}
					else{
						ok= 1;
					}
					if( Info == &set_draw_set ){
						AllSets[i].draw_set= bval;
						sfield= &(theWin_Info->draw_set[i]);
						do_redraw= !mod1mask;
					}
					else if( Info == &set_mark_set ){
						sfield= &(theWin_Info->mark_set[i]);
						do_redraw= !mod1mask;
					}
					else if( Info == &set_highlight ){
/* 						AllSets[i].highlight= bval;	*/
						field= &(theWin_Info->legend_line[i].highlight);
						do_redraw= !mod1mask;
						if( CheckMask(xtb_modifier_state, Mod1Mask) ){
						  char message[512];
						  int p;
						  extern double highlight_par[];
						  extern int highlight_mode, highlight_npars;
						  _ALLOCA( ebuf, char, highlight_npars* 32+ 128, ebuf_len);
							sprintf( ebuf, "-hl_mode %d -hl_pars %s", highlight_mode, d2str(highlight_par[0], NULL, NULL) );
							for( p= 1; p< highlight_npars; p++ ){
								sprintf( ebuf, "%s,%s", ebuf, d2str( highlight_par[p], NULL, NULL) );
							}
							STRINGCHECK( ebuf, ebuf_len );
							sprintf( message,
								" Mode 0: total-width= 5* pow( MAX(1, line-width), hl_pars[0]\n"
								" Mode 1: total-width= line-width + hl_pars[0]\n"
								" hl_pars[1] is a Boolean controlling whether or not legend-text\n"
								" and title-text are highlighted too when a set is highlighted.\n"
							);
							STRINGCHECK( message, sizeof(message)/sizeof(char) );
							if( xtb_input_dialog( SD_Dialog.win, ebuf, 64, ebuf_len, message,
									"Enter/edit highlight mode and parameters", False,
									"", NULL, "", NULL, "Edit", SimpleEdit_h )
							){
								ParseArgsString( ebuf );
							}
							  /* Don't change the selected field in this case ;-)	*/
							bval= *field;
						}
					}
					else if( Info== &set_raw_display ){
						field= &AllSets[i].raw_display;
						do_redraw= !mod1mask;
					}
					else if( Info== &points_added ){
						field= &AllSets[i].points_added;
					}
					else if( Info== &set_sarrow ){
						field= &AllSets[i].arrows;
						if( new_bval ){
							bval= *field | 0x01;
						}
						else{
							bval= *field & (~0x01);
						}
						do_redraw= !mod1mask;
					}
					else if( Info== &set_earrow ){
						field= &AllSets[i].arrows;
						if( new_bval ){
							bval= *field | 0x02;
						}
						else{
							bval= *field & (~0x02);
						}
						do_redraw= !mod1mask;
					}
					else if( Info== &set_show_legend ){
						field= &AllSets[i].show_legend;
						do_redraw= !mod1mask;
					}
					else if( Info== &set_show_llines ){
						field= &AllSets[i].show_llines;
						do_redraw= !mod1mask;
					}
					else if( Info== &set_noLines ){
						field= &AllSets[i].noLines;
						do_redraw= !mod1mask;
					}
					else if( Info== &set_floating ){
						field= &AllSets[i].floating;
						do_redraw= !mod1mask;
					}
					else if( Info== &set_barFlag ){
						if( CheckMask( xtb_modifier_state, Mod1Mask) ){
							return( SD_set_bardimensions(win,bval, info) );
						}
						else{
							field= &AllSets[i].barFlag;
							do_redraw= !mod1mask;
						}
					}
					else if( Info== &set_markFlag ){
						field= &AllSets[i].markFlag;
						do_redraw= !mod1mask;
					}
					else if( Info== &set_show_errors ){
						field= &AllSets[i].use_error;
						do_redraw= !mod1mask;
					}
					else if( Info== &set_overwrite_marks ){
						field= &AllSets[i].overwrite_marks;
						do_redraw= !mod1mask;
					}
					else if( Info== &set_pixelMarks ){
						field= &AllSets[i].pixelMarks;
						do_redraw= !mod1mask;
						if( bval== 1 ){
/* 							xtb_bt_set_text( AF(SNPM_F).win, bval, "Dts", (xtb_data) 0);	*/
						}
						else if( bbval== 1 ){	
							bval= 2;
/* 							xtb_bt_set_text( AF(SNPM_F).win, bval, "Blbs", (xtb_data) 0);	*/
						}
						else{
							bval= 0;
/* 							xtb_bt_set_text( AF(SNPM_F).win, bval, "Smbl", (xtb_data) 0);	*/
						}
					}
					else switch( (int) info ){
						case VMARK_ON:
							field= &AllSets[i].valueMarks;
							do_redraw= !mod1mask;
							if( CheckMask(*field, VMARK_ON) ){
								bval= *field & ~VMARK_ON;
							}
							else{
								bval= *field | VMARK_ON;
							}
							break;
						case VMARK_FULL:
							field= &AllSets[i].valueMarks;
							do_redraw= !mod1mask;
							if( CheckMask(*field, VMARK_FULL) ){
								bval= *field & ~VMARK_FULL;
							}
							else{
								bval= *field | VMARK_FULL;
							}
							break;
							break;
						case VMARK_RAW:
							field= &AllSets[i].valueMarks;
							do_redraw= !mod1mask;
							if( CheckMask(*field, VMARK_RAW) ){
								bval= *field & ~VMARK_RAW;
							}
							else{
								bval= *field | VMARK_RAW;
							}
							break;
							break;
						default:
							break;
					}
					if( !ok && !(sfield || field) ){
						do_error( "Illegal button in SD_sdds_sl_fun\nLine="STRING(__LINE__)"\nReport to sysop!\n");
						level-= 1;
						return( XTB_HANDLED );
					}
					xtb_sr_set( AF(SNNRSL_F).win, (double) i, NULL);
				}
			}
			else if( i< setNumber+ theWin_Info->ulabels  ){
			  UserLabel *ul= GetULabelNr( i- setNumber);
				if( !(field || sfield) ){
					ok= 0;
				}
				else{
					ok= 1;
				}
				if( Info == &set_draw_set ){
					field= &(ul->do_draw);
					do_redraw= mod1mask;
				}
				else if( Info== &set_raw_display ){
				  int b= !bval;
					if( ul->do_transform!= b ){
						theWin_Info->redraw= 1;
						do_redraw= mod1mask;
						ul->do_transform= b;
					}
					ok= 1;
				}
				if( !ok && !(sfield || field) ){
					do_error( "Illegal button in SD_sdds_sl_fun\nLine="STRING(__LINE__)"\nReport to sysop!\n");
					level-= 1;
					return( XTB_HANDLED );
				}
				xtb_sr_set( AF(SNNRSL_F).win, (double) i, NULL);
			}
			else if( !(sfield || field) && !aaa ){
				do_error( "Illegal button in SD_sdds_sl_fun\nLine="STRING(__LINE__)"\nReport to sysop!\n");
				level-= 1;
				return( XTB_HANDLED );
			}
			if( field ){
				if( *field!= bval ){
					theWin_Info->redraw= do_redraw | (ard && !no_ard);
					if( theWin_Info->redraw && theWin_Info->redrawn== -3 ){
						theWin_Info->redrawn= 0;
					}
					*field= bval;
					if( field== &apply_to_all ){
						no_ard= True;
					}
					else if( field== &apply_to_prev ){
						apply_to_all= *field;
						apply_to_rest= 0;
						xtb_bt_set( AF(SNATR_F).win, 0, NULL );
						no_ard= True;
					}
					else if( field== &apply_to_rest ){
						apply_to_all= *field;
						apply_to_prev= 0;
						xtb_bt_set( AF(SNATP_F).win, 0, NULL );
						no_ard= True;
					}
					else if( field== &apply_to_drawn ){
						apply_to_all= *field;
						xtb_bt_set( AF(SNATA_F).win, apply_to_all, NULL );
						no_ard= True;
					}
					else if( field== &apply_to_marked ){
						apply_to_all= *field;
						xtb_bt_set( AF(SNATA_F).win, apply_to_all, NULL );
						no_ard= True;
					}
					else if( field== &apply_to_hlt ){
						apply_to_all= *field;
						xtb_bt_set( AF(SNATA_F).win, apply_to_all, NULL );
						no_ard= True;
					}
					else if( field== &apply_to_new ){
						apply_to_all= *field;
						xtb_bt_set( AF(SNATA_F).win, apply_to_all, NULL );
						no_ard= True;
					}
					else if( field== &apply_to_src ){
						apply_to_all= *field;
						xtb_bt_set( AF(SNATA_F).win, apply_to_all, NULL );
						no_ard= True;
					}
				}
			}
			else if( sfield ){
				if( *sfield!= (short) bval ){
					theWin_Info->redraw= do_redraw | (ard && !no_ard);
					if( theWin_Info->redraw && theWin_Info->redrawn== -3 ){
						theWin_Info->redrawn= 0;
					}
					*sfield= (short) bval;
				}
			}
			field= NULL;
			sfield= NULL;
		}
			if( Info== &htickFlag ){
				field= &theWin_Info->htickFlag;
				*field= !*field;
			}
			if( Info== &vtickFlag ){
				field= &theWin_Info->vtickFlag;
				*field= !*field;
			}
		if( do_redraw && level<= 1 ){
			theWin_Info->printed= 0;
			if( do_redraw== 2 ){
				SD_redraw_fun( 0, 0, rinfo);
				if( !theWin_Info ){
					return( XTB_STOP );
				}
			}
		}
		if( aaa ){
			_apply_to_drawn= 0;
			_apply_to_marked= 0;
			_apply_to_hlt= 0;
			_apply_to_new= 0;
			_apply_to_src= 0;
			xtb_bt_set( AF(SNATD_F).win, 0, NULL );
			xtb_bt_set( AF(SNATM_F).win, 0, NULL );
			xtb_bt_set( AF(SNATH_F).win, 0, NULL );
			xtb_bt_set( AF(SNATN_F).win, 0, NULL );
			xtb_bt_set( AF(SNATS_F).win, 0, NULL );
		}
		autoredraw= ard;
		if( level<= 1 ){
			if( do_update_SD_size ){
/* 				update_SD_size();	*/
			}
			else{
				set_changeables( (autoredraw)? 2 : 0, !no_ard );
			}
		}
	}
	else{
		Boing( 5);
	}
	level-= 1;
	return( XTB_HANDLED );
}

extern void _CloseSD_Dialog( xtb_frame *dial, Boolean delete);
extern void CloseSD_Dialog( xtb_frame *dial );

/*ARGSUSED*/
static xtb_hret ok_fun(win, bval, info)
Window win;			/* Button window     */
int bval;			/* Button value      */
xtb_data info;			/* Local button info */
/*
 * This is the handler function for when the `Ok' button
 * is hit.  It sets the button,  does the hardcopy output,
 * and turns the button off.  It returns a status which
 * deactivates the dialog.
 */
{

    xtb_bt_set(win, 1, (xtb_data) 0);
    xtb_bt_set(win, 0, (xtb_data) 0);
	SD_Dialog.mapped= -1;
    _CloseSD_Dialog( &SD_Dialog, CheckMask( xtb_modifier_state, Mod1Mask) );
    return XTB_HANDLED;
}

xtb_hret SD_edit_fun( Window win, int bval, xtb_data info)
{ ALLOCA( errbuf, char, ASCANF_FUNCTION_BUF+ 256, errbuf_len);
    xtb_bt_set(win, 1, (xtb_data) 0);
	if( info== (xtb_data) SD_edit_fun ){
		errbuf[0]= '\0';
		if( data_sn_number== -19 ){
			EditExpression( theWin_Info->process.enter_raw_after, "*ENTER_RAW_AFTER*", errbuf );
			if( errbuf[0] ){
				xtb_error_box( theWin_Info->window, errbuf, "Message" );
			}
		}
		else if( data_sn_number== -18 ){
			EditExpression( theWin_Info->process.leave_raw_after, "*LEAVE_RAW_AFTER*", errbuf );
			if( errbuf[0] ){
				xtb_error_box( theWin_Info->window, errbuf, "Message" );
			}
		}
		else if( data_sn_number== -17 ){
			EditExpression( theWin_Info->process.dump_before, "**DUMP_BEFORE**", errbuf );
			if( errbuf[0] ){
				xtb_error_box( theWin_Info->window, errbuf, "Message" );
			}
		}
		else if( data_sn_number== -16 ){
			EditExpression( theWin_Info->process.dump_after, "**DUMP_AFTER**", errbuf );
			if( errbuf[0] ){
				xtb_error_box( theWin_Info->window, errbuf, "Message" );
			}
		}
		else if( data_sn_number== -15 ){
			EditExpression( theWin_Info->process.draw_before, "**DRAW_BEFORE**", errbuf );
			if( errbuf[0] ){
				xtb_error_box( theWin_Info->window, errbuf, "Message" );
			}
		}
		else if( data_sn_number== -14 ){
			EditExpression( theWin_Info->process.draw_after, "**DRAW_AFTER**", errbuf );
			if( errbuf[0] ){
				xtb_error_box( theWin_Info->window, errbuf, "Message" );
			}
		}
		else if( data_sn_number== -12 ){
		  char as= ascanf_separator;
			ascanf_window= theWin_Info->window;
			ActiveWin= theWin_Info;
			ascanf_separator= SD_ascanf_separator_buf[0];
			EditExpression( "", "**EVAL**", errbuf );
			ascanf_separator= as;
			if( errbuf[0] ){
				xtb_error_box( theWin_Info->window, errbuf, "Message" );
			}
			if( !SD_Dialog.mapped || !theWin_Info ){
			  /* While this can theoretically also arrive in the other situations,
			   \ it is more likely to happen when user kills us (= tells us to quit)
			   \ in the middle of a lengthy EVAL command.
			   */
				return( XTB_HANDLED );
			}
		}
		else if( data_sn_number== -11 ){
			EditExpression( theWin_Info->transform.x_process, "**TRANSFORM_X**", errbuf );
			if( errbuf[0] ){
				xtb_error_box( theWin_Info->window, errbuf, "Message" );
			}
		}
		else if( data_sn_number== -10 ){
			EditExpression( theWin_Info->transform.y_process, "**TRANSFORM_Y**", errbuf );
			if( errbuf[0] ){
				xtb_error_box( theWin_Info->window, errbuf, "Message" );
			}
		}
		else if( data_sn_number== -9 ){
			EditExpression( theWin_Info->process.data_init, "**DATA_INIT**", errbuf );
			if( errbuf[0] ){
				xtb_error_box( theWin_Info->window, errbuf, "Message" );
			}
		}
		else if( data_sn_number== -8 ){
			EditExpression( theWin_Info->process.data_before, "**DATA_BEFORE**", errbuf );
			if( errbuf[0] ){
				xtb_error_box( theWin_Info->window, errbuf, "Message" );
			}
		}
		else if( data_sn_number== -7 ){
			EditExpression( theWin_Info->process.data_process, "**DATA_PROCESS**", errbuf );
			if( errbuf[0] ){
				xtb_error_box( theWin_Info->window, errbuf, "Message" );
			}
		}
		else if( data_sn_number== -6 ){
			EditExpression( theWin_Info->process.data_after, "**DATA_AFTER**", errbuf );
			if( errbuf[0] ){
				xtb_error_box( theWin_Info->window, errbuf, "Message" );
			}
		}
		else if( data_sn_number== -5 ){
			EditExpression( theWin_Info->process.data_finish, "**DATA_FINISH**", errbuf );
			if( errbuf[0] ){
				xtb_error_box( theWin_Info->window, errbuf, "Message" );
			}
		}
		else{
		  Sinc sinc;
			Sinc_string_behaviour( &sinc, NULL, 0,0, SString_Dynamic );
			Sflush(&sinc);
			xtb_ti_get( SD_info->sn_legend, data_legend_buf, (xtb_data *) 0);
			SimpleEdit( data_legend_buf, &sinc, errbuf );
			if( sinc._tlen>= LMAXBUFSIZE ){
				LMAXBUFSIZE= sinc._tlen+ 1;
				update_SD_size();
			}
/* 			Boing(6);	*/
			if( sinc.sinc.string && *sinc.sinc.string && strcmp( data_legend_buf, sinc.sinc.string ) ){
				legend_changed+= 1;
				xtb_ti_set( SD_info->sn_legend, sinc.sinc.string, NULL);
				SD_snl_fun(SD_info->sn_legend, XK_Up, sinc.sinc.string, (xtb_data*) &new_legend);
			}
			xfree( sinc.sinc.string );
		}
	}
	else{
		Boing(10);
	}
    xtb_bt_set(win, 0, (xtb_data) 0);
    return XTB_HANDLED;
}

char sncols[]= "X,Y,E,L,N cols (of %d): ";

int SD_ULabels= 0;
static int dsnn= 0;

/* Update the values/states of a number of entries	*/
int set_changeables(int do_it,int allow_auto)
{ char buf[MAXCHBUF];
  int new_file= 0;
  static int count= 0;

  if( !theWin_Info || SD_Dialog.destroy_it || SD_Dialog.mapped<= 0 ){
    /* The dialog was probably closed behind our backs, deleting the info
	 \ about the belonging Window. Thus we return without doing anything
	 \ whatsoever.
	 \ Similarly, when the SD_Dialog wants to close, we wouldn't want
	 \ to interfere with that neither.
	 */
	  return(0);
  }

  if( !do_it ){
	  count+= 1;
	  return(0);
  }
  else if( count== 0 && do_it!= 2 ){
	  dont_set_changeables= 0;
	  return(0);
  }
  else{
	  count= 0;
  }

  if( dont_set_changeables ){
	  dont_set_changeables= 0;
	  return(0);
  }

  dsnn= data_sn_number;

	xtb_bt_set( AF(DEBUG_F).win, debugFlag, NULL );
	sprintf( buf, "%d", debugLevel );
	xtb_ti_set( AF(DEBUGL_F).win, buf, 0);
	if( theWin_Info->debugFlag ){
		if( theWin_Info->debugFlag> 0 ){
			xtb_bt_set_text( AF(WDEBUG_F).win, 0, "Wdbg", NULL );
			xtb_bt_set2( AF(WDEBUG_F).win, 1, 1, NULL );
		}
		else{
			xtb_bt_set_text( AF(WDEBUG_F).win, 0, "no.Wdbg", NULL );
			xtb_bt_set2( AF(WDEBUG_F).win, 1, 0, NULL );
		}
	}
	else{
		xtb_bt_set_text( AF(WDEBUG_F).win, 0, "Wdbg", NULL );
		xtb_bt_set2( AF(WDEBUG_F).win, 0, 0, NULL );
	}

	xtb_bt_set( AF(FRACP_F).win, Allow_Fractions, NULL);
	xtb_ti_set( AF(D3STRF_F).win, d3str_format, 0);
	xtb_bt_set( AF(SYNC_F).win, Synchro_State, NULL);
	xtb_bt_set( AF(DSYNC_F).win, theWin_Info->data_sync_draw, NULL);
	xtb_bt_set( AF(DSILENT_F).win, theWin_Info->data_silent_process, NULL);

	xtb_bt_set( AF(SDDR_F).win, SD_dynamic_resize, NULL );

	if( SD_ascanf_separator_buf[0]== '\0' ){
		SD_ascanf_separator_buf[0]= ascanf_separator;
	}
	SD_ascanf_separator_buf[1]= '\0';
	xtb_ti_set( AF(DPNAS_F).win, SD_ascanf_separator_buf, NULL);

		xtb_br_set( AF(OERRBROW_F).win, error_type2num[ SD_get_errb() ] );
		xtb_br_set( AF(SNLEGFUN_F).win, LegendFunctionNr );
		xtb_bt_set( AF(SNLP_F).win, theWin_Info->legend_placed, NULL);
		xtb_bt_set( AF(SNILP_F).win, theWin_Info->IntensityLegend.legend_placed, NULL);
		xtb_bt_set( AF(SNXP_F).win, theWin_Info->xname_placed, NULL);
		xtb_bt_set( AF(SNYP_F).win, theWin_Info->yname_placed, NULL);
		xtb_bt_set( AF(SNYV_F).win, theWin_Info->yname_vertical, NULL);
		xtb_bt_set( AF(SNOUL_F).win, theWin_Info->no_ulabels, (xtb_data) 0 );
		xtb_bt_set( AF(SNL_F).win, theWin_Info->no_legend, (xtb_data) 0 );
		xtb_bt_set( AF(SNIL_F).win, theWin_Info->no_intensity_legend, (xtb_data) 0 );
		xtb_bt_set( AF(SNNP_F).win, theWin_Info->no_pens, (xtb_data) 0 );
		if( theWin_Info->pen_list ){
			xtb_enable( AF(SNNP_F).win );
		}
		else{
			xtb_disable( AF(SNNP_F).win );
		}
		xtb_bt_set( AF(SNLAV_F).win, theWin_Info->legend_always_visible, (xtb_data) 0 );
		xtb_bt_set( AF(SNEXA_F).win, theWin_Info->exact_X_axis, (xtb_data) 0 );
		xtb_bt_set( AF(SNEYA_F).win, theWin_Info->exact_Y_axis, (xtb_data) 0 );
		xtb_bt_set( AF(SNVCXA_F).win, theWin_Info->ValCat_X_axis, (xtb_data) 0 );
		xtb_bt_set( AF(SNVCXG_F).win, theWin_Info->ValCat_X_grid, (xtb_data) 0 );
		xtb_ti_set( AF(SNVCXL_F).win, d3str( buf, d3str_format, theWin_Info->ValCat_X_levels), NULL );
		xtb_bt_set( AF(SNVCYA_F).win, theWin_Info->ValCat_Y_axis, (xtb_data) 0 );
		xtb_bt_set( AF(SNSAVCI_F).win, theWin_Info->show_all_ValCat_I, (xtb_data) 0 );
		xtb_bt_set( AF(SNVCIA_F).win, theWin_Info->ValCat_I_axis, (xtb_data) 0 );
		xtb_bt_set( AF(SUAE_F).win, theWin_Info->use_average_error, (xtb_data) 0 );
		xtb_bt_set( AF(SNNLB_F).win, theWin_Info->no_legend_box, (xtb_data) 0 );
		xtb_bt_set( AF(SOL_F).win, theWin_Info->overwrite_legend, (xtb_data) 0 );
		xtb_bt_set( AF(SNT_F).win, theWin_Info->no_title, (xtb_data) 0 );
		xtb_bt_set( AF(SNOL_F).win, theWin_Info->show_overlap, (xtb_data) 0 );
		xtb_bt_set( AF(SNSF_F).win, theWin_Info->filename_in_legend, (xtb_data) 0 );
		xtb_bt_set( AF(SNSLL_F).win, theWin_Info->labels_in_legend, (xtb_data) 0 );
		xtb_bt_set( AF(SNAXF_F).win, theWin_Info->axisFlag, (xtb_data) 0 );
		xtb_bt_set( AF(SOAG_F).win, theWin_Info->overwrite_AxGrid, (xtb_data) 0 );
		xtb_bt_set( AF(SNBBF_F).win, (theWin_Info->bbFlag && theWin_Info->bbFlag!= -1)? theWin_Info->bbFlag : 0, (xtb_data) 0 );
		xtb_bt_set( AF(SNHTKF_F).win, (theWin_Info->htickFlag>0)? 0 : 1, (xtb_data) 0 );
		xtb_bt_set( AF(SNVTKF_F).win, (theWin_Info->vtickFlag>0)? 0 : 1, (xtb_data) 0 );
		xtb_bt_set( AF(SNZLF_F).win, theWin_Info->zeroFlag, (xtb_data) 0 );

	sprintf( buf, "%d", AxisValueMinDigits );
	xtb_ti_set( AF(AXMIDIG_F).win, buf, NULL );
	xtb_ti_set( AF(AXSTR_F).win, (AxisValueFormat)? AxisValueFormat : "", NULL );

	xtb_bt_set( AF(XFIT_F).win, theWin_Info->fit_xbounds, NULL );
	xtb_bt_set( AF(YFIT_F).win, theWin_Info->fit_ybounds, NULL );
	if( theWin_Info->fit_xbounds== 2 ){
		xtb_bt_set( AF(PBFIT_F).win, theWin_Info->fit_xbounds, NULL );
	}
	else{
		xtb_bt_set( AF(PBFIT_F).win, 0, NULL );
	}
	xtb_bt_set( AF(FITAD_F).win, theWin_Info->fit_after_draw, NULL );
	xtb_bt_set( AF(ASPECT_F).win, ABS(theWin_Info->aspect>0), NULL );
	xtb_bt_set( AF(XSYMM_F).win, ABS(theWin_Info->x_symmetric), NULL );
	xtb_bt_set( AF(YSYMM_F).win, ABS(theWin_Info->y_symmetric), NULL );

	bounds_changed= 0;
	if( theWin_Info->win_geo.nobb_range_X || theWin_Info->win_geo.nobb_range_Y ){
	}
	else{
	}
	if( theWin_Info->win_geo.nobb_coordinates ){
		d3str( buf, d3str_format, theWin_Info->win_geo.nobb_loX );
		if( theWin_Info->win_geo.nobb_range_X ){
			strcat( buf, " # set" );
		}
		bounds_changed+= (xtb_ti_set( AF(XMIN_F).win, buf, NULL )== 1)? 1 : 0;
		d3str( buf, d3str_format, theWin_Info->win_geo.nobb_loY );
		if( theWin_Info->win_geo.nobb_range_Y ){
			strcat( buf, " # set" );
		}
		bounds_changed+= (xtb_ti_set( AF(YMIN_F).win, buf, NULL )== 1)? 1 : 0;
		d3str( buf, d3str_format, theWin_Info->win_geo.nobb_hiX );
		if( theWin_Info->win_geo.nobb_range_X ){
			strcat( buf, " # set" );
		}
		bounds_changed+= (xtb_ti_set( AF(XMAX_F).win, buf, NULL )== 1)? 1 : 0;
		d3str( buf, d3str_format, theWin_Info->win_geo.nobb_hiY );
		if( theWin_Info->win_geo.nobb_range_Y ){
			strcat( buf, " # set" );
		}
		bounds_changed+= (xtb_ti_set( AF(YMAX_F).win, buf, NULL )== 1)? 1 : 0;
	}
	else{
		d3str( buf, d3str_format, theWin_Info->win_geo.bounds._loX );
		bounds_changed+= (xtb_ti_set( AF(XMIN_F).win, buf, NULL )== 1)? 1 : 0;
		d3str( buf, d3str_format, theWin_Info->win_geo.bounds._loY );
		bounds_changed+= (xtb_ti_set( AF(YMIN_F).win, buf, NULL )== 1)? 1 : 0;
		d3str( buf, d3str_format, theWin_Info->win_geo.bounds._hiX );
		bounds_changed+= (xtb_ti_set( AF(XMAX_F).win, buf, NULL )== 1)? 1 : 0;
		d3str( buf, d3str_format, theWin_Info->win_geo.bounds._hiY );
		bounds_changed+= (xtb_ti_set( AF(YMAX_F).win, buf, NULL )== 1)? 1 : 0;
	}

	d3str( buf, d3str_format, (theWin_Info->ValCat_X_axis)? theWin_Info->ValCat_X_incr : theWin_Info->Xincr_factor );
	xtb_ti_set( AF(SNXIF_F).win, buf, NULL );
	d3str( buf, d3str_format, (theWin_Info->ValCat_Y_axis)? theWin_Info->ValCat_Y_incr : theWin_Info->Yincr_factor );
	xtb_ti_set( AF(SNYIF_F).win, buf, NULL );
	d3str( buf, d3str_format, theWin_Info->Xscale );

	xtb_ti_set( AF(SNS2X_F).win, buf, NULL );
	d3str( buf, d3str_format, theWin_Info->Yscale );
	xtb_ti_set( AF(SNS2Y_F).win, buf, NULL );
	d3str( buf, d3str_format, theWin_Info->Xbias_thres );
	xtb_ti_set( AF(SNBTX_F).win, buf, NULL );
	d3str( buf, d3str_format, theWin_Info->Ybias_thres );
	xtb_ti_set( AF(SNBTY_F).win, buf, NULL );
	d3str( buf, d3str_format, theWin_Info->log_zero_x );
	xtb_ti_set( AF(SNLZXV_F).win, buf, NULL );
	d3str( buf, d3str_format, theWin_Info->log_zero_y );
	xtb_ti_set( AF(SNLZYV_F).win, buf, NULL );
	xtb_bt_set( AF(SNLZXMI_F).win, (theWin_Info->log_zero_x_mFlag< 0), NULL);
	xtb_bt_set( AF(SNLZXMA_F).win, (theWin_Info->log_zero_x_mFlag> 0), NULL);
	xtb_bt_set( AF(SNLZYMI_F).win, (theWin_Info->log_zero_y_mFlag< 0), NULL);
	xtb_bt_set( AF(SNLZYMA_F).win, (theWin_Info->log_zero_y_mFlag> 0), NULL);
	xtb_ti_set( AF(SNLZXS_F).win, theWin_Info->log_zero_sym_x, NULL );
	xtb_ti_set( AF(SNLZYS_F).win, theWin_Info->log_zero_sym_y, NULL );

	if( !theWin_Info->polarFlag ){
		d3str( buf, d3str_format, theWin_Info->powXFlag );
		xtb_ti_set( AF(SNTPX_F).win, buf, (xtb_data) &powXFlag );
	}
	else{
		d3str( buf, d3str_format, theWin_Info->powAFlag );
		xtb_ti_set( AF(SNTPX_F).win, buf, (xtb_data) &powAFlag );
	}
	xtb_bt_set( AF(SNTPF_F).win, theWin_Info->polarFlag, NULL );
	d3str( buf, d3str_format, theWin_Info->powYFlag );
	xtb_ti_set( AF(SNTPY_F).win, buf, NULL );
	d3str( buf, d3str_format, theWin_Info->radix );
	xtb_ti_set( AF(SNTPB_F).win, buf, NULL );
	d3str( buf, d3str_format, theWin_Info->radix_offset );
	xtb_ti_set( AF(SNTPO_F).win, buf, NULL );
	xtb_bt_set( AF(SNTAYF_F).win, theWin_Info->absYFlag, NULL );

	xtb_bt_set( AF(SNTLX_F).win, theWin_Info->logXFlag && theWin_Info->logXFlag!= 3, NULL);
	xtb_bt_set( AF(SNTLY_F).win, theWin_Info->logYFlag && theWin_Info->logYFlag!= 3, NULL);
	xtb_bt_set( AF(SNTSX_F).win, theWin_Info->sqrtXFlag, NULL);
	xtb_bt_set( AF(SNTSY_F).win, theWin_Info->sqrtYFlag, NULL);
	xtb_bt_set( AF(USER_F).win, theWin_Info->win_geo.user_coordinates, NULL );

	xtb_bt_set2( AF(NOBB_F).win, theWin_Info->win_geo.nobb_coordinates,
		(theWin_Info->win_geo.nobb_range_X || theWin_Info->win_geo.nobb_range_Y)? 1 : 0, NULL );

	xtb_bt_set( AF(PROCB_F).win, theWin_Info->process_bounds, NULL);
/* 	if( theWin_Info->transform.x_len || theWin_Info->transform.y_len ){	*/
/* 		xtb_bt_set( AF(TRAX_F).win, theWin_Info->transform_axes, NULL);	*/
/* 	}	*/
/* 	else{	*/
		xtb_bt_set( AF(TRAX_F).win, (theWin_Info->transform_axes> 0)? 1 : 0, NULL);
/* 	}	*/
	xtb_bt_set( AF(RAWD_F).win, theWin_Info->raw_display, NULL);

	if( setNumber+ theWin_Info->ulabels!= SD_ULabels ){
		SD_ULabels= setNumber+ theWin_Info->ulabels;
		xtb_sr_set_scale( AF(SNNRSL_F).win, (double) 0, (double) setNumber+ theWin_Info->ulabels- 1, NULL);
		format_SD_Dialog( &SD_Dialog, 0 );
	}
	xtb_sr_set( AF(SNNRSL_F).win, (double) data_sn_number, NULL);

	if( data_sn_number>= setNumber+ theWin_Info->ulabels ){
		data_sn_number= 0;
		xtb_ti_set( SD_info->sn_number, "", NULL );
		  /* Simulate input of character '0' to an empty legendnumber
		   \ box to correctly initialise all boxes depending on
		   \ this number.
		   */
		snn_fun( SD_info->sn_number, '0', data_sn_number_buf, NULL);
	}
	if( data_sn_number>= setNumber && data_sn_number< setNumber+ theWin_Info->ulabels ){
	  UserLabel *ul= GetULabelNr( data_sn_number- setNumber);
		Data_SN_Number( data_sn_number_buf );
		xtb_ti_set( AF(SNNR_F).win, data_sn_number_buf, (xtb_data) 0);
		d3str( buf, d3str_format, ul->x1 );
		xtb_ti_set( AF(LBLX1_F).win, buf, NULL );
		d3str( buf, d3str_format, ul->y1 );
		xtb_ti_set( AF(LBLY1_F).win, buf, NULL );
		d3str( buf, d3str_format, ul->x2 );
		xtb_ti_set( AF(LBLX2_F).win, buf, NULL );
		d3str( buf, d3str_format, ul->y2 );
		xtb_ti_set( AF(LBLY2_F).win, buf, NULL );
		sprintf( buf, "%d", ul->vertical );
		xtb_ti_set( AF(LSFN_F).win, buf, NULL );
		sprintf( buf, "%d", ul->set_link);
		xtb_ti_set( AF(LBLSL_F).win, buf, NULL );
		sprintf( buf, "%d", ul->pnt_nr);
		xtb_ti_set( AF(LBLVL_F).win, buf, NULL );
		strncpy( data_legend_buf, LegendorTitle(data_sn_number, LegendFunctionNr), SD_LMaxBufSize);
		xtb_bt_set( AF(SNDS_F).win, ul->do_draw, (xtb_data) 0);
		xtb_bt_set( AF(SNSRAWD_F).win, !ul->do_transform, (xtb_data)0 ); 
		xtb_bt_set( AF(SNSPADD_F).win, 0, (xtb_data)0 ); 
		xtb_ti_set( AF(SNXC_F).win, "-", NULL );
		xtb_ti_set( AF(SNYC_F).win, "-", NULL );
		xtb_ti_set( AF(SNEC_F).win, "-", NULL );
		xtb_ti_set( AF(SNVC_F).win, "-", NULL );
		xtb_disable( AF(SNNC_F).win );
		xtb_ti_set( AF(SNNC_F).win, "-", NULL );
		xtb_bt_set_text( AF(SNFNS_F).win, ul->nobox, "nobox", (xtb_data) 0);
		d3str( buf, d3str_format, ul->lineWidth );
		xtb_ti_set( AF(SNLW_F).win, buf, NULL );
	}
	else if( data_sn_number>= 0 && data_sn_number< setNumber ){
	  extern double psm_base, psm_incr;
	  DataSet *this_set= &AllSets[data_sn_number];
		Data_SN_Number( data_sn_number_buf );
		xtb_ti_set( AF(SNNR_F).win, data_sn_number_buf, (xtb_data) 0);
		strncpy( data_legend_buf, LegendorTitle(data_sn_number, LegendFunctionNr), SD_LMaxBufSize);
		sprintf( data_sn_lineWidth_buf, "%g", (data_sn_lineWidth= this_set->lineWidth) );
		sprintf( data_sn_linestyle_buf, "%d", (data_sn_linestyle= this_set->linestyle) );

		d3str( buf, d3str_format, this_set->markSize );
		sprintf( AF(SNSMSz_F).description, MARKERSIZEDESCR, psm_base+ (int)(data_sn_number/internal_psMarkers)* psm_incr, '\0' );
		sprintf( data_sn_markstyle_buf, "%d", (data_sn_markstyle= abs(this_set->markstyle)) );

		sprintf( data_sn_elineWidth_buf, "%g", (data_sn_elineWidth= this_set->elineWidth) );
		sprintf( data_sn_elinestyle_buf, "%d", (data_sn_elinestyle= this_set->elinestyle) );

		sprintf( data_sn_plot_interval_buf, "%d", (data_sn_plot_interval= this_set->plot_interval) );
		sprintf( data_sn_adorn_interval_buf, "%d", (data_sn_adorn_interval= this_set->adorn_interval) );

		xtb_ti_set( AF(SNLW_F).win, data_sn_lineWidth_buf, (xtb_data) 0);
		xtb_ti_set( AF(SNLS_F).win, data_sn_linestyle_buf, (xtb_data) 0);
		xtb_ti_set( AF(SNSMSz_F).win, buf, (xtb_data) 0);
		xtb_ti_set( AF(SNSMS_F).win, data_sn_markstyle_buf, (xtb_data) 0);
		xtb_ti_set( AF(SNELW_F).win, data_sn_elineWidth_buf, (xtb_data) 0);
		xtb_ti_set( AF(SNELS_F).win, data_sn_elinestyle_buf, (xtb_data) 0);
		if( this_set->ebarWidth_set ){
			xtb_ti_set( AF(SNEBW_F).win, d3str( buf, d3str_format, this_set->ebarWidth), NULL );
		}
		else{
			xtb_ti_set( AF(SNEBW_F).win, "NaN", NULL );
		}
		xtb_ti_set( AF(SNPLI_F).win, data_sn_plot_interval_buf, (xtb_data) 0);
		xtb_ti_set( AF(SNAdI_F).win, data_sn_adorn_interval_buf, (xtb_data) 0);
		if( this_set->barBase_set ){
			xtb_ti_set( AF(SNBPB_F).win, d3str( buf, d3str_format, this_set->barBase), NULL );
		}
		else{
			xtb_ti_set( AF(SNBPB_F).win, "NaN", NULL );
		}
		if( this_set->barWidth_set ){
			xtb_ti_set( AF(SNBPW_F).win, d3str( buf, d3str_format, this_set->barWidth), NULL );
		}
		else{
			xtb_ti_set( AF(SNBPW_F).win, "NaN", NULL );
		}
		xtb_ti_set( AF(SNBPT_F).win, d3str( buf, "%g", this_set->barType), NULL );
		xtb_bt_set( AF(SNNL_F).win, noLines_Value(), (xtb_data)0 ); 
		xtb_bt_set( AF(SNFLT_F).win, floating_Value(), (xtb_data)0 ); 
		xtb_bt_set( AF(SNBF_F).win, barFlag_Value(), (xtb_data)0 ); 
		xtb_bt_set( AF(SNMF_F).win, markFlag_Value(), (xtb_data)0 ); 
		switch( pixelMarks_Value() ){
			case 1:
				xtb_bt_set_text( AF(SNPM_F).win, 1, "Dts", (xtb_data) 0);
				break;
			case 2:
				xtb_bt_set_text( AF(SNPM_F).win, 1, "Blbs", (xtb_data) 0);
				break;
			case 0:
				xtb_bt_set_text( AF(SNPM_F).win, 0, "Smbl", (xtb_data) 0);
				break;
		}
		if( CheckMask(this_set->valueMarks, VMARK_ON) ){
			xtb_enable( AF(SNVFM_F).win );
			xtb_enable( AF(SNVRM_F).win );
		}
		else{
			xtb_disable( AF(SNVFM_F).win );
			xtb_disable( AF(SNVRM_F).win );
		}
		xtb_bt_set( AF(SNVMF_F).win, CheckMask(this_set->valueMarks, VMARK_ON), (xtb_data) 0 );
		xtb_bt_set( AF(SNVFM_F).win, CheckMask(this_set->valueMarks, VMARK_FULL), (xtb_data) 0 );
		xtb_bt_set( AF(SNVRM_F).win, CheckMask(this_set->valueMarks, VMARK_RAW), (xtb_data) 0 );

		xtb_bt_set( AF(SNSE_F).win, use_error_Value(), (xtb_data)0 ); 
		xtb_bt_set( AF(OWM_F).win, overwrite_marks_Value(), (xtb_data)0 ); 
		xtb_bt_set( AF(SNPM_F).win, pixelMarks_Value(), (xtb_data)0 ); 
		xtb_bt_set( AF(SNDS_F).win, draw_set_Value(), (xtb_data) 0);
		xtb_bt_set( AF(SNMS_F).win, mark_set_Value(), (xtb_data) 0);
		xtb_bt_set( AF(SNHL_F).win, highlight_Value(), (xtb_data) 0);
		xtb_bt_set( AF(SNSL_F).win, show_legend_Value(), (xtb_data) 0);
		xtb_bt_set( AF(SNSLLS_F).win, show_llines_Value(), (xtb_data) 0);
		xtb_bt_set( AF(SNSRAWD_F).win, raw_display_Value(), (xtb_data)0 ); 
		xtb_bt_set( AF(SNSPADD_F).win, points_added_Value(), (xtb_data)0 ); 

		xtb_bt_set( AF(SNSSARROW_F).win, start_arrow_Value(), (xtb_data)0 ); 
		xtb_bt_set( AF(SNSEARROW_F).win, end_arrow_Value(), (xtb_data)0 ); 
		d2str( (data_start_arrow_orn= this_set->sarrow_orn), NULL, data_start_arrow_orn_buf );
		xtb_ti_set( AF(SNSARRORN_F).win, data_start_arrow_orn_buf, (xtb_data)0 ); 
		d2str( (data_end_arrow_orn= this_set->earrow_orn), NULL, data_end_arrow_orn_buf );
		xtb_ti_set( AF(SNEARRORN_F).win, data_end_arrow_orn_buf, (xtb_data)0 ); 

		xtb_enables( AF(LBLX2_F).win, AF(LBLY2_F).win, AF(LSFN_F).win, AF(LBLSL_F).win, AF(VTTYPE_F).win,
			AF(LBLVL_F).win, AF(SNXC_F).win, AF(SNYC_F).win, AF(SNEC_F).win, AF(SNVC_F).win,
			(Window) 0 );

		if( this_set->show_legend ){
			d3str( buf, d3str_format, theWin_Info->_legend_ulx );
			xtb_ti_set( AF(LBLX1_F).win, buf, NULL );
			d3str( buf, d3str_format, theWin_Info->_legend_uly );
			xtb_ti_set( AF(LBLY1_F).win, buf, NULL );
		}
		else if( theWin_Info->error_type[data_sn_number]== INTENSE_FLAG ){
			d3str( buf, d3str_format, theWin_Info->IntensityLegend._legend_ulx );
			xtb_ti_set( AF(LBLX1_F).win, buf, NULL );
			d3str( buf, d3str_format, theWin_Info->IntensityLegend._legend_uly );
			xtb_ti_set( AF(LBLY1_F).win, buf, NULL );
		}
		else{
			xtb_ti_set( AF(LBLX1_F).win, "-", NULL );
			xtb_ti_set( AF(LBLY1_F).win, "-", NULL );
		}
		sprintf( buf, "%d", this_set->error_point );
		xtb_ti_set( AF(LBLX2_F).win, buf, NULL );
		sprintf( buf, "%d", this_set->NumObs );
		xtb_ti_set( AF(LBLY2_F).win, buf, NULL );
		sprintf( buf, "%d", theWin_Info->fileNumber[data_sn_number] );
		xtb_ti_set( AF(LSFN_F).win, buf, NULL );
		sprintf( buf, "%d", this_set->set_link );
		xtb_ti_set( AF(LBLSL_F).win, buf, NULL );
		new_file= theWin_Info->new_file[data_sn_number];
		xtb_bt_set_text( AF(SNFNS_F).win, new_file, "split", (xtb_data) 0);
		switch( this_set->vectorType ){
			case 0:
			case 2:
			default:
				sprintf( buf, "%d,%s", this_set->vectorType,
					d2str( this_set->vectorLength, d3str_format, NULL )
				);
				break;
			case 1:
			case 3:
			case 4:
				sprintf( buf, "%d,%s,%s,%s", this_set->vectorType,
					d2str( this_set->vectorLength, d3str_format, NULL ),
					d2str( this_set->vectorPars[0], d3str_format, NULL ),
					d2str( this_set->vectorPars[1], d3str_format, NULL )
				);
				break;
		}
		xtb_ti_set( AF(LBLVL_F).win, buf, NULL );

		sprintf( buf, sncols, this_set->ncols );
		xtb_to_set( AF(SNCOLS_F).win, buf );
		sprintf( buf, "%d", theWin_Info->xcol[data_sn_number] );
		xtb_ti_set( AF(SNXC_F).win, buf, NULL );
		sprintf( buf, "%d", theWin_Info->ycol[data_sn_number] );
		xtb_ti_set( AF(SNYC_F).win, buf, NULL );
		sprintf( buf, "%d", theWin_Info->ecol[data_sn_number] );
		xtb_ti_set( AF(SNEC_F).win, buf, NULL );
		sprintf( buf, "%d", theWin_Info->lcol[data_sn_number] );
		xtb_ti_set( AF(SNVC_F).win, buf, NULL );
		sprintf( buf, "%d", AllSets[data_sn_number].Ncol );
		xtb_ti_set( AF(SNNC_F).win, buf, NULL );
#if ADVANCED_STATS == 2
		xtb_enable( AF(SNNC_F).win );
#else
		xtb_disable( AF(SNNC_F).win );
#endif
	}
	else if( data_sn_number< 0 ){
		if( data_sn_number== -2 || data_sn_number== -1 ){
		  char buf2[64];
			if( data_sn_number== -2 ){
				d3str( buf, d3str_format, theWin_Info->xname_x );
				d3str( buf2, d3str_format, theWin_Info->xname_y );
			}
			else{
				d3str( buf, d3str_format, theWin_Info->yname_x );
				d3str( buf2, d3str_format, theWin_Info->yname_y );
			}
			xtb_ti_set( AF(LBLX1_F).win, buf, NULL );
			xtb_ti_set( AF(LBLY1_F).win, buf2, NULL );
		}
		xtb_disables( AF(LBLX2_F).win, AF(LBLY2_F).win, AF(LSFN_F).win, AF(LBLSL_F).win, AF(VTTYPE_F).win,
			AF(LBLVL_F).win, AF(SNXC_F).win, AF(SNYC_F).win, AF(SNEC_F).win, AF(SNVC_F).win,
			(Window) 0 );
#if ADVANCED_STATS == 2
		xtb_disable( AF(SNNC_F).win );
		xtb_ti_set( AF(SNNC_F).win, "-", NULL );
#endif
		Data_SN_Number( data_sn_number_buf );
		xtb_ti_set( AF(SNNR_F).win, data_sn_number_buf, (xtb_data) 0);
		get_data_legend_buf();
	}
	else{
		Data_SN_Number( data_sn_number_buf );
		xtb_ti_set( AF(SNNR_F).win, data_sn_number_buf, (xtb_data) 0);
		get_data_legend_buf();
	}
/* 	if( data_sn_number< 0 ){	*/
		xtb_enable( AF(SNLEGEDIT_F).win );
/* 	}	*/
/* 	else{	*/
/* 		xtb_disable( AF(SNLEGEDIT_F).win );	*/
/* 	}	*/
	xtb_to_set( AF(SNpLEG_F).win, LegendorTitle(data_sn_number, pLegFunNr) );
	xtb_ti_set( SD_info->sn_legend, data_legend_buf, (xtb_data) 0 );
	xtb_ti_set( AF(SNFN_F).win, Data_fileName(), (xtb_data) 0);
	xtb_bt_set( AF(SNDAS_F).win, theWin_Info->ctr_A, NULL);
	xtb_bt_set( AF(RFRS_F).win, Raw_NewSets, NULL);

	xtb_bt_set( AF(ADH_F).win, theWin_Info->AlwaysDrawHighlighted, NULL);

	xtb_bt_set( AF(AUTOREDRAW_F).win, autoredraw, NULL );
	if( autoredraw && allow_auto && theWin_Info->redraw ){
	  static char active= 0;
		if( !active ){
			active= 1;
			SD_redraw_fun( redrawbtn.win, 1, rinfo);
			active= 0;
		}
	}
	xtb_bt_set( redrawbtn.win, theWin_Info->redraw, NULL);

	if( !CheckMask( xtb_modifier_state, ControlMask) ){
		xtb_XSync( disp, False);
	}
	return(1);
}

/* Redraw the window	*/
static xtb_hret SD_redraw_fun(Window win, int val, xtb_data info)
/* 
 \ This is the handler for the redraw button.
 \ It sets the redraw field of theWin_Info to 1
 */
{  extern int DrawWindow();
   extern Boolean dialog_redraw;
   extern Boolean set_sn_nr;
   int ss= -1;

	if( !theWin_Info ){
		return( XTB_STOP );
	}
	
	if( info== &autoredraw ){
		autoredraw= !autoredraw;
		set_changeables(0,True);
		return( XTB_HANDLED );
	}

	if( !theWin_Info->redraw ){
		theWin_Info->redraw= 1;
	}
	else{
		theWin_Info->printed= 0;
	}
	theWin_Info->halt= 0;
	theWin_Info->draw_count= 0;
	if( win== redrawbtn.win ){
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
/* 			ss= Synchro_State;	*/
/* 			Synchro_State= 0;	*/
/* 			HardSynchro= True;	*/
/* 			X_Synchro(theWin_Info);	*/
			ss= theWin_Info->data_sync_draw;
			theWin_Info->data_sync_draw= True;
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
	if( !theWin_Info ){
		return( XTB_STOP );
	}
	dialog_redraw= False;
	if( (win== redrawbtn.win) ){
		if( ss!= -1 ){
/* 			Synchro_State= ss;	*/
/* 			X_Synchro(theWin_Info);	*/
			theWin_Info->data_sync_draw= ss;
		}
		xtb_bt_set( win, 0, info);
	}
	set_changeables(0,True);
	return( XTB_HANDLED );
}

xtb_hret SD_quit_fun( Window win, int val, xtb_data info)
{ 
	if( !theWin_Info ){
		return( XTB_STOP );
	}

	if( CheckMask( xtb_modifier_state, Mod1Mask|ShiftMask) ){
		if( CheckMask( xtb_modifier_state, ControlMask) ){
			xtb_bt_set( win, 1, NULL);
			ExitProgramme(-1);
		}
		else{
			DelWindow( theWin_Info->window, theWin_Info );
		}
	}
	else{
/* 		xtb_error_box( theWin_Info->window,	*/
/* 			"Sorry, you must hold down the Mod1 and Shift keys to activate this function!\n", "Notice" );	*/
		if( xtb_error_box( theWin_Info->window,
			"\001This will close all windows and quit XGraph\n"
			"\001are you sure?\n",
			"Warning"
			)> 0
		){
			xtb_bt_set( win, 1, NULL);
			ExitProgramme(-1);
		}
	}
	return( XTB_HANDLED );
}

static xtb_hret SD_update(Window win, int val, xtb_data info)
/* 
 \ This is the handler for the update button.
 */
{
	if( !theWin_Info ){
		return( XTB_STOP );
	}

	set_changeables(0,True);
	return( XTB_HANDLED );
}

/* Swap information between bounds and pure_bounds structures	*/
static xtb_hret SD_swap_pure(Window win, int val, xtb_data info)
{ Local_Boundaries lwg= theWin_Info->win_geo.bounds;
	if( !theWin_Info ){
		return( XTB_STOP );
	}

	if( !theWin_Info->redraw ){
		theWin_Info->redraw= 1;
		if( theWin_Info->redrawn== -3 ){
			theWin_Info->redrawn= 0;
		}
	}
	theWin_Info->printed= 0;
	theWin_Info->win_geo.bounds= theWin_Info->win_geo.pure_bounds;
	theWin_Info->win_geo.pure_bounds= lwg;
	xtb_bt_set( win, theWin_Info->win_geo.bounds.pure, info);
	set_changeables(0,True);
	return( XTB_HANDLED );
}

static xtb_hret SD_dynfunc( Window win, int bval, xtb_data info)
{
	if( !theWin_Info ){
		return( XTB_STOP );
	}

	SD_dynamic_resize= !SD_dynamic_resize;
	update_SD_size();
	set_changeables(0,True);
	format_SD_Dialog( &SD_Dialog, 0 );

	return( XTB_HANDLED );
}

static int print_it= 0;

static xtb_hret print_fun(win, bval, info)
Window win;			/* Button window     */
int bval;			/* Button value      */
xtb_data info;			/* Local button info */
{
	if( !theWin_Info ){
		return( XTB_STOP );
	}

	print_it= 1;
	SD_Dialog.mapped= -1;
    CloseSD_Dialog( &SD_Dialog);
	return( XTB_HANDLED );
}

/* reverse all drawn sets (drawing those not currently drawn)	*/
static xtb_hret sdds_rds_fun(win, bval, info)
Window win;			/* Button window     */
int bval;			/* Button value      */
xtb_data info;			/* Local button info */
{  int i;
	if( !theWin_Info ){
		return( XTB_STOP );
	}

	xtb_bt_set( win, 1, NULL);
	if( win== AF(SNRDS_F).win ){ 
		for( i= 0; i< setNumber; i++ ){
			AllSets[i].draw_set= !AllSets[i].draw_set;
			theWin_Info->draw_set[i]= !theWin_Info->draw_set[i];
		}
		xtb_bt_set( win, 0, NULL);
	}
	else{
		for( i= 0; i< setNumber; i++ ){
			AllSets[i].draw_set= (AllSets[i].draw_set<0)? 0 : (AllSets[i].draw_set==0)? -1 : 1;
			theWin_Info->draw_set[i]= (theWin_Info->draw_set[i]<0)? 0 : (theWin_Info->draw_set[i]==0)? -1 : 1;
		}
		theWin_Info->ctr_A= ! theWin_Info->ctr_A;
	}
	if( data_sn_number>= 0 && data_sn_number< setNumber ){
		xtb_bt_set( AF(SNDS_F).win, draw_set_Value(), 0 );
		xtb_bt_set( AF(SNMS_F).win, mark_set_Value(), (xtb_data) 0);
		xtb_bt_set( AF(SNHL_F).win, highlight_Value(), (xtb_data) 0);
	}
	xtb_bt_set( AF(SNDAS_F).win, theWin_Info->ctr_A, NULL);
	SD_redraw_fun( 0, bval, info );
	if( !theWin_Info ){
		return( XTB_STOP );
	}
	return( XTB_HANDLED );
}

int set_shift_left, set_shift_right;

static xtb_hret sdds_shiftset_fun( Window win, int bval, xtb_data info)
{ int new;
  int all= False, dsn, n= 0, nr, N, i, ard= autoredraw;
  Boolean aaa= False;
	if( !theWin_Info ){
		return( XTB_STOP );
	}

	xtb_bt_set( win, 1, NULL);

	if( apply_to_marked || apply_to_hlt || apply_to_drawn || apply_to_new || apply_to_src ){
		_apply_to_drawn= apply_to_drawn;
		apply_to_drawn= 0;
		_apply_to_marked= apply_to_marked;
		apply_to_marked= 0;
		_apply_to_hlt= apply_to_hlt;
		apply_to_hlt= 0;
		_apply_to_new= apply_to_new;
		apply_to_new= 0;
		_apply_to_src= apply_to_src;
		apply_to_src= 0;
		apply_to_all= True;
		aaa= True;
	}
	if( apply_to_all ){
		nr= (apply_to_rest)? data_sn_number : 0;
		N= (apply_to_prev)? MIN(data_sn_number+1,setNumber) : setNumber;
		apply_to_all= 0;
		apply_to_rest= 0;
		apply_to_prev= 0;
		apply_to_src= 0;
		all= 1;
		dsn= data_sn_number;
		xtb_bt_set( AF(SNATA_F).win, 0, NULL );
		xtb_bt_set( AF(SNATP_F).win, 0, NULL );
		xtb_bt_set( AF(SNATR_F).win, 0, NULL );
		autoredraw= False;
	}
	else{
		nr= data_sn_number;
		N= nr+ 1;
	}

	if( all ){
		for( i= nr; i< N; i++ ){
			if( i>= 0 && apply_ok(i) ){
				AllSets[i].draw_set= 1;
				n+= 1;
			}
			else{
				AllSets[i].draw_set= 0;
			}
		}
		if( ShiftDataSets_Drawn( theWin_Info, (info== &set_shift_left)? -1 : 1, CheckMask(xtb_modifier_state, Mod1Mask), True )> 0
		){
			if( CheckMask(xtb_modifier_state, Mod1Mask) ){
				new= (info== &set_shift_left)? 0 : setNumber-1;
			}
			else{
				CLIP_EXPR( new, data_sn_number+ (info== &set_shift_left)? -1 : 1, 0, setNumber-1 );
			}
		}
	}
	else{
		new= ShiftDataSet( data_sn_number, (info== &set_shift_left)? -1 : 1, CheckMask(xtb_modifier_state, Mod1Mask), True );
	}
	if( new>= 0 && new< setNumber ){
		data_sn_number= new;
		set_changeables(2,True);
	}

	if( aaa ){
		_apply_to_drawn= 0;
		_apply_to_marked= 0;
		_apply_to_hlt= 0;
		_apply_to_new= 0;
		_apply_to_src= 0;
		xtb_bt_set( AF(SNATD_F).win, 0, NULL );
		xtb_bt_set( AF(SNATM_F).win, 0, NULL );
		xtb_bt_set( AF(SNATH_F).win, 0, NULL );
		xtb_bt_set( AF(SNATN_F).win, 0, NULL );
		xtb_bt_set( AF(SNATS_F).win, 0, NULL );
	}
	autoredraw= ard;

	xtb_bt_set( win, 0, NULL);
	if( !theWin_Info ){
		return( XTB_STOP );
	}
	return( XTB_HANDLED );
}

extern int use_errors, triangleFlag, error_regionFlag;

int get_error_type( LocalWin *wi, int snr )
{
	if( wi->error_type[snr]== -1 ){
		if( error_type== -1 ){
			error_type= !no_errors;
		}
		return( error_type );
	}
	else{
		return( wi->error_type[snr] );
	}
}

extern int FitOnce;

int set_error_type( LocalWin *wi, int snr, int *type, int no_fit )
{ int change= (wi->error_type[snr]!= *type);
	switch( *type){
		default:
			break;
		case -1:
			*type= 0;
		case 0:
		case 5:
		case 6:
			wi->error_type[snr]= *type;
			if( AllSets[snr].error_type== -1 ){
				wi->redraw+= change;
				AllSets[snr].error_type= *type;
			}
			wi->use_errors= 0;
			wi->triangleFlag= 0;
			wi->error_region= 0;
			wi->vectorFlag= 0;
			break;
		case 1:
		case 7:
			wi->error_type[snr]= *type;
			if( AllSets[snr].error_type== -1 ){
				wi->redraw+= change;
				AllSets[snr].error_type= *type;
			}
			wi->use_errors= 1;
			wi->triangleFlag= 0;
			wi->error_region= 0;
			wi->vectorFlag= 0;
			break;
		case 2:
			wi->error_type[snr]= *type;
			if( AllSets[snr].error_type== -1 ){
				wi->redraw+= change;
				AllSets[snr].error_type= *type;
			}
			wi->use_errors= 2;
			wi->triangleFlag= 1;
			wi->error_region= 0;
			wi->vectorFlag= 0;
			break;
		case 3:
			if( !lowYsegs || !highYsegs ){
				if( !lowYsegs && !highYsegs ){
					do_error( "No memory for error region\n" );
					*type= 1;
					return( SD_set_errb(type) );
				}
				else{
					do_error( "Not enough memory: error region will be incomplete\n");
				}
			}
			wi->error_type[snr]= *type;
			if( AllSets[snr].error_type== -1 ){
				wi->redraw+= change;
				AllSets[snr].error_type= *type;
			}
			wi->use_errors= 3;
			wi->triangleFlag= 0;
			wi->error_region= 1;
			wi->vectorFlag= 0;
			break;
		case 4:
			wi->error_type[snr]= *type;
			if( AllSets[snr].error_type== -1 ){
				wi->redraw+= change;
				AllSets[snr].error_type= *type;
			}
			wi->use_errors= 4;
			wi->triangleFlag= 0;
			wi->error_region= 0;
			if( !wi->vectorFlag && !no_fit ){
				wi->FitOnce= 1;
			}
			wi->vectorFlag= 1;
			break;
	}
	return( wi->use_errors);
}

int SD_get_errb()
{
	if( !theWin_Info || data_sn_number< 0 || data_sn_number>= setNumber ){
		return(0);
	}
	return( get_error_type( theWin_Info, data_sn_number ) );
}

int SD_set_errb( int *type)
{
	if( data_sn_number< 0 || data_sn_number>= setNumber ){
		return( 0 );
	}
	return( set_error_type( theWin_Info, data_sn_number, type, CheckMask(xtb_modifier_state, Mod1Mask) ) );
}

/* specify type of errorbars (or none); buttonrow handler	*/
static xtb_hret errb_fun(Window win, int old, int new, xtb_data info)
/* Window win;		Button row window */
/* int old;			Previous button   */
/* int new;			Current button    */
{ struct d_info *data = (struct d_info *) info;
  int errb_type;
  int nn= 0, dsn, nr, N, i, ard= autoredraw;
  Boolean aaa= False;
	if( !theWin_Info ){
		return( XTB_STOP );
	}

	errb_type= error_num2type[ xtb_br_get( data->errb) ];
	dsn= data_sn_number;

	if( apply_to_marked || apply_to_hlt || apply_to_drawn || apply_to_new || apply_to_src ){
		_apply_to_drawn= apply_to_drawn;
		apply_to_drawn= 0;
		_apply_to_marked= apply_to_marked;
		apply_to_marked= 0;
		_apply_to_hlt= apply_to_hlt;
		apply_to_hlt= 0;
		_apply_to_new= apply_to_new;
		apply_to_new= 0;
		_apply_to_src= apply_to_src;
		apply_to_src= 0;
		apply_to_all= True;
		aaa= True;
	}
	if( apply_to_all ){
		nr= (apply_to_rest)? data_sn_number : 0;
		N= (apply_to_prev)? MIN(data_sn_number+1,setNumber) : setNumber;
		apply_to_all= 0;
		apply_to_rest= 0;
		apply_to_prev= 0;
		apply_to_src= 0;
		xtb_bt_set( AF(SNATA_F).win, 0, NULL );
		xtb_bt_set( AF(SNATP_F).win, 0, NULL );
		xtb_bt_set( AF(SNATR_F).win, 0, NULL );
		autoredraw= False;
	}
	else{
		nr= data_sn_number;
		N= nr+ 1;
	}

	for( i= nr; i< N; i++ ){
		data_sn_number= i;
		if( apply_ok(i) ){
			SD_set_errb( &errb_type);
			set_changeables(2,True);
			nn+= 1;
		}
	}

	data_sn_number= dsn;

	if( aaa ){
		_apply_to_drawn= 0;
		_apply_to_marked= 0;
		_apply_to_hlt= 0;
		_apply_to_new= 0;
		_apply_to_src= 0;
		xtb_bt_set( AF(SNATD_F).win, 0, NULL );
		xtb_bt_set( AF(SNATM_F).win, 0, NULL );
		xtb_bt_set( AF(SNATH_F).win, 0, NULL );
		xtb_bt_set( AF(SNATN_F).win, 0, NULL );
		xtb_bt_set( AF(SNATS_F).win, 0, NULL );
	}
	autoredraw= ard;
	if( nn ){
		set_changeables( 2 ,True);
	}

	if( old!= new ){
		theWin_Info->redraw= 1;
		if( theWin_Info->redrawn== -3 ){
			theWin_Info->redrawn= 0;
		}
		theWin_Info->printed= 0;
	}
	return( XTB_HANDLED);
}

static xtb_hret sn_legend_fun(win, old, new, info)
Window win;			/* Button row window */
int old;			/* Previous button   */
int new;			/* Current button    */
xtb_data info;			/* User data         */
{ int type;
	if( !theWin_Info ){
		return( XTB_STOP );
	}

	if( (type= xtb_br_get( AF(SNLEGFUN_F).win ))>= 0 ){
		if( LegendFunctionNr!= pLegFunNr ){
			pLegFunNr= LegendFunctionNr;
			xfree( AF(SNpLEG_F).description );
			xtb_describe( aF(SNpLEG_F), LegendFTypes[pLegFunNr] );
		}
		LegendFunctionNr= type;
		if( LegendFunctionNr== 7 ){
			xtb_disable( SD_info->sn_legend );
		}
		else{
			xtb_enable( SD_info->sn_legend );
		}
	}
	if( old!= new ){
		do_update_SD_size= True;
	}
	return( XTB_HANDLED);
}

int SD_tabset= 0;
int tabselect( int id, int tab, int select )
{
	if( id== BAR_F && (select== 2 || select== 3) ){
		return(0);
	}
	else if( select== 3 ){
		return( (tab<=0)? 1 : 0 );
	}
	else if( select> 0 ){
		return( tab== select || tab<= 0 );
	}
	else{
		return( 1 );
	}
}

static xtb_hret tabs_fun( Window win, int old, int new, xtb_data info)
{
	if( !theWin_Info ){
		return( XTB_STOP );
	}

	if( (SD_tabset= xtb_br_get( AF(TABSELECT_F).win ))>= 0 ){
		if( SD_tabset== 4 ){
			xtb_br_set( AF(TABSELECT_F).win, (SD_tabset= old) );
			print_it= 1;
			SD_Dialog.mapped= -1;
			CloseSD_Dialog( &SD_Dialog);
			return( XTB_HANDLED );
		}
		xtb_select_frames_tabset( LAST_F, sd_af, SD_tabset, tabselect );
	}
	else{
		xtb_br_set( AF(TABSELECT_F).win, 0 );
		xtb_select_frames_tabset( LAST_F, sd_af, 0, tabselect );
		new= 0;
		old= 1;
	}
	if( old!= new ){
		do_update_SD_size= True;
		if( !update_SD_size() ){
		  /* But we do! :) */
			format_SD_Dialog( &SD_Dialog, 0 );
		}
		set_changeables(2,True);
	}
	return( XTB_HANDLED);
}

#define BAR_SLACK	10

static Boolean JustResized= False;

int format_SD_Dialog( xtb_frame *frame, int discard )
{
    xtb_fmt *def, *snarea;
	xtb_fmt *tabarea;
    XSizeHints hints;
	int i, max_width;
    char *desc= AF(BAR_F).description;
	static char active= 0;
	XEvent evt;

	if( active ){
		return(1);
	}
	active++;

      /* Dividing bar */
    max_width = 0;
	  /* BAR_F must be among the last controls! */
    for (i = 0;  i < ((int) BAR_F);  i++) {
		if( i!= BAR_F && i!= VBAR_F && i!= TBAR_F && AF(i).width > max_width ){
			max_width = AF(i).width;
		}
    }
	if( AF(BAR_F).mapped ){
	  int ts= AF(BAR_F).tabset;
		if( AF(BAR_F).win ){
			xtb_bk_del( AF(BAR_F).win );
			xfree( AF(BAR_F).framelist );
			AF(BAR_F).win= 0;
		}
		xtb_bk_new( frame->win, max_width - BAR_SLACK, 1, aF(BAR_F));
		AF(BAR_F).description= desc;
		AF(BAR_F).tabset= ts;
	}
	if( AF(TBAR_F).mapped ){
	  int ts= AF(TBAR_F).tabset;
		if( AF(TBAR_F).win ){
			xtb_bk_del( AF(TBAR_F).win );
			xfree( AF(TBAR_F).framelist );
			AF(TBAR_F).win= 0;
		}
		xtb_bk_new( frame->win, max_width - BAR_SLACK, 1, aF(TBAR_F));
		AF(TBAR_F).tabset= ts;
	}

	xtb_fmt_tabset= 0;
	tabarea= xtb_vert( XTB_CENTER, 0, 0,
			xtb_hort( XTB_CENTER, D_HPAD, D_INT, 
				xtb_w(aF(TABSELECT_F)), xtb_w(aF(SDDR_F)), NE),
			xtb_w(aF(VBAR_F)), xtb_w(aF(TBAR_F)), NE);

    snarea = xtb_vert(XTB_CENTER, 0,0,
		xtb_hort(XTB_CENTER_J, 0, 0,
			xtb_vert(XTB_CENTER_J, D_HPAD/2, D_INT/2,
				xtb_hort(XTB_CENTER_J, D_HPAD, D_INT,
					xtb_vert( XTB_LEFT_J, D_VPAD/2, D_INT/2,
						xtb_vert( XTB_CENTER, D_VPAD/2, D_INT/2,
							xtb_hort( XTB_CENTER, D_HPAD, D_INT,
								xtb_w(aF(DEBUG_F)), xtb_w(aF(DEBUGL_F)), xtb_w(aF(SYNC_F)), xtb_w(aF(WDEBUG_F)),
								NE
							), 
							xtb_hort( XTB_CENTER, D_HPAD, D_INT,
								xtb_w(aF(FRACP_F)), xtb_w(aF(D3STRF_F)), xtb_w(aF(DSYNC_F)), xtb_w(aF(DSILENT_F)),
								NE
							), NE
						),
						xtb_w( aF(KEY1_F)),
						xtb_w( aF(KEY2_F)),
/* 
						xtb_w( aF(KEY3_F)),
						xtb_w( aF(KEY4_F)),
 */
						NE
					),
					xtb_vert( XTB_JUST, D_VPAD/2, D_INT/2,
						xtb_hort( XTB_CENTER_J, D_HPAD, D_INT,
							xtb_w( aF(SNOL_F)), xtb_w( aF(SNSF_F)), xtb_w( aF(SNSLL_F)),
							xtb_hort( XTB_CENTER, 0, 0, xtb_w( aF(SOAG_F)), xtb_w( aF(SNAXF_F)), NE),
								xtb_w( aF(SNBBF_F)),
							xtb_hort( XTB_CENTER, 0, 0, xtb_w( aF(SNHTKF_F)), xtb_w( aF(SNVTKF_F)), NE),
							xtb_w( aF(SNZLF_F)), xtb_w( aF(SNT_F)),
							NE
						),
						xtb_hort( XTB_JUST, D_HPAD, D_INT,
							xtb_hort( XTB_CENTER, 0, 0,
								xtb_w( aF(SNLP_F)), xtb_w( aF(SNILP_F)),
								xtb_w( aF(SNXP_F)),  xtb_w( aF(SNYP_F)), xtb_w( aF(SNYV_F)), NE
							),
							xtb_hort( XTB_CENTER, 0, 0, xtb_w( aF(SNL_F)), xtb_w( aF(SOL_F)), NE),
							xtb_w(aF(SNIL_F)), xtb_w( aF(SNNP_F)),
							NE
						),
						xtb_hort(XTB_JUST, D_HPAD, D_INT,
/* 							xtb_w(aF(SNERRB_F)), xtb_w(aF(OERRBROW_F)), xtb_w(aF(SUAE_F)), xtb_w(aF(SNNLB_F)),	*/
							xtb_w( aF(ADH_F)), xtb_w( aF(RFC_F)), xtb_w( aF(RFRS_F)),
							xtb_w( aF(SNRDS_F)), xtb_w( aF(SNDAS_F)),
							xtb_w(aF(SUAE_F)), xtb_w(aF(SNNLB_F)), xtb_w( aF(SNOUL_F)),
							NE
						),
						NE
					),
					xtb_vert(XTB_JUST, D_VPAD/2, D_INT/2,
						xtb_hort(XTB_CENTER, D_HPAD, D_INT,
							xtb_w(aF(SNTPF_F)),
							xtb_vert(XTB_CENTER, D_VPAD/2, D_INT/2,
								xtb_w(aF(SNTPB_F)), xtb_w(aF(SNTPO_F)),
								NE
							),
							xtb_w(aF(SNTAYF_F)),
							NE
						),
						xtb_hort(XTB_CENTER, D_HPAD, D_INT,
						   xtb_w(aF(SNTLX_F)),	xtb_w(aF(SNTSX_F)),	xtb_w(aF(SNTPX_F)),	/* xtb_w(aF(SNTLX2_F)),	*/
						   NE
						),
						xtb_hort(XTB_CENTER, D_HPAD, D_INT,
						   xtb_w(aF(SNTLY_F)),	xtb_w(aF(SNTSY_F)),	xtb_w(aF(SNTPY_F)),	/* xtb_w(aF(SNTLY2_F)),	*/
						   NE
						),
					   NE
					),
					NE
				),
				xtb_vert(XTB_CENTER, D_VPAD/2, D_INT/2,
					xtb_hort(XTB_CENTER, D_HPAD, D_INT,
						xtb_w( aF(BOUNDS_F)),
						xtb_w( aF(AXMIDIG_F)), xtb_w( aF(AXSTR_F)),
						xtb_hort( XTB_CENTER, 0, 0, 
							xtb_w( aF(XFIT_F)), xtb_w( aF(YFIT_F)), xtb_w( aF(PBFIT_F)), xtb_w( aF(FITAD_F)), NE
						),
						xtb_w( aF(ASPECT_F)), xtb_w( aF(XSYMM_F)), xtb_w( aF(YSYMM_F)),
						xtb_w( aF(SNLAV_F)), xtb_w( aF(SNEXA_F)), xtb_w( aF(SNEYA_F)),
						xtb_hort( XTB_CENTER, 0, 0, xtb_w( aF(SNVCXA_F)), xtb_w( aF(SNVCXG_F)), xtb_w( aF(SNVCXL_F)), NE),
						xtb_w( aF(SNVCYA_F)), xtb_w( aF(SNSAVCI_F)), xtb_w( aF(SNVCIA_F)),
						NE
					),
					xtb_hort(XTB_JUST, D_HPAD, D_INT,
						xtb_w( aF(XMIN_F)), xtb_w( aF(YMIN_F)),
/* 						xtb_w( aF(BOUNDSSEP_F)),	*/
						xtb_w( aF(NOBB_F)),
						xtb_w( aF(XMAX_F)), xtb_w( aF(YMAX_F)),
						xtb_w( aF(USER_F)), xtb_w( aF(SWAPPURE_F)), xtb_w( aF(PROCB_F)), xtb_w( aF(RAWD_F)),
						xtb_w( aF(TRAX_F)),
						NE
					),
					NE
				),
/* 
				xtb_hort(XTB_CENTER, D_HPAD, D_INT,
					xtb_w(aF(SNIFLBL_F)),
					xtb_w(aF(SNXIF_F)), xtb_w(aF(SNYIF_F)),
					xtb_w(aF(SNBTLBL_F)), xtb_w(aF(SNBTX_F)), xtb_w(aF(SNBTY_F)),
					xtb_w(aF(SNS2LBL_F)), xtb_w(aF(SNS2X_F)), xtb_w(aF(SNS2Y_F)),
				   NE
				),
 */
				xtb_hort(XTB_JUST, D_HPAD, D_INT,
					xtb_vert( XTB_CENTER, D_VPAD/2, D_INT/2,
						xtb_w(aF(SNIFLBL_F)),
						xtb_hort(XTB_CENTER, D_HPAD, D_INT,xtb_w(aF(SNXIF_F)), xtb_w(aF(SNYIF_F)), NE),
						NE
					),
					xtb_vert( XTB_CENTER, D_VPAD/2, D_INT/2,
						xtb_w(aF(SNS2LBL_F)),
						xtb_hort(XTB_CENTER, D_HPAD, D_INT,xtb_w(aF(SNS2X_F)), xtb_w(aF(SNS2Y_F)), NE),
						NE
					),
					xtb_vert( XTB_CENTER, D_VPAD/2, D_INT/2,
						xtb_w(aF(SNBTLBL_F)),
						xtb_hort(XTB_CENTER, D_HPAD, D_INT,xtb_w(aF(SNBTX_F)), xtb_w(aF(SNBTY_F)), NE),
						NE
					),
				   NE
				),
				xtb_hort(XTB_JUST, D_HPAD, D_INT,
				   xtb_w(aF(SNLZLBL_F)),
				   xtb_w(aF(SNLZXV_F)),	xtb_w(aF(SNLZX_F)), xtb_w(aF(SNLZXMI_F)), xtb_w(aF(SNLZXMA_F)), xtb_w(aF(SNLZXS_F)),
				   xtb_w(aF(SNLZYV_F)),	xtb_w(aF(SNLZY_F)), xtb_w(aF(SNLZYMI_F)), xtb_w(aF(SNLZYMA_F)), xtb_w(aF(SNLZYS_F)),
				   NE
				),

				xtb_hort(XTB_CENTER, D_HPAD, D_INT, xtb_w(aF(BAR_F)), NE),
				xtb_hort(XTB_CENTER, D_HPAD, D_INT, 
					xtb_w(aF(SNNRSL_F)), xtb_w(aF(SNNR_F)), xtb_w(aF(SNNRM_F)), xtb_w(aF(SNNRLBL_F)),
					xtb_w( aF(SNATA_F)), xtb_w( aF(SNATP_F)), xtb_w( aF(SNATR_F)),
					xtb_w( aF(SNATD_F)), xtb_w( aF(SNATM_F)), xtb_w( aF(SNATH_F)),
					xtb_w( aF(SNATN_F)), xtb_w( aF(SNATS_F)),
					NE
				),

				xtb_hort(XTB_CENTER, D_HPAD, D_INT,
					xtb_vert( XTB_CENTER, D_VPAD/2, D_INT/2,
						xtb_hort(XTB_CENTER, D_HPAD, D_INT,
							xtb_vert( XTB_CENTER, D_VPAD/2, D_INT/2,
								xtb_w(aF(SNERRB_F)), xtb_w(aF(OERRBROW_F)), NE
							),
/* 							xtb_w(aF(SNNRLBL_F)),	*/
/* 							xtb_vert( XTB_CENTER, D_VPAD/2, D_INT,	*/
/* 								xtb_hort( XTB_CENTER_J, D_HPAD, D_INT,	*/
/* 									xtb_w(aF(SNNRSL_F)), xtb_w(aF(SNNR_F)), xtb_w(aF(SNNRM_F)),	*/
/* 									NE	*/
/* 								),	*/
/* 								xtb_hort( XTB_JUST, D_HPAD, D_INT,	*/
/* 									xtb_w( aF(SNATA_F)), xtb_w( aF(SNATP_F)), xtb_w( aF(SNATR_F)),	*/
/* 									xtb_w( aF(SNATD_F)), xtb_w( aF(SNATM_F)), xtb_w( aF(SNATH_F)),	*/
/* 									xtb_w( aF(SNATN_F)),	*/
/* 									NE	*/
/* 								),	*/
/* 								NE	*/
/* 							),	*/
							xtb_vert( XTB_CENTER, D_VPAD/2, D_INT,
								xtb_hort( XTB_CENTER, D_HPAD, D_INT,
									xtb_w( aF(SNLWLBL_F)), xtb_w( aF(SNLW_F)), xtb_w( aF(SNLS_F)),
									xtb_w( aF(SNMSLBL_F)), xtb_w( aF(SNSMSz_F)), xtb_w( aF(SNSMS_F)),
									xtb_w( aF(SNPLILBL_F)), xtb_w( aF(SNPLI_F)), xtb_w( aF(SNAdI_F)),
									NE
								),
								xtb_hort( XTB_CENTER, D_HPAD, D_INT,
									xtb_w( aF(SNELWLBL_F)), xtb_w( aF(SNELW_F)), xtb_w( aF(SNELS_F)), xtb_w( aF(SNEBW_F)),
									xtb_w( aF(SNBPLBL_F)), xtb_w( aF(SNBPB_F)), xtb_w( aF(SNBPW_F)), xtb_w( aF(SNBPT_F)),
									NE
								),
								NE
							),
							NE
						),
						xtb_vert( XTB_CENTER, D_VPAD/2, D_INT/2,
							xtb_hort( XTB_CENTER, D_HPAD, D_INT,
								xtb_w( aF(SNDS_F)), xtb_w( aF(SNMS_F)), xtb_w( aF(SNHL_F)),
								xtb_w( aF(SNNL_F)),
								xtb_w( aF(SNBF_F)),
								xtb_hort( XTB_CENTER, 0, 0, xtb_w( aF(SNMF_F)), xtb_w( aF(SNPM_F)), NE),
								xtb_hort( XTB_CENTER, 0, 0, xtb_w( aF(SNVMF_F)), xtb_w( aF(SNVFM_F)), xtb_w( aF(SNVRM_F)), NE),
								xtb_w( aF(SNSE_F)),
								xtb_w( aF(OWM_F)), xtb_w( aF(SNSL_F)), xtb_w( aF(SNSLLS_F)), xtb_w( aF(SNSRAWD_F)),
								xtb_w( aF(SNFLT_F)), xtb_w( aF(SNSPADD_F)),
								NE
							),
							NE
						),
						NE
					),
/* 					xtb_vert( XTB_CENTER, D_VPAD/2, D_INT/2,	*/
/* 						xtb_w( aF(SNPLILBL_F)), xtb_w( aF(SNPLI_F)), xtb_w( aF(SNAdI_F)),	*/
/* 						NE	*/
/* 					),	*/
					xtb_vert( XTB_CENTER, D_VPAD/2, D_INT/2,
						xtb_w( aF(SNSSARROW_F)),
						xtb_w( aF(SNSARRORN_F)),
						xtb_w( aF(SNSEARROW_F)),
						xtb_w( aF(SNEARRORN_F)),
						NE
					),
					NE
				),
				xtb_hort( XTB_CENTER, D_HPAD, D_INT,
/* 					xtb_vert( XTB_CENTER, D_VPAD/2, D_INT/2,	*/
/* 						xtb_hort( XTB_CENTER, D_HPAD, D_INT,	*/
/* 							xtb_w( aF(ADH_F)), xtb_w( aF(RFC_F)), xtb_w( aF(RFRS_F)),	*/
/* 							NE	*/
/* 						),	*/
/* 						xtb_hort( XTB_CENTER, D_HPAD, D_INT,	*/
/* 							xtb_w( aF(SNRDS_F)), xtb_w( aF(SNDAS_F)),	*/
/* 							NE	*/
/* 						),	*/
/* 						NE	*/
/* 					),	*/
					xtb_vert( XTB_CENTER, D_VPAD/2, D_INT/2,
						xtb_hort( XTB_CENTER, D_VPAD/2, D_INT/2,
							xtb_w( aF(LBLX1_F)), xtb_w( aF(LBLY1_F)),
							xtb_w( aF(LBLX2_F)), xtb_w( aF(LBLY2_F)), xtb_w( aF(LSFN_F)),
							xtb_w( aF(LBLSL_F)), xtb_w( aF(VTTYPE_F)), xtb_w( aF(LBLVL_F)),
							NE
						),
						xtb_hort( XTB_CENTER, D_VPAD/2, D_INT/2,
							xtb_w( aF(SNCOLS_F)),
							xtb_w( aF(SNXC_F)), xtb_w( aF(SNYC_F)), xtb_w( aF(SNEC_F)), xtb_w( aF(SNVC_F)),
							xtb_w( aF(SNNC_F)),
							xtb_w( aF(SNFNS_F)), xtb_w( aF(SNFN_F)), xtb_w( aF(SNINF_F)),
							xtb_w( aF(SNSSU_F)), xtb_w( aF(SNSSD_F)),
							NE
						),
						NE
					),
					NE
				),
				NE
			),
			NE
		),
		xtb_hort(XTB_CENTER, D_HPAD, D_INT,
		   xtb_hort( XTB_CENTER, D_HPAD/2, D_INT/2, xtb_w(aF(SNLEGFUN_F)), xtb_w(aF(DPNAS_F)), NE),
			xtb_vert( XTB_LEFT, D_VPAD/2, D_INT, xtb_w(aF(SNpLEG_F)),
				xtb_hort( XTB_CENTER, D_HPAD/2, D_INT/2, xtb_w(aF(SNLEG_F)), xtb_w(aF(SNLEGEDIT_F)), NE), NE),
		   NE
		),
		NE
	);

    frame->width = frame->height = frame->x_loc = frame->y_loc = 0;

    def = xtb_fmt_do(
#ifdef SLIDER
		xtb_hort( XTB_CENTER, D_HPAD, D_INT, xtb_w(aF(SLIDE_F)), xtb_w(aF(SLIDE2_F)),
#endif
		xtb_vert(XTB_CENTER, D_VPAD/2, D_INT/2,
			xtb_hort( XTB_CENTER_J, 0, 0,
				xtb_vert( XTB_CENTER_J, D_HPAD/2, D_INT/2,
					xtb_hort( XTB_CENTER, D_HPAD, D_INT,
						xtb_w( aF(SNLBL_F)), xtb_w(aF(TITLE_F)),
						NE
					),
					tabarea,
					snarea,
					NE
				),
				NE
			),
			xtb_hort(XTB_JUST, D_HPAD, D_INT,
				 xtb_w(aF(OK_F)), xtb_w(aF(REDRAW_F)), xtb_w(aF(AUTOREDRAW_F)), xtb_w(aF(UPDATE_F)),
				 xtb_w(aF(PHIST_F)), xtb_w(aF(HELP_F)),
				 xtb_w(aF(VARIABLES_F)), xtb_w(aF(PARGS_F)),
/* 				 xtb_w(aF(PRINT_F)),	*/
				 xtb_w(aF(QUIT_F)),
				 NE
			),
			NE
		),
#ifdef SLIDER
		NE),
#endif
       &frame->width, &frame->height
	);
    xtb_mv_frames(LAST_F, sd_af);
    xtb_fmt_free(def);

    frame->width += ( D_BRDR);
    frame->height += ( D_BRDR);

      /* Make window large enough to contain the info */
    XResizeWindow(disp, frame->win, frame->width, frame->height);
	JustResized= True;
    hints.flags = USSize|USPosition;	/* PSize;	*/
    hints.width = frame->width;
    hints.height = frame->height;
    XSetNormalHints(disp, frame->win, &hints);
    XGetNormalHints(disp, frame->win, &hints);
	  /* Check if the dialog disappeared from the screen. There may be no easy
	   \ way for the user to get it back, so we just move it to the appropriate
	   \ corner.
	   */
	{ Window dummy;
	  int move= 0, rx, ly;
		XTranslateCoordinates( disp, frame->win, RootWindow(disp, screen),
				  0, 0, &hints.x, &hints.y, &dummy
		);
		rx= hints.x+ hints.width;
		ly= hints.y+ hints.height;
		if( rx> DisplayWidth(disp, screen) ){
			hints.x= DisplayWidth( disp, screen)- hints.width;
			move= 1;
		}
		else if( rx< 0 ){
			hints.x= 0;
			move= 1;
		}
/* 		if( hints.x> DisplayWidth(disp, screen) ){	*/
/* 			hints.x= DisplayWidth(disp, screen)- hints.width;	*/
/* 			move= 1;	*/
/* 		}	*/
		if( ly> DisplayHeight(disp, screen) ){
			hints.y= DisplayHeight(disp, screen)- hints.height;
			move= 1;
		}
		else if( ly< 0 ){
			hints.y= 0;
			move= 1;
		}
		if( move ){
			XMoveWindow( disp, frame->win, hints.x, hints.y );
			XSetNormalHints(disp, frame->win, &hints);
		}
	}
	
	  /* Why would this be???	*/
    frame->width += (2 * D_BRDR);
    frame->height += (2 * D_BRDR);
	active--;
	if( discard ){
		  /* Discard any ConfigureNotify events this action might generate for this window	*/
		for( i= 0; XCheckTypedWindowEvent( disp, frame->win, ConfigureNotify, &evt ); i++ );
		if( i && debugFlag ){
			fprintf( StdErr, "format_SD_Dialog(): %d ConfigureNotify events discarded for window 0x%lx\n",
				i, frame->win
			);
		}
		for( i= 0; XCheckWindowEvent( disp, frame->win, ExposureMask|StructureNotifyMask, &evt ); i++ );
		if( i && debugFlag ){
			fprintf( StdErr, "format_SD_Dialog(): %d ExposureMask|StructureNotifyMask events discarded for window 0x%lx\n",
				i, frame->win
			);
			fflush( StdErr );
		}
	}
	return(1);
}

int Minimal_LegendField= True;

int update_SD_size()
{ int lw= legend_width, ll= legend_len,
	fnl= fileName_len, fnw= fileName_width, format= 0;
  Pixel xbp= xtb_back_pix, xnp= xtb_norm_pix, xlp= xtb_light_pix, xmp= xtb_middle_pix;

	xtb_back_pix= xtb_white_pix;
	xtb_norm_pix= xtb_black_pix;
	xtb_light_pix= xtb_Lgray_pix;
	xtb_middle_pix= xtb_Mgray_pix;

	do_update_SD_size= 0;

	if( SD_Dialog.mapped== 0 || !theWin_Info ){
		return(0);
	}
	if( Minimal_LegendField ){
	  int mlw, st;
		LegendFunctionNr= st= 0;
		find_fileName_max_AND_legend_len();
		mlw= legend_width;
		for( LegendFunctionNr= 1; LegendFunctionNr< LEGENDFUNCTIONS; LegendFunctionNr++ ){
			find_fileName_max_AND_legend_len();
			if( legend_width< mlw && strlen( LegendorTitle(0, LegendFunctionNr)) ){
				st= LegendFunctionNr;
				mlw= legend_width;
			}
		}
		switch( (LegendFunctionNr= st) ){
			case 0:
				pLegFunNr= 1;
				break;
			default:
				pLegFunNr= 0;
				break;
		}
		legend_width= mlw;
		Minimal_LegendField= False;
	}
	else{
		find_fileName_max_AND_legend_len();
	}
	if( legend_width!= lw  || legend_len!= ll || SD_LMaxBufSize!= LMAXBUFSIZE ){
	   xtb_data info;
	   char *desc= AF(SNLEG_F).description, *desc2= AF(SNNRSL_F).description; 
	   int ts= AF(SNLEG_F).tabset, ts2= AF(SNNRSL_F).tabset, ts3= AF(SNpLEG_F).tabset;
		xfree( AF(SNpLEG_F).description );
		xtb_to_del(AF(SNpLEG_F).win );
		xfree( AF(SNpLEG_F).framelist );
		xtb_ti_del(AF(SNLEG_F).win, &info );
		xfree( AF(SNLEG_F).framelist );
		SD_LMaxBufSize= LMAXBUFSIZE;
		  /* 990716: we MUST reallocate this one, after changing SD_LMaxBufSize..!	*/
		data_legend_buf= realloc( data_legend_buf, (1+LMAXBUFSIZE)* sizeof(char));
		dlb_len= LMAXBUFSIZE;
		xtb_ti_new(SD_Dialog.win, setName_max, legend_len, SD_LMaxBufSize, SD_snl_fun, &new_legend, aF(SNLEG_F));
		AF(SNLEG_F).tabset= ts;
		xtb_to_new2(SD_Dialog.win, setName_max, AF(SNLEG_F).width/XFontWidth( dialogFont.font),
			XTB_TOP_LEFT, &dialogFont.font, NULL, aF(SNpLEG_F)
		);
		AF(SNpLEG_F).tabset= ts3;
		xtb_to_set(AF(SNpLEG_F).win, LegendorTitle(data_sn_number, pLegFunNr) );
		xtb_ti_set(AF(SNLEG_F).win, data_legend_buf, NULL);
		xtb_describe( aF(SNpLEG_F), LegendFTypes[pLegFunNr] );
		SD_info->sn_legend = AF(SNLEG_F).win;
		AF(SNLEG_F).description= desc;
		xtb_sr_del( AF(SNNRSL_F).win, &info );
		xfree( AF(SNNRSL_F).framelist );
		xtb_sri_new( SD_Dialog.win, 0, setNumber+ theWin_Info->ulabels- 1, data_sn_number,
/* 			- legend_len/2, 0, snn_slide_f, NULL, aF(SNNRSL_F)	*/
			AF(SNLEG_F).width/2, 0, snn_slide_f, NULL, aF(SNNRSL_F)
		);
		AF(SNNRSL_F).tabset= ts2;
		AF(SNNRSL_F).description= desc2;
		format= 1;
	}
	if( fileName_width!= fnw  || fileName_len!= fnl ){
	   xtb_data info;
	   char *desc= AF(SNFN_F).description;
	   int ts= AF(SNFN_F).tabset;
		xtb_ti_del(AF(SNFN_F).win, &info );
		xfree( AF(SNFN_F).framelist );
		xtb_ti_new(SD_Dialog.win, fileName_max, fileName_len, MAXCHBUF, SD_snl_fun, &new_set, aF(SNFN_F));
		AF(SNFN_F).tabset= ts;
		xtb_ti_set(AF(SNFN_F).win, fileName_buf, NULL);
		AF(SNFN_F).description= desc;
		format= 1;
	}
	if( format ){
		xtb_select_frames_tabset( LAST_F, sd_af, SD_tabset, tabselect );
		format_SD_Dialog( &SD_Dialog, 0 );
	}
	set_changeables(0,True);

	xtb_back_pix= xbp;
	xtb_norm_pix= xnp;
	xtb_light_pix= xlp;
	xtb_middle_pix= xmp;
	return(format);
}

#ifdef SLIDER
/*  a sample slideruler	*/
xtb_hret slide_f( Window win, int pos, double val, xtb_data info)
{
	if( !theWin_Info ){
		return( XTB_STOP );
	}

	if( win== AF(SLIDE_F).win ){
		xtb_sr_set( AF(SLIDE2_F).win, val, NULL);
	}
	else{
		xtb_sr_set( AF(SLIDE_F).win, val, NULL);
	}
	return( XTB_HANDLED);
}
#endif

void get_data_legend_buf()
{ Sinc sinc;
	Sinc_string_behaviour( &sinc, NULL, 0,0, SString_Dynamic );
	if( LegendFunctionNr== 6 || LegendFunctionNr== 7 ){
		strcpalloc( &data_legend_buf, &dlb_len, LegendorTitle(data_sn_number, LegendFunctionNr) );
	}
	else switch( data_sn_number ){
		case -22:
			Sprint_LabelsList( &sinc, theWin_Info->ColumnLabels, NULL );
			strcpalloc( &data_legend_buf, &dlb_len, (sinc.sinc.string)? sinc.sinc.string : "" );
			break;
		case -19:
			Sprint_string( &sinc, "", "\n", "", theWin_Info->process.enter_raw_after );
			strcpalloc( &data_legend_buf, &dlb_len, (sinc.sinc.string)? sinc.sinc.string : theWin_Info->process.enter_raw_after );
			break;
		case -18:
			Sprint_string( &sinc, "", "\n", "", theWin_Info->process.leave_raw_after );
			strcpalloc( &data_legend_buf, &dlb_len, (sinc.sinc.string)? sinc.sinc.string : theWin_Info->process.leave_raw_after );
			break;
		case -17:
			Sprint_string( &sinc, "", "\n", "", theWin_Info->process.dump_before );
			strcpalloc( &data_legend_buf, &dlb_len, (sinc.sinc.string)? sinc.sinc.string : theWin_Info->process.dump_before );
			break;
		case -16:
			Sprint_string( &sinc, "", "\n", "", theWin_Info->process.dump_after );
			strcpalloc( &data_legend_buf, &dlb_len, (sinc.sinc.string)? sinc.sinc.string : theWin_Info->process.dump_after );
			break;
		case -15:
			Sprint_string( &sinc, "", "\n", "", theWin_Info->process.draw_before );
			strcpalloc( &data_legend_buf, &dlb_len, (sinc.sinc.string)? sinc.sinc.string : theWin_Info->process.draw_before );
			break;
		case -14:
			Sprint_string( &sinc, "", "\n", "", theWin_Info->process.draw_after );
			strcpalloc( &data_legend_buf, &dlb_len, (sinc.sinc.string)? sinc.sinc.string : theWin_Info->process.draw_after );
			break;
		case -11:
			Sprint_string( &sinc, "", "\n", "", theWin_Info->transform.x_process );
			strcpalloc( &data_legend_buf, &dlb_len, (sinc.sinc.string)? sinc.sinc.string : theWin_Info->transform.x_process );
			break;
		case -10:
			Sprint_string( &sinc, "", "\n", "", theWin_Info->transform.y_process );
			strcpalloc( &data_legend_buf, &dlb_len, (sinc.sinc.string)? sinc.sinc.string : theWin_Info->transform.y_process );
			break;
		case -9:
			Sprint_string( &sinc, "", "\n", "", theWin_Info->process.data_init );
			strcpalloc( &data_legend_buf, &dlb_len, (sinc.sinc.string)? sinc.sinc.string : theWin_Info->process.data_init );
			break;
		case -8:
			Sprint_string( &sinc, "", "\n", "", theWin_Info->process.data_before );
			strcpalloc( &data_legend_buf, &dlb_len, (sinc.sinc.string)? sinc.sinc.string : theWin_Info->process.data_before );
			break;
		case -7:
			Sprint_string( &sinc, "", "\n", "", theWin_Info->process.data_process );
			strcpalloc( &data_legend_buf, &dlb_len, (sinc.sinc.string)? sinc.sinc.string : theWin_Info->process.data_process );
			break;
		case -6:
			Sprint_string( &sinc, "", "\n", "", theWin_Info->process.data_after );
			strcpalloc( &data_legend_buf, &dlb_len, (sinc.sinc.string)? sinc.sinc.string : theWin_Info->process.data_after );
			break;
		case -5:
			Sprint_string( &sinc, "", "\n", "", theWin_Info->process.data_finish );
			strcpalloc( &data_legend_buf, &dlb_len, (sinc.sinc.string)? sinc.sinc.string : theWin_Info->process.data_finish );
			break;
		case -4:
			if( titleText2 ){
				strncpy( data_legend_buf, titleText2, dlb_len-1);
			}
			else{
				data_legend_buf[0]= '\0';
			}
			break;
		case -3:
			if( titleText ){
			  extern int titleTextSet;
				strncpy( data_legend_buf, titleText, dlb_len-1);
				titleTextSet= 1;
			}
			else{
				data_legend_buf[0]= '\0';
			}
			break;
		case -2:
			strncpy( data_legend_buf, raw_XLABEL(theWin_Info), dlb_len-1);
			break;
		case -1:
			strncpy( data_legend_buf, raw_YLABEL(theWin_Info), dlb_len-1);
			break;
		case -21:
		case -20:
		case -13:
		case -12:
		default:
			if( data_sn_number>= 0 && data_sn_number< setNumber ){
				strcpalloc( &data_legend_buf, &dlb_len, LegendorTitle(data_sn_number, LegendFunctionNr) );
			}
			else{
			  char *c= GetLabelNr( data_sn_number- setNumber);
				if( c ){
					strncpy( data_legend_buf, c, SD_LMaxBufSize);
				}
				else{
					xtb_ti_get( SD_info->sn_legend, data_legend_buf, NULL);
				}
			}
			break;
	}
	xfree( sinc.sinc.string );
}

  /* Warning: one such static variable per module that can perform "smart" xtb_init calling	*/
static int xtb_UseColours= False;

extern Atom wm_delete_window;

 /* callback function for xtb_br2D_new() that will optimally lay out the buttons in the radio-button
  \ ("button row") frame.
  */
xtb_fmt *format_legfun( xtb_frame *br_frame, int cnt, xtb_frame **buttons, xtb_data val )
{ xtb_fmt *format= NULL;
	if( cnt== LEGENDFUNCTIONS && val== SD_info ){
	  /* The LegendFunction selector: */
		format= xtb_fmt_do( 
			xtb_vert( XTB_CENTER, BR_YPAD, BR_INTER/2,
				xtb_hort( XTB_CENTER_J, 0, BR_INTER,
/* 					xtb_w(buttons[0]), xtb_w( buttons[1]), xtb_w(buttons[7]), xtb_w( buttons[3]),	*/
					xtb_w(buttons[0]), xtb_w( buttons[7]), xtb_w(buttons[3]), xtb_w( buttons[4]),
					NE
				),
				xtb_hort( XTB_JUST, 0, BR_INTER,
/* 					xtb_w(buttons[2]), xtb_w( buttons[4]), xtb_w( buttons[5]), xtb_w( buttons[6]), 	*/
					xtb_w(buttons[1]), xtb_w( buttons[2]), xtb_w( buttons[6]), xtb_w( buttons[5]), 
					NE
				),
				NE
			),
			&br_frame->width, &br_frame->height
		);
	}
	return( format );
}

static void make_SD_dialog(win, spawned, prog, cookie, OKbtn, REDRAWbtn, frame, title, in_title)
Window win;			/* Parent window          */
Window spawned;			/* Spawned from window    */
char *prog;			/* Program name           */
xtb_data cookie;		/* Info for do_hardcopy   */
xtb_frame *OKbtn;		/* Frame for OK button    */
xtb_frame *REDRAWbtn;		/* Frame for REDRAW button    */
xtb_frame *frame;		/* Returned window/size   */
char *title, *in_title;
/*
 * This routine constructs a new dialog for asking the user about
 * hardcopy devices.  The dialog and its size is returned in 
 * `frame'.  The window of the `ok' button is returned in `btnwin'.  
 * This can be used to reset some of the button state to reuse the dialog.
 */
{
    Window SD_Dialog_win;
    Cursor diag_cursor;
    XColor fg_color, bg_color;
    XSizeHints hints;
    XWMHints *wmhints, WMH;
    unsigned long wamask;
    XSetWindowAttributes wattr;
    int i, found;
    static char *errb_types[ERROR_TYPES] = { "No", "Bar", "Trng", "Box", "Regn", "Vect", "Intz", "Msze" };
	extern int bdrSize;			/* Width of border         */
	static xtb_frame *local_sd_af= sd_af;
	static char *tabs[5]= { "Show all controls", "Global controls", "Set controls", "None", "Print dialog" };
	char pc[]= "\\#x7f\\\\";
	char *bt= parse_codes(pc);

	{ XColor zero, grid, norm, bg;
	  extern Pixel gridPixel;
	  extern int ButtonContrast;
		zero.pixel= zeroPixel;
		XQueryColor( disp, cmap, &zero );
		grid.pixel= gridPixel;
		XQueryColor( disp, cmap, &grid );
		norm.pixel= normPixel;
		XQueryColor( disp, cmap, &norm );
		bg.pixel= bgPixel;
		XQueryColor( disp, cmap, &bg );
		  /* If there's enough luminance-contrast between those 2 colours, use 'm for
		   \ the dialog.
		   */
		if( xtb_UseColours && fabs( xtb_PsychoMetric_Gray(&zero) - xtb_PsychoMetric_Gray(&grid) )>= ButtonContrast ){
			xtb_init(disp, screen, zeroPixel, gridPixel, dialogFont.font, dialog_greekFont.font, True );
		}
		else if( xtb_UseColours && fabs( xtb_PsychoMetric_Gray(&norm) - xtb_PsychoMetric_Gray(&bg) )>= ButtonContrast ){
			xtb_init(disp, screen, normPixel, bgPixel, dialogFont.font, dialog_greekFont.font, True );
		}
		else{
			xtb_init(disp, screen, black_pixel, white_pixel, dialogFont.font, dialog_greekFont.font, True );
		}
	}

	SD_Dialog.mapped= -1;

    wamask = ux11_fill_wattr(&wattr, CWBackPixel, bgPixel,
			     CWBorderPixel, bdrPixel,
/* 			     CWOverrideRedirect, True,	*/
				CWBackingStore, Always,
			     CWSaveUnder, True,
			     CWColormap, cmap, UX11_END);
    XGetNormalHints(disp, spawned, &hints);
    SD_Dialog_win = XCreateWindow(disp, win, hints.x, hints.y, hints.width, hints.height, bdrSize,
			    depth, InputOutput, vis,
			    wamask, &wattr);
	XSetWMProtocols( disp, SD_Dialog_win, &wm_delete_window, 1 );
	if( !(wmhints= XAllocWMHints()) ){
		wmhints= &WMH;
	}
	wmhints->flags = InputHint | StateHint | XUrgencyHint;
	wmhints->input = True;
/* 	wmhints->initial_state = IconicState;	*/
	XSetWMHints(disp, SD_Dialog_win, wmhints);

    frame->win = SD_Dialog_win;
	frame->mapped= 0;
	frame->description= strdup( "This is the Settings Dialog's main window" );
    if( !title){
	    XStoreName(disp, SD_Dialog_win, "Settings Dialog");
	}
	else{
	    XStoreName(disp, SD_Dialog_win, title);
	}
/*     XSetTransientForHint(disp, spawned, SD_Dialog_win);	*/
    SD_info = (struct d_info *) calloc( 1, sizeof(struct d_info));
    SD_info->prog = prog;
    SD_info->cookie = cookie;
	xtb_register( frame, frame->win, NULL, NULL );

    for (i = 0;  i < ((int) LAST_F);  i++) {
		AF(i).tabset= -1;
    }

    /* Make all frames */
    xtb_to_new(SD_Dialog_win, in_title, XTB_TOP_LEFT, &dialogFont.font, &dialog_greekFont.font, aF(TITLE_F));
    xtb_to_new(SD_Dialog_win, "EDIT:  ", XTB_TOP_LEFT, &dialogFont.font, &dialog_greekFont.font, aF(SNLBL_F));

	if( Minimal_LegendField ){
	  int mlw, st;
		LegendFunctionNr= st= 0;
		find_fileName_max_AND_legend_len();
		mlw= legend_width;
		for( LegendFunctionNr= 1; LegendFunctionNr< LEGENDFUNCTIONS; LegendFunctionNr++ ){
			find_fileName_max_AND_legend_len();
			if( legend_width< mlw && strlen( LegendorTitle(0, LegendFunctionNr)) ){
				st= LegendFunctionNr;
				mlw= legend_width;
			}
		}
		switch( (LegendFunctionNr= st) ){
			case 0:
				pLegFunNr= 1;
				break;
			default:
				pLegFunNr= 0;
				break;
		}
		legend_width= mlw;
	}
	else{
		find_fileName_max_AND_legend_len();
	}

    xtb_br_new(SD_Dialog_win, 5, tabs, 0,
	       tabs_fun, (xtb_data) NULL, aF(TABSELECT_F)
	);
	xtb_describe( aF(TABSELECT_F), "A group of buttons allow to select a control widget subset.\n");
	xtb_describe( AF(TABSELECT_F).framelist[0], "Show all control widgets\n");
	xtb_describe( AF(TABSELECT_F).framelist[1], "Show only widgets controlling global settings\n");
	xtb_describe( AF(TABSELECT_F).framelist[2], "Show only widgets controlling individual set settings\n");
	xtb_describe( AF(TABSELECT_F).framelist[3], "Show only the striclty necessary widgets\n");
	xtb_describe( AF(TABSELECT_F).framelist[4], "Close this window, and open the Hardcopy Dialog\n");

	xtb_bt_new(SD_Dialog_win, "no.Wdbg", SD_sdds_sl_fun, (xtb_data) &WdebugFlag, aF(WDEBUG_F) );
	xtb_describe_s( aF(WDEBUG_F), "A window specific debug/debug-off flag\n", 1);
	xtb_bt_new( SD_Dialog_win, "dbg", SD_sdds_sl_fun, &debugFlag, aF(DEBUG_F) );
	AF(DEBUG_F).tabset= 1;
	xtb_ti_new( SD_Dialog_win, "", 2, MAXCHBUF, SD_dfn_fun, &debugLevel, aF(DEBUGL_F) );
	xtb_describe_s( aF(DEBUGL_F), "Enter debugLevel here\n", 1);
/* 	{ char buf[64];	*/
/* 		sprintf( buf, "%d", debugLevel );	*/
/* 		xtb_ti_set( AF(DEBUGL_F).win, buf, 0);	*/
/* 	}	*/
	xtb_bt_new( SD_Dialog_win, "sync", SD_sdds_sl_fun, &Synchro_State, aF(SYNC_F) );
	xtb_describe_s( aF(SYNC_F), "Controls X11 synchronised mode (global setting)\n", 1 );
	xtb_bt_new( SD_Dialog_win, "Dsync", SD_sdds_sl_fun, &dataSynchro_State, aF(DSYNC_F) );
	xtb_describe_s( aF(DSYNC_F), "X11 synchronised mode only for the drawing of data (per-window setting)\n", 1 );
	xtb_bt_new( SD_Dialog_win, "Frac", SD_sdds_sl_fun, &Allow_Fractions, aF(FRACP_F) );
	xtb_describe_s( aF(FRACP_F), "If set, real numbers are printed as a fraction\n"
		"instead of in decimal notation if\n"
		"that notation does not take more place\n",
		1
	);
	xtb_ti_new( SD_Dialog_win, d3str_format, 4, sizeof(d3str_format)/sizeof(char), SD_df_fun, (xtb_data) d3str_format, aF(D3STRF_F) );
	xtb_describe_s( aF(D3STRF_F),
		"printf(3) format string for printing (real) numbers\nAlso used in XGraph and SpreadSheet dumps", 1);
	xtb_bt_new( SD_Dialog_win, "Dslnt", SD_sdds_sl_fun, &data_silent_process, aF(DSILENT_F) );
	xtb_describe_s( aF(DSILENT_F), "Ignore non-user-generated events during the processing of data (per-window setting)\n", 1 );

    xtb_to_new(SD_Dialog_win, "Error type:", XTB_TOP_LEFT, &dialogFont.font, NULL, aF(SNERRB_F));
	AF(SNERRB_F).tabset= 2;
	found= error_type2num[ SD_get_errb()];
    xtb_br_new(SD_Dialog_win, ERROR_TYPES, errb_types, found,
	       errb_fun, (xtb_data) SD_info, aF(OERRBROW_F)
	);
	SD_info->errb= AF(OERRBROW_F).win;
	xtb_describe_s( aF(OERRBROW_F), "Set-specific setting of error-representation.\n"
		"Co-operate with the [All], [Prev], etc. buttons below\n",
		2
	);
	xtb_describe( AF(OERRBROW_F).framelist[0], "Don't draw error bars\n");
	xtb_describe( AF(OERRBROW_F).framelist[1], "Draw error bars (endstopped lines)\n");
	xtb_describe( AF(OERRBROW_F).framelist[2], "Draw error \"triangular error bars\"\n");
	xtb_describe( AF(OERRBROW_F).framelist[3], "Draw error error rectangles\n");
	xtb_describe( AF(OERRBROW_F).framelist[4], "Connect all lower resp. higher Y values with lines,\n"
		"creating an error region.\n"
	);
	xtb_describe( AF(OERRBROW_F).framelist[5],
		"Interprete the \"error\" data as the orientation\n"
		"of a vector drawn with one end in the corresponding datapoint\n"
		"Uses the same gonio-base (radix,radix_offset) as polar-plotting\n"
		"Hold down the Mod1 key to prevent automatic rescaling\n"
	);
	xtb_describe( AF(OERRBROW_F).framelist[6], "Error is used to index the *INTENSITY_COLOUR* colourtable\n"
		" The colour found is used to draw the set's marker in\n"
	);
	xtb_describe( AF(OERRBROW_F).framelist[7], "Error is used to determine the markersize\n" );

    xtb_to_new(SD_Dialog_win, "\001TAB to accept/proceed", XTB_TOP_LEFT, &dialogFont.font, NULL, aF(KEY1_F));
	xtb_describe_s( aF(KEY1_F),
		"TAB key accepts entered value and proceeds to next field\nSPACE accepts numericals\n"
		" Cursor Down/Up accept (and decrease/increase numericals by 1)\n",
		1
	);
    xtb_to_new(SD_Dialog_win, "\001ESC to cancel/backup", XTB_TOP_LEFT, &dialogFont.font, NULL, aF(KEY2_F));
	xtb_describe_s( aF(KEY2_F), "ESC backs up to previous field without\nchanging the value of the current field\n", 1);
	{ char txt[16]= "Selection:";
/* 		sprintf( txt, "%c", (char) XC_exchange );	*/
/* 		xtb_to_new(SD_Dialog_win, txt, XTB_CENTERED, cursorFont.font, cursorFont.font, aF(SNNRLBL_F));	*/
		xtb_to_new(SD_Dialog_win, txt, XTB_TOP_LEFT, &dialogFont.font, &dialog_greekFont.font, aF(SNNRLBL_F));
		xtb_describe_s( aF(SNNRLBL_F), "A number of fields that apply to individual sets\n"
			"or to options like names, labels and processing\n"
			"Ctrl-Click on the window next to the slider for\n"
			"an overview of available options",
			2
		);
	}
	xtb_sri_new( SD_Dialog_win, 0, setNumber+ theWin_Info->ulabels- 1, data_sn_number,
		- legend_len/2, 0, snn_slide_f, NULL, aF(SNNRSL_F)
	);
	xtb_describe( aF(SNNRSL_F), "slider to select a set or User Label (no options)\n"
		"Ctrl-Click on the window to the right for more info\n"
	);
    xtb_ti_new(SD_Dialog_win, "", D_SN, MAXCHBUF, snn_fun, (xtb_data) 0, aF(SNNR_F));
	xtb_describe( aF(SNNR_F), "The number of the set to which\nthe set-specific fields apply\nor an option:\n"
		" -T for the alternative titletext\n"
		" X or Y for axis-labels\n"
		" DB for *DATA_BEFORE*\n"
		" DP for *DATA_PROCESS*\n"
		" DA for *DATA_AFTER*\n"
		" TX or TY for *TRANSFORM_X or -Y*\n"
		" EN for *EVAL* (was: *PARAM_NOW*; PN) \n"
		" RF for \"Read a New File\" (*READ_FILE*)\n"
		"    Use a '|' as first character to read from a pipe\n"
		"     (e.g. |echo \"*ARGUMENTS* -T bla\"\n"
		"          to define a new \"supertitle\")\n"
		" BD for *DRAW_BEFORE*\n"
		" AD for *DRAW_AFTER* (DB & DA were used...)\n"
		" ARG for *ARGUMENT* (easier than as shown under RF..)\n"
		" TXT for parsing text (e.g. \"*ARGUMENTS* -T bla\")\n"
		" User Labels are numbered beyond the last set\n"
		" (i.e ULabel #1 is <LastSet#>+1\n"
	);
	{ char buf[64];
		sprintf( buf, "%d", data_sn_number);
	}
    SD_info->sn_number = AF(SNNR_F).win;
	{ char pc[]= "\\#xbc\\";
		xtb_bt_new( SD_Dialog_win, parse_codes(pc), SD_selectfun, (xtb_data) NULL, aF(SNNRM_F) );
	}

	xtb_bt_new( SD_Dialog_win, "DWidth", SD_dynfunc, (xtb_data) &SD_dynamic_resize, aF(SDDR_F) );
	xtb_describe_s( aF(SDDR_F),
		"This activates the dynamic resize option. With this option, the Settings dialog will\n"
		" always be sized according to the widest entry (legend, title, ...) currently selected\n"
		" for editing via the large text entry field below. Note that it is always possible to\n"
		" call up an editor, or the field's full text by Control-Clicking on it.\n",
		0
	);

	xtb_bt_new( SD_Dialog_win, "AllSets", SD_sdds_sl_fun, (xtb_data) &apply_to_all, aF(SNATA_F) );
	xtb_describe_s( aF(SNATA_F), "Apply the next settings-change to all sets\ninstead of just to the selected set\n", 2);
	xtb_bt_new( SD_Dialog_win, "Prev..", SD_sdds_sl_fun, (xtb_data) &apply_to_prev, aF(SNATP_F) );
	xtb_describe_s( aF(SNATP_F),
		"Apply the next settings-change up to and including this set\ninstead of just to the selected set\n", 2);
	xtb_bt_new( SD_Dialog_win, "Rest..", SD_sdds_sl_fun, (xtb_data) &apply_to_rest, aF(SNATR_F) );
	xtb_describe_s( aF(SNATR_F),
		"Apply the next settings-change to this set and the rest\ninstead of just to the selected set\n", 2);
	xtb_bt_new( SD_Dialog_win, "Drwn..", SD_sdds_sl_fun, (xtb_data) &apply_to_drawn, aF(SNATD_F) );
	xtb_describe_s( aF(SNATD_F), "Apply the next settings-change to drawn sets\n"
		"instead of just to the selected set\n"
		"Combines with [AllSets] [Prev..] [Rest..] [Mark..] [HiLt..] [New..] [Source]\n",
		2
	);
	xtb_bt_new( SD_Dialog_win, "Mark..", SD_sdds_sl_fun, (xtb_data) &apply_to_marked, aF(SNATM_F) );
	xtb_describe_s( aF(SNATM_F), "Apply the next settings-change to marked sets\n"
		"instead of just to the selected set\n"
		"Combines with [AllSets] [Prev..] [Rest..] [Drwn..] [HiLt..] [New..] [Source]\n",
		2
	);
	xtb_bt_new( SD_Dialog_win, "HiLt..", SD_sdds_sl_fun, (xtb_data) &apply_to_hlt, aF(SNATH_F) );
	xtb_describe_s( aF(SNATH_F), "Apply the next settings-change to highlighted sets\n"
		"instead of just to the selected set\n"
		"Combines with [AllSets] [Prev..] [Rest..] [Drwn..] [Mark..] [New..] [Source]\n",
		2
	);
	xtb_bt_new( SD_Dialog_win, "New..", SD_sdds_sl_fun, (xtb_data) &apply_to_new, aF(SNATN_F) );
	xtb_describe_s( aF(SNATN_F), "Apply the next settings-change to sets with recently added points\n"
		"instead of just to the selected set\n"
		"Combines with [AllSets] [Prev..] [Rest..] [Drwn..] [Mark..] [HiLt..] [Source]\n",
		2
	);
	xtb_bt_new( SD_Dialog_win, "Source", SD_sdds_sl_fun, (xtb_data) &apply_to_src, aF(SNATS_F) );
	xtb_describe_s( aF(SNATS_F), "Apply the next settings-change to source sets (not linked sets)\n"
		"instead of just to the selected set\n"
		"Combines with [AllSets] [Prev..] [Rest..] [Mark..] [HiLt..] [New..] [Source]\n",
		2
	);

    xtb_ti_new(SD_Dialog_win, SD_ascanf_separator_buf, 2, 2, SD_df_fun, (xtb_data) SD_ascanf_separator_buf, aF(DPNAS_F));
	xtb_describe( aF(DPNAS_F),
		"The character separating values, variables or expressions in\n"
		" *EVAL* expressions evaluated via the Settings Dialog\n"
		" Defaults to a ',' as used in most other cases\n"
	);

	xtb_to_new( SD_Dialog_win, "l.w & st", XTB_TOP_LEFT, &dialogFont.font, NULL, aF(SNLWLBL_F) );
	AF(SNLWLBL_F).tabset= 2;
	xtb_ti_new( SD_Dialog_win, "", D_SN, MAXCHBUF, SD_dfn_fun, (xtb_data) &data_sn_lineWidth, aF(SNLW_F) );
	xtb_describe_s( aF(SNLW_F), "linewidth of set or ULabel", 2 );
	SD_info->sn_lineWidth= AF(SNLW_F).win;
	xtb_ti_new( SD_Dialog_win, "", D_SN, MAXCHBUF, SD_dfn_fun, (xtb_data) &data_sn_linestyle, aF(SNLS_F) );
	xtb_describe_s( aF(SNLS_F), "linestyle", 2 );
	SD_info->sn_linestyle= AF(SNLS_F).win;

	xtb_to_new( SD_Dialog_win, "m.w & st", XTB_TOP_LEFT, &dialogFont.font, NULL, aF(SNMSLBL_F) );
	AF(SNMSLBL_F).tabset= 2;
	xtb_ti_new( SD_Dialog_win, "", D_SN, MAXCHBUF, SD_dfn_fun, (xtb_data) &data_sn_markSize, aF(SNSMSz_F) );
	xtb_describe_s( aF(SNSMSz_F), MARKERSIZEDESCR, 2);
	xtb_ti2_new( SD_Dialog_win, "", D_SN, MAXCHBUF, SD_dfn_fun,
		bt, SD_markerStyle_selectfun, (xtb_data) &data_sn_markstyle, aF(SNSMS_F) );
	xtb_describe_s( aF(SNSMS_F), "Marker number, or markstyle\n Only positive values", 2 );

	xtb_to_new( SD_Dialog_win, "err l.w,st; w", XTB_TOP_LEFT, &dialogFont.font, NULL, aF(SNELWLBL_F) );
	AF(SNELWLBL_F).tabset= 2;
	xtb_ti_new( SD_Dialog_win, "", D_SN, MAXCHBUF, SD_dfn_fun, (xtb_data) &data_sn_elineWidth, aF(SNELW_F) );
	xtb_describe_s( aF(SNELW_F), "the width of errorbars/triangles.\nValue < 0 means equal to l.width\n", 2);
	xtb_ti_new( SD_Dialog_win, "", D_SN, MAXCHBUF, SD_dfn_fun, (xtb_data) &data_sn_elinestyle, aF(SNELS_F) );
	xtb_describe_s( aF(SNELS_F), "the style of errorbars/triangles.\nValue < 0 means equal to l.style\n", 2);
	xtb_ti_new( SD_Dialog_win, "", D_SN, MAXCHBUF, SD_dfn_fun, (xtb_data) &ebarWidth_set, aF(SNEBW_F) );
	xtb_describe_s( aF(SNEBW_F),
		"the width of errorbars/triangles.\nValue < 0 means scale to plotwidth\n NaN means use default\n", 2);

	xtb_to_new( SD_Dialog_win, "b.pars", XTB_TOP_LEFT, &dialogFont.font, NULL, aF(SNBPLBL_F) );
	xtb_describe_s( aF(SNBPLBL_F), "barplot parameters (*BARBASE*) barBase, barWidth, barType\n Depend on the [Bars] setting!", 2);
	xtb_ti_new( SD_Dialog_win, "", D_SN, MAXCHBUF, SD_dfn_fun, (xtb_data) &barBase_set, aF(SNBPB_F) );
	xtb_describe_s( aF(SNBPB_F), "the base for barplotting this set\n NaN means non-specified.", 2);
	xtb_ti_new( SD_Dialog_win, "", D_SN, MAXCHBUF, SD_dfn_fun, (xtb_data) &barWidth_set, aF(SNBPW_F) );
	xtb_describe_s( aF(SNBPW_F),
		"the barWidth for barplotting this set\n Width <= 0 means scale to plotwidth\n NaN means non-specified", 2);
	xtb_ti_new( SD_Dialog_win, "", D_SN, MAXCHBUF, SD_dfn_fun, (xtb_data) &barType, aF(SNBPT_F) );
	xtb_describe_s( aF(SNBPT_F), "this set's type of barplotting:\n"
		" 0:   a bar filled with the selected linestyle & colour, with a solid outline of same colour and selected linewidth\n"
		" 1:   a transparent bar defined by an outline in selected linewidth, linecolour and linestyle\n"
		" 2:   a bar filled in appropriate highlightcolour and outlined in selected linewidth, linecolour and linestyle\n"
		" 3:   as 0, filled in appropriate highlightcolour and outlined in selected linecolour\n"
		" 4:   as 2, filled in the appropriate intensitycolour\n"
		" 5,6: \"bars\" as proposed by E. Tufte: only the left (5) or right (6) stem, and half of the high line are drawn\n"
		"      in the selected linewidth, colour and style\n",
		2
	);

	{ char pc[]= "int#x00\\#xaf\\";
		xtb_to_new( SD_Dialog_win, parse_codes(pc), XTB_TOP_LEFT, &dialogFont.font, NULL, aF(SNPLILBL_F) );
	}
	AF(SNPLILBL_F).tabset= 2;
	xtb_ti_new( SD_Dialog_win, "", D_SN, MAXCHBUF, SD_dfn_fun, (xtb_data) &data_sn_plot_interval, aF(SNPLI_F) );
	xtb_describe_s( aF(SNPLI_F), "If >0, the interval at which points are drawn\n", 2);
	xtb_ti_new( SD_Dialog_win, "", D_SN, MAXCHBUF, SD_dfn_fun, (xtb_data) &data_sn_adorn_interval, aF(SNAdI_F) );
	xtb_describe_s( aF(SNAdI_F), "If >0, the interval at which points are adorned with markers, bars, etc.\n", 2);

	xtb_bt_new( SD_Dialog_win, "Draw", SD_sdds_sl_fun, (xtb_data) &set_draw_set, aF(SNDS_F) );
	xtb_describe_s( aF(SNDS_F), "Whether or not this set will be drawn\n", 2);
	SD_info->sn_draw_set= AF(SNDS_F).win;
	xtb_bt_new( SD_Dialog_win, "Mark", SD_sdds_sl_fun, (xtb_data) &set_mark_set, aF(SNMS_F) );
	xtb_describe_s( aF(SNMS_F), "Whether or not this set is marked\n", 2);
	xtb_bt_new( SD_Dialog_win, "HiLt", SD_sdds_sl_fun, (xtb_data) &set_highlight, aF(SNHL_F) );
	xtb_describe_s( aF(SNHL_F), "Whether or not this set is highlighted\n", 2);
	xtb_bt_new( SD_Dialog_win, "noLine", SD_sdds_sl_fun, (xtb_data) &set_noLines, aF(SNNL_F) );
	xtb_describe_s( aF(SNNL_F), "Connect datapoints in this set with lines\n", 2);
	xtb_bt_new( SD_Dialog_win, "Bars", SD_sdds_sl_fun, (xtb_data) &set_barFlag, aF(SNBF_F) );
	xtb_describe_s( aF(SNBF_F),
		"This set is a bar graph\n Parameters controlled with the \"b.pars\" fields\n"
		" <Mod1>-click this button to configure how the bars\n"
		" should be shown in the legend.\n", 2
	);
	xtb_bt_new( SD_Dialog_win, "Marks", SD_sdds_sl_fun, (xtb_data) &set_markFlag, aF(SNMF_F) );
	xtb_describe_s( aF(SNMF_F),
		"Mark datapoints of this set with symbols,\n"
		" blobs (big dots), or pixels (small dots)\n"
		" depending on state of the button to the right\n",
		2
	);
	{ char *pixels[]= { "Blbs", "Dts", "Smbl" };
		xtb_bt_new( SD_Dialog_win, "smbl", SD_sdds_sl_fun, (xtb_data) &set_pixelMarks, aF(SNPM_F) );
		xtb_bt_set_text( AF(SNPM_F).win, pixelMarks_Value(), pixels[pixelMarks_Value()], (xtb_data) 0 );
		xtb_describe_s( aF(SNPM_F), "Determines marker type for this set\n", 2);
	}
	xtb_bt_new( SD_Dialog_win, "Value", SD_sdds_sl_fun, (xtb_data) VMARK_ON, aF(SNVMF_F) );
	xtb_describe_s( aF(SNVMF_F),
		"Show the value of the datapoint\n"
		" (x,y) (and maybe error) for line/scatter plots\n"
		" y (and maybe error) for barplots\n"
		" These textual labels are rendered with the axis font.\n",
		2
	);
	xtb_bt_new( SD_Dialog_win, "Full", SD_sdds_sl_fun, (xtb_data) VMARK_FULL, aF(SNVFM_F) );
	xtb_describe_s( aF(SNVFM_F),
		"Show the error/orientation/intensity value too",
		2
	);
	xtb_bt_new( SD_Dialog_win, "Raw", SD_sdds_sl_fun, (xtb_data) VMARK_RAW, aF(SNVRM_F) );
	xtb_describe_s( aF(SNVRM_F),
		"The raw, untransformed value is shown; the location at which\n"
		" it is shown *does* depend on whatever transformations are defined.\n",
		2
	);

	xtb_bt_new( SD_Dialog_win, "Errrs", SD_sdds_sl_fun, (xtb_data) &set_show_errors, aF(SNSE_F) );
/* 	xtb_bt_set( AF(SNSE_F).win, use_error_Value(), (xtb_data) 0 );	*/
	xtb_describe_s( aF(SNSE_F), "Show this set's errorbars\n", 2);
	xtb_bt_new( SD_Dialog_win, "Ovrwr mrks", SD_sdds_sl_fun, (xtb_data) &set_overwrite_marks, aF(OWM_F) );
/* 	xtb_bt_set( AF(OWM_F).win, overwrite_marks_Value(), (xtb_data) 0 );	*/
	xtb_describe_s( aF(OWM_F), "Draw marks before or after (overwrite) drawing the lines\nconnecting the datapoints\n", 2);
	xtb_bt_new( SD_Dialog_win, "Shw lgnd", SD_sdds_sl_fun, (xtb_data) &set_show_legend, aF(SNSL_F) );
/* 	xtb_bt_set( AF(SNSL_F).win, show_legend_Value(), (xtb_data) 0 );	*/
	xtb_describe_s( aF(SNSL_F), "Show this set in the legend\n", 2);
	SD_info->sn_show_legend= AF(SNSL_F).win;
	xtb_bt_new( SD_Dialog_win, "LegLins", SD_sdds_sl_fun, (xtb_data) &set_show_llines, aF(SNSLLS_F) );
/* 	xtb_bt_set( AF(SNSLLS_F).win, show_llines_Value(), (xtb_data) 0 );	*/
	xtb_describe_s( aF(SNSLLS_F), "Show this set's line (if it has one) in the type 1 legend\n", 2);
	xtb_bt_new( SD_Dialog_win, "Raw", SD_sdds_sl_fun, (xtb_data) &set_raw_display, aF(SNSRAWD_F) );
/* 	xtb_bt_set( AF(SNSRAWD_F).win, raw_display_Value(), (xtb_data) 0 );	*/
	xtb_describe_s( aF(SNSRAWD_F), "Don't apply *DATA_PROCESS* to this set\n", 2);
	xtb_bt_new( SD_Dialog_win, "flt", SD_sdds_sl_fun, (xtb_data) &set_floating, aF(SNFLT_F) );
/* 	xtb_bt_set( AF(SNFLT_F).win, floating_Value(), (xtb_data) 0 );	*/
	xtb_describe_s( aF(SNFLT_F),
		"Don't include this set in automatic scaling\n"
		" And clip to the window instead of the plotting region\n",
		2
	);
	xtb_bt_new( SD_Dialog_win, "New", SD_sdds_sl_fun, (xtb_data) &points_added, aF(SNSPADD_F) );
/* 	xtb_bt_set( AF(SNSPADD_F).win, points_added_Value(), (xtb_data) 0 );	*/
	xtb_describe_s( aF(SNSPADD_F), "Whether this points has had points added to it recently\nE.g. by including a file", 2);

	{ char pc[]= "#xab#xad";
		xtb_bt_new( SD_Dialog_win, parse_codes(pc), SD_sdds_sl_fun, (xtb_data) &set_sarrow, aF(SNSSARROW_F) );
	}
/* 	xtb_bt_set( AF(SNSSARROW_F).win, start_arrow_Value(), (xtb_data) 0 );	*/
	xtb_describe_s( aF(SNSSARROW_F), "Start arrow (at first point)\n", 2);
	{ char pc[]= "#xad#xbb";
		xtb_bt_new( SD_Dialog_win, parse_codes(pc), SD_sdds_sl_fun, (xtb_data) &set_earrow, aF(SNSEARROW_F) );
	}
/* 	xtb_bt_set( AF(SNSEARROW_F).win, end_arrow_Value(), (xtb_data) 0 );	*/
	xtb_describe_s( aF(SNSEARROW_F), "End arrow (at last point)\n", 2);

	xtb_ti_new( SD_Dialog_win, "", D_SN, MAXCHBUF, SD_dfn_fun, (xtb_data) &data_start_arrow_orn, aF(SNSARRORN_F) );
	xtb_describe_s( aF(SNSARRORN_F),
		"The orienation of the start arrow.\n"
		"Changing the value deactivates automatic orientation;\n"
		"Setting to NaN re-activates automatic orientation",
		2
	);
	xtb_ti_new( SD_Dialog_win, "", D_SN, MAXCHBUF, SD_dfn_fun, (xtb_data) &data_end_arrow_orn, aF(SNEARRORN_F) );
	xtb_describe_s( aF(SNEARRORN_F),
		"The orienation of the end arrow.\n"
		"Changing the value deactivates automatic orientation;\n"
		"Setting to NaN re-activates automatic orientation",
		2
	);

	xtb_bt_new( SD_Dialog_win, "Hlt too", SD_sdds_sl_fun, (xtb_data) &AlwaysDrawHighlighted, aF(ADH_F) );
	xtb_describe_s( aF(ADH_F), "If set, highlighted sets are always drawn.\n", 1);

	xtb_bt_new( SD_Dialog_win, "Clear", SD_sdds_sl_fun, (xtb_data) &CleanSheet, aF(RFC_F) );
/* 	xtb_bt_set( AF(RFC_F).win, CleanSheet, (xtb_data) 0 );	*/
	xtb_describe_s( aF(RFC_F), "If set, reading a new file\nerases old datasets.\n", 1);

	xtb_bt_new( SD_Dialog_win, "IRaw", SD_sdds_sl_fun, (xtb_data) &Raw_NewSets, aF(RFRS_F) );
/* 	xtb_bt_set( AF(RFRS_F).win, Raw_NewSets, (xtb_data) 0 );	*/
	xtb_describe_s( aF(RFRS_F), "If set, all (new) sets to which points are added will \n"
		"have their raw field set. And thus escape any further transformations.\n",
		1
	);

	xtb_bt_new( SD_Dialog_win, "Swap [^S]", sdds_rds_fun, (xtb_data) NULL, aF(SNRDS_F) );
	SD_info->sn_reverse_draw_set= AF(SNRDS_F).win;
	xtb_describe_s( aF(SNRDS_F), "Swap drawn sets with not-drawn sets\n", 1);
	xtb_bt_new( SD_Dialog_win, "All [^A]", sdds_rds_fun, (xtb_data) NULL, aF(SNDAS_F) );
	xtb_describe_s( aF(SNDAS_F), "Draw all sets\n", 1);

	xtb_ti_new(SD_Dialog_win, "", DFN_LEN, SD_dfn_fun, (xtb_data) &set_lblx1, aF(LBLX1_F) );
	xtb_describe_s( aF(LBLX1_F),
		" X-coordinate of arrow-side of UserLabel\n"
		" or X-coordinate of legendbox for set shown in the legendbox\n"
		" or X-coordinate of intensitylegend for set not shown in legendbox AND in intensity mode\n"
		" or X-coordinate for X- or Y-label\n",
		2
	);
	xtb_ti_new(SD_Dialog_win, "", DFN_LEN, SD_dfn_fun, (xtb_data) &set_lbly1, aF(LBLY1_F) );
	xtb_describe_s( aF(LBLY1_F),
		" Y-coordinate of arrow-side of UserLabel\n"
		" or Y-coordinate of legendbox for set shown in the legendbox\n"
		" or Y-coordinate of intensitylegend for set not shown in legendbox AND in intensity mode\n"
		" or Y-coordinate for X- or Y-label\n",
		2
	);
	xtb_ti_new(SD_Dialog_win, "", DFN_LEN, SD_dfn_fun, (xtb_data) &set_lblx2, aF(LBLX2_F) );
	xtb_describe_s( aF(LBLX2_F), "X-coordinate of label-side\nof UserLabel\nor *ERROR_POINT* for current set\n"
		" *ERROR_POINT* >= 0: draw errorbar/triangle/region/vector at this point\n"
		"               = -1: draw them all\n"
		"               = -2: draw only the first point with a non-zero error/orientation\n"
		"               = -3: idem for last point\n"
		"               = -4: idem for first AND last point\n"
		"               = -5: draw them all, with first and last non-zero error/orientation in\n"
		"                     highlight or non-highlight depending on the set's highlighting\n"
		"                     (doesn't work for region)\n"
		"               NB: this setting is ignored for the intensity and marker-size modes (Intz, Msze)\n",
		2
	);
	xtb_ti_new(SD_Dialog_win, "", DFN_LEN, SD_dfn_fun, (xtb_data) &set_lbly2, aF(LBLY2_F) );
	xtb_describe_s( aF(LBLY2_F), "Y-coordinate of label-side\nof UserLabel\nor *N* for current set\n", 2);
	xtb_ti_new( SD_Dialog_win, "", D_SN, MAXCHBUF, SD_dfn_fun, (xtb_data) &fileNumber, aF(LSFN_F) );
	xtb_describe_s( aF(LSFN_F), "File number for current set\n Or for a UserLabel whether it is vertical\n", 2 );
	xtb_ti_new( SD_Dialog_win, "", D_SN, MAXCHBUF, SD_dfn_fun, (xtb_data) &set_setlink, aF(LBLSL_F) );
	xtb_describe_s( aF(LBLSL_F), "Set to link label to,\nor -1 for general label,\n"
		" -2= label visible when all marked sets are drawn\n"
		" -3= label visible when all highlighted sets are drawn\n"
		" -4= label visible when none of the marked sets are drawn\n"
		" -5= label visible when none of the highlighted sets are drawn\n"
		" For linked sets, the set linked to is shown\n",
		2
	);
	{ char pc[]= "\\#xbc\\";
		xtb_bt_new( SD_Dialog_win, parse_codes(pc), SD_vanes_or_ultype_selectfun, (xtb_data) NULL, aF(VTTYPE_F) );
	}
	xtb_describe_s( aF(VTTYPE_F), "Vector plot type popup menu", 2 );
	xtb_ti_new( SD_Dialog_win, "", 13, MAXCHBUF, SD_dfn_fun, (xtb_data) &vectorLength, aF(LBLVL_F) );
	xtb_describe_s( aF(LBLVL_F),
		"Type & length of vectors plotted on sets in vector mode :)\n"
		" And possibly vectorparameters as for -vectorpars and *VECTORPARS*:\n"
		"   type 0: 0,<length>\n"
		"   type 1: 1,<length>,<front_fraction[1/3]>,<arrow_length[5]>\n"
		"           negative length inverses direction; <front_fraction> e.g. 1/3 (or 3!; default) means\n"
		"           that the arrowtip is at 1/3 of the vectorlength from the \"centre\"point;\n"
		"           <arrow_length> e.g. 5 (default) means the arrowhead's length is approx. 1/5\n"
		"           of the vectorlength.\n"
		"   type 2: 2\n"
		"           Like type 0, except that the length is taken from the lcol column ($DATA{3})\n"
		"   type 3: 3,<front_fraction[1/3]>,<arrow_length[5]>\n"
		"           Like type 1, except that the length is taken from the lcol column ($DATA{3})\n"
		"   type 4: 4,<length>,<front_fraction[1/3]>,<arrow_length[5]>\n"
		"           Like type 1, except that the length is taken from the lcol column ($DATA{3})\n"
		"           Unlike type 4, the arrowhead's size is based on the <length> parameter.\n"
		" For UserLabels, the specific point linked to, or -1 (changing labeltext unlinks!)\n",
		2
	);

	{ char buf[64];
	  extern char MaxCols;
		sprintf( buf, sncols, MaxCols);
		xtb_to_new( SD_Dialog_win, buf, XTB_TOP_LEFT, &dialogFont.font, NULL, aF(SNCOLS_F) );
		AF(SNCOLS_F).tabset= 2;
	}
	  /* 20020429: implemented the xtb_TI2 widgets, currently only used for easy specification
	   \ of the columns to be used (see the corresponding comment in xtb.c):
	   */
	xtb_ti2_new( SD_Dialog_win, "", D_SN, MAXCHBUF, SD_dfn_fun,
		bt, SD_column_selectfun, (xtb_data) &xcol, aF(SNXC_F) );
	xtb_describe_s( aF(SNXC_F), "# of column with X values\n", 2);
	xtb_ti2_new( SD_Dialog_win, "", D_SN, MAXCHBUF, SD_dfn_fun,
		bt, SD_column_selectfun, (xtb_data) &ycol, aF(SNYC_F) );
	xtb_describe_s( aF(SNYC_F), "# of column with Y values\n", 2);
	xtb_ti2_new( SD_Dialog_win, "", D_SN, MAXCHBUF, SD_dfn_fun,
		bt, SD_column_selectfun, (xtb_data) &ecol, aF(SNEC_F) );
	xtb_describe_s( aF(SNEC_F), "# of column with E values\n", 2);
	xtb_ti2_new( SD_Dialog_win, "", D_SN, MAXCHBUF, SD_dfn_fun,
		bt, SD_column_selectfun, (xtb_data) &lcol, aF(SNVC_F) );
	xtb_describe_s( aF(SNVC_F), "# of column with L (vector length) values\n", 2);
	xtb_ti2_new( SD_Dialog_win, "", D_SN, MAXCHBUF, SD_dfn_fun,
		bt, SD_column_selectfun, (xtb_data) &Ncol, aF(SNNC_F) );
	xtb_describe_s( aF(SNNC_F), "# of column with N (NumObs) values\n", 2);

	xtb_bt_new( SD_Dialog_win, "nobox", sSD_dfn_fun, NULL, aF(SNFNS_F) );
/* 	xtb_bt_set( AF(SNFNS_F).win, theWin_Info->new_file[data_sn_number], (xtb_data) 0);	*/
	xtb_describe_s( aF(SNFNS_F),
		"This button shows whether a subdivision is made at this set\n"
		" For UserLabels, it shows whether or not a frame (box) should be drawn\n", 2);
    xtb_ti_new(SD_Dialog_win, fileName_max, fileName_len, MAXCHBUF, SD_snl_fun, (xtb_data) &new_set, aF(SNFN_F));
/* 	xtb_ti_set( AF(SNFN_F).win, Data_fileName(), (xtb_data) 0);	*/
	xtb_describe_s( aF(SNFN_F), "The (file)Name of the current set(-group)\n"
		"When changing a set's fileName, \"split\" at\n"
		"the next set to make the new name visible\n",
		2
	);
    xtb_bt_new(SD_Dialog_win, "Info", SD_set_info, (xtb_data) 0, aF(SNINF_F));
	xtb_describe_s( aF(SNINF_F),
		"Popup a set's info text, if it has one.\n"
		" If no set or user-label is selected, popup the global info/*VERSION_LIST* data.\n"
		" Mod-click to open an editor for this data.\n",
		2
	);

	{ char pc[]= "#xab<";
		xtb_bt_new( SD_Dialog_win, parse_codes(pc), sdds_shiftset_fun, (xtb_data) &set_shift_left, aF(SNSSU_F) );
	}
	xtb_describe_s( aF(SNSSU_F), "Shift set left (to first with <Mod1>)\n", 2);
	{ char pc[]= ">#xbb";
		xtb_bt_new( SD_Dialog_win, parse_codes(pc), sdds_shiftset_fun, (xtb_data) &set_shift_right, aF(SNSSD_F) );
	}
	xtb_describe_s( aF(SNSSD_F), "Shift set right (to last with <Mod1>)\n", 2);

	xtb_bt_new( SD_Dialog_win, "l_ul", SD_sdds_sl_fun, (xtb_data) &set_legend_placed, aF(SNLP_F) );
	xtb_describe_s( aF(SNLP_F), "Legend box placed at user-coordinates,\nor at default location\n", 1);
	xtb_bt_new( SD_Dialog_win, "il_ul", SD_sdds_sl_fun, (xtb_data) &intensity_legend_placed, aF(SNILP_F) );
	xtb_describe_s( aF(SNILP_F), "Intensity legend placed at user-coordinates,\nor at default location\n", 1);
	xtb_bt_new( SD_Dialog_win, "x_ul", SD_sdds_sl_fun, (xtb_data) &set_xname_placed, aF(SNXP_F) );
	xtb_describe_s( aF(SNXP_F), "X unitname placed at user-coordinates,\nor at default location\n", 1);
	xtb_bt_new( SD_Dialog_win, "y_ul", SD_sdds_sl_fun, (xtb_data) &set_yname_placed, aF(SNYP_F) );
	xtb_describe_s( aF(SNYP_F), "Y unitname placed at user-coordinates,\nor at default location\n", 1);
	xtb_bt_new( SD_Dialog_win, "y_vrt", SD_sdds_sl_fun, (xtb_data) &yname_vertical, aF(SNYV_F) );
	xtb_describe_s( aF(SNYV_F), "Y unitname is drawn vertically\n (in PS: X shows a vertical rect as an indicator)\n", 1);

	xtb_bt_new( SD_Dialog_win, "Av Err.", SD_sdds_sl_fun, (xtb_data) &use_average_error, aF(SUAE_F) );
	xtb_describe_s( aF(SUAE_F), "Use the per-set average error\ninstead of per-point error\n", 1);
	xtb_bt_new( SD_Dialog_win, "No lbox", SD_sdds_sl_fun, (xtb_data) &no_legend_box, aF(SNNLB_F) );
	xtb_describe_s( aF(SNNLB_F), "Do not draw a box around the legends\n", 1);
	xtb_bt_new( SD_Dialog_win, "No ULb", SD_sdds_sl_fun, (xtb_data) &no_ulabels, aF(SNOUL_F) );
	xtb_describe_s( aF(SNOUL_F), "Do not show the UserLabels\n", 1);
	xtb_bt_new( SD_Dialog_win, "No leg.", SD_sdds_sl_fun, (xtb_data) &set_no_legend, aF(SNL_F) );
	xtb_describe_s( aF(SNL_F), "Do not show the legends\n", 1);
	SD_info->sn_no_legend= AF(SNL_F).win;
	xtb_bt_new( SD_Dialog_win, "No ILeg.", SD_sdds_sl_fun, (xtb_data) &no_intensity_legend, aF(SNIL_F) );
	xtb_describe_s( aF(SNIL_F), "Do not show the Intensity legend\n", 1);
	xtb_bt_new( SD_Dialog_win, "Ow. leg.", SD_sdds_sl_fun, (xtb_data) &overwrite_legend, aF(SOL_F) );
	xtb_describe_s( aF(SOL_F), "Legend box overwrites data\n", 1);
	xtb_bt_new( SD_Dialog_win, "No pens", SD_sdds_sl_fun, (xtb_data) &no_pens, aF(SNNP_F) );
	xtb_describe_s( aF(SNNP_F), "Do not draw the pens for this window.\n", 1);
	xtb_bt_new( SD_Dialog_win, "No tit.", SD_sdds_sl_fun, (xtb_data) &set_no_title, aF(SNT_F) );
	xtb_describe_s( aF(SNT_F), "Do not show the title(s) in the graph\n", 1);
	SD_info->sn_no_title= AF(SNT_F).win;
	xtb_bt_new( SD_Dialog_win, "Overlap", SD_sdds_sl_fun, (xtb_data) overlap, aF(SNOL_F) );
	xtb_describe_s( aF(SNOL_F), "Show the average overlap between the\ndisplayed datasets within the legend box\n", 1);
	xtb_bt_new( SD_Dialog_win, "Files", SD_sdds_sl_fun, (xtb_data) &set_filename_in_legend, aF(SNSF_F) );
	xtb_describe_s( aF(SNSF_F), "Show fileNames (or Labels) within the legend box\n", 1);
	xtb_bt_new( SD_Dialog_win, "Labels", SD_sdds_sl_fun, (xtb_data) &labels_in_legend, aF(SNSLL_F) );
	xtb_describe_s( aF(SNSLL_F), "Show YXlabel's instead of fileNames within the legend box\n", 1);
	xtb_bt_new( SD_Dialog_win, "Axes", SD_sdds_sl_fun, (xtb_data) &axisFlag, aF(SNAXF_F) );
	xtb_describe_s( aF(SNAXF_F), "Draw the X and Y axes\n", 1);
	xtb_bt_new( SD_Dialog_win, "Ow.", SD_sdds_sl_fun, (xtb_data) &overwrite_AxGrid, aF(SOAG_F) );
	xtb_describe_s( aF(SOAG_F), "Axes and grid/ticks are drawn after/over data\n", 1);
	xtb_bt_new( SD_Dialog_win, "Border", SD_sdds_sl_fun, (xtb_data) &set_bbFlag, aF(SNBBF_F) );
	xtb_describe_s( aF(SNBBF_F), "Draw a border around the graph\n", 1);
	xtb_bt_new( SD_Dialog_win, "H.", SD_sdds_sl_fun, (xtb_data) &htickFlag, aF(SNHTKF_F) );
	xtb_describe_s( aF(SNHTKF_F), "Draw horizontal gridlines or ticks along the Y-axis?\n", 1);
	xtb_bt_new( SD_Dialog_win, "V.grid", SD_sdds_sl_fun, (xtb_data) &vtickFlag, aF(SNVTKF_F) );
	xtb_describe_s( aF(SNVTKF_F), "Draw vertical gridlines or ticks along the X-axis?\n", 1);
	xtb_bt_new( SD_Dialog_win, "Zero", SD_sdds_sl_fun, (xtb_data) &set_zeroFlag, aF(SNZLF_F) );
	xtb_describe_s( aF(SNZLF_F), "Draw lines x=0 and y=0\n", 1);

	xtb_br2D_new(SD_Dialog_win, LEGENDFUNCTIONS, LegendFTypes, 0,0, format_legfun, LegendFunctionNr,
		   sn_legend_fun, (xtb_data) SD_info, aF(SNLEGFUN_F)
	);
	xtb_describe_s( aF(SNLEGFUN_F),
		" Specifies what the legend-window shows\n\001When a set is selected\n"
		" Initially, the function needing the narrowest width is selected\n",
		2
	);
	xtb_describe( AF(SNLEGFUN_F).framelist[0], "Legend-window edits a set's legend\n");
	xtb_describe( AF(SNLEGFUN_F).framelist[1], "Legend-window edits a set's title\n");
	xtb_describe( AF(SNLEGFUN_F).framelist[2], "Legend-window edits a set's *SET_PROCESS*\n");
#ifdef OLD_ASSOC_DISPLAY
	xtb_describe( AF(SNLEGFUN_F).framelist[3], "Legend-window shows a (possibly incomplete list of a) set's *ASSOCIATE* associations\n");
#else
	xtb_describe( AF(SNLEGFUN_F).framelist[3], "Legend-window edits a set's *ASSOCIATE* associations\n");
#endif
	xtb_describe( AF(SNLEGFUN_F).framelist[4],
		" Legend-window edits a set's or label's colour\n"
		" Specify \"default\" to return to -Colour or -MonoChrome default XGraph value\n"
		" For linked labels, specify \"linked\" to use the linked-to set's colour\n"
	);
	xtb_describe( AF(SNLEGFUN_F).framelist[5],
		" Legend-window edits a set's highlighting colour\n"
		" Specify \"default\" to return to default highlightPixel value (-hc option)\n"
	);
	xtb_describe( AF(SNLEGFUN_F).framelist[6],
		" Legend-window edits the *TRANSFORM_X/Y*, *DATA_PROCESS*, or *SET_PROCESS* description\n"
	);
	xtb_describe( AF(SNLEGFUN_F).framelist[7],
		" Legend-window shows the set's *LABELS* data that override the %CX, %CY, etc. values\n"
		" set through the global specifier *COLUMNLABELS*. Editing is only possible via the\n"
		" editor button next to the legend-window!\n"
	);
	SD_LMaxBufSize= LMAXBUFSIZE;
    xtb_to_new2(SD_Dialog_win, setName_max, legend_len, XTB_TOP_LEFT, &dialogFont.font, NULL, aF(SNpLEG_F));
	xtb_describe_s( aF(SNpLEG_F), LegendFTypes[pLegFunNr], 0 );
    xtb_ti_new(SD_Dialog_win, setName_max, legend_len, SD_LMaxBufSize, SD_snl_fun, (xtb_data) &new_legend, aF(SNLEG_F));
    SD_info->sn_legend = AF(SNLEG_F).win;
	xtb_describe( aF(SNLEG_F),
		"The legend for the current dataset\n"
		" OR an axis-label\n"
		" or a *DATA_???*, *TRANSFORM_?*, *EVAL* expression\n"
		" or a filename or pipe-command to read from\n"
		" Hit TAB or Cursor Up/Down to accept ALL (new) settings\n"
		" for this set; TAB also advances one set. ESC backs up and discards.\n"
	);
	xtb_bt_new(SD_Dialog_win, "...", SD_edit_fun, (xtb_data) SD_edit_fun, aF(SNLEGEDIT_F) );
	xtb_describe( aF(SNLEGEDIT_F),
		"Click this button to call up an editor window ($XG_EDITOR)\n"
		" to edit the current input buffer.\n"
	);

#ifdef SLIDER
	xtb_sr_new( SD_Dialog_win, 10, 110, 55.1234, - legend_len, 1, slide_f, NULL, aF(SLIDE_F));
	xtb_describe( aF(SLIDE_F), "A Test Slide ruler\n");
	xtb_sr_new( SD_Dialog_win, 110, 10, 55.1234, - legend_len, 1, slide_f, NULL, aF(SLIDE2_F));
	xtb_describe( aF(SLIDE2_F), "A reversed Test Slide ruler\n");
#endif

	if( theWin_Info->logXFlag< 0 ){
		theWin_Info->logXFlag= 0;
	}
	if( theWin_Info->logYFlag< 0 ){
		theWin_Info->logYFlag= 0;
	}
	if( theWin_Info->sqrtXFlag< 0 ){
		theWin_Info->sqrtXFlag= 0;
	}
	if( theWin_Info->sqrtYFlag< 0 ){
		theWin_Info->sqrtYFlag= 0;
	}
	xtb_bt_new(SD_Dialog_win, "absY", SD_sdds_sl_fun, (xtb_data) &absYFlag, aF(SNTAYF_F) );
	xtb_describe_s( aF(SNTAYF_F), "Take absolute value of Y-values\n", 1);
	xtb_bt_new(SD_Dialog_win, "polar", SD_sdds_sl_fun, (xtb_data) &polarFlag, aF(SNTPF_F) );
	xtb_describe_s( aF(SNTPF_F), "Enables polar mode\n", 1);
	xtb_bt_new(SD_Dialog_win, "logX", SD_sdds_sl_fun, (xtb_data) &logXFlag, aF(SNTLX_F) );
	xtb_describe_s( aF(SNTLX_F), "Enable logarithmic transformation\n", 1);
	xtb_bt_new(SD_Dialog_win, "logY", SD_sdds_sl_fun, (xtb_data) &logYFlag, aF(SNTLY_F) );
	xtb_describe_s( aF(SNTLY_F), "Enable logarithmic transformation\n", 1);
/* 
	xtb_bt_new(SD_Dialog_win, "logX2", SD_sdds_sl_fun, (xtb_data) &logXFlag2, aF(SNTLX2_F) );
	xtb_bt_set( AF(SNTLX2_F).win, theWin_Info->logXFlag== 3, NULL);
	xtb_bt_new(SD_Dialog_win, "logY2", SD_sdds_sl_fun, (xtb_data) &logYFlag2, aF(SNTLY2_F) );
	xtb_bt_set( AF(SNTLY2_F).win, theWin_Info->logYFlag== 3, NULL);
 */
	xtb_bt_new(SD_Dialog_win, "powX", SD_sdds_sl_fun, (xtb_data) &sqrtXFlag, aF(SNTSX_F) );
	xtb_describe_s( aF(SNTSX_F), "Enable pow(x,<power>) transformation\n", 1);
	xtb_bt_new(SD_Dialog_win, "powY", SD_sdds_sl_fun, (xtb_data) &sqrtYFlag, aF(SNTSY_F) );
	xtb_describe_s( aF(SNTSY_F), "Enable pow(y,<power>) transformation\n", 1);

	xtb_ti_new( SD_Dialog_win, (AxisValueFormat)? AxisValueFormat : "", 8, 10,
		SD_df_fun, (xtb_data) &AxisValueFormat, aF(AXSTR_F)
	);
	xtb_describe_s( aF(AXSTR_F), "printf(3) format string for printing axis numbers\n"
		" If not used, an internal heuristic is used\n",
		1
	);
	xtb_ti_new( SD_Dialog_win, "", D_SN, MAXCHBUF, SD_dfn_fun, (xtb_data) &AxisValueMinDigits, aF(AXMIDIG_F) );
	xtb_describe_s( aF(AXMIDIG_F),
		" The minimal # of decimals to show in automatically formatted axis numbers\n"
		" Used only when no specific axis number formatstring has been specified\n",
		1
	);

	xtb_to_new(SD_Dialog_win, "AxPars:", XTB_TOP_LEFT, &dialogFont.font, NULL, aF(BOUNDS_F) );
	AF(BOUNDS_F).tabset= 1;
	xtb_bt_new(SD_Dialog_win, "FitX", SD_sdds_sl_fun, (xtb_data) &fit_x, aF(XFIT_F) );
	xtb_describe_s( aF(XFIT_F), "Fit the X bounds to the current range\n"
		"Hold the Shift key to switch both fitting buttons\n",
		1
	);
	xtb_bt_new(SD_Dialog_win, "FitY", SD_sdds_sl_fun, (xtb_data) &fit_y, aF(YFIT_F) );
	xtb_describe_s( aF(YFIT_F), "Fit the Y bounds to the current range\n"
		"Hold the Shift key to switch both fitting buttons\n",
		1
	);
	xtb_bt_new(SD_Dialog_win, "FitPB", SD_sdds_sl_fun, (xtb_data) &fit_pB, aF(PBFIT_F) );
	xtb_describe_s( aF(PBFIT_F), "Fit the radix to the current range:\n"
		"base=MaxAngle - MinAngle\n"
		"Hold the Shift key to switch all fitting buttons\n",
		1
	);
	xtb_bt_new(SD_Dialog_win, "After", SD_sdds_sl_fun, (xtb_data) &fit_after_draw, aF(FITAD_F) );
	xtb_describe_s( aF(FITAD_F), "Do the fitting after the redraw, instead of before.\n"
		" When fitting after, redrawing continues until the final fit is found.\n",
		1
	);
	xtb_bt_new(SD_Dialog_win, "1:1", SD_sdds_sl_fun, (xtb_data) &aspect, aF(ASPECT_F) );
	xtb_describe_s( aF(ASPECT_F), "Apply largest range to both axes to have aspect 1:1\n"
		"Hold down the Mod1 key to prevent automatic redraw\n",
		1
	);
	xtb_bt_new(SD_Dialog_win, "X:X", SD_sdds_sl_fun, (xtb_data) &XSymmetric, aF(XSYMM_F) );
	xtb_describe_s( aF(XSYMM_F), "Make X axis symmetric around X=0\n", 1);
	xtb_bt_new(SD_Dialog_win, "Y:Y", SD_sdds_sl_fun, (xtb_data) &YSymmetric, aF(YSYMM_F) );
	xtb_describe_s( aF(YSYMM_F), "Make Y axis symmetric around Y=0\n", 1);

	xtb_bt_new( SD_Dialog_win, "Legend", SD_sdds_sl_fun, (xtb_data) &legend_always_visible, aF(SNLAV_F) );
	xtb_describe_s( aF(SNLAV_F), "Adapt the data-window to ensure that the legendbox is always visible\n"
		" X and Y label are considered too when placed [x_ul] and/or [y_ul]",
		1
	);
	xtb_bt_new( SD_Dialog_win, "ExctX", SD_sdds_sl_fun, (xtb_data) &exact_X_axis, aF(SNEXA_F) );
	xtb_describe_s( aF(SNEXA_F),
		" Always print X axis values exactly where they should be\n"
		" (Prevents round-off errors but can cause weird behaviour)\n"
		" When VCatX is selected, prevents the use of nearest-neighbour categories\n"
		" in case an axis-value is not represented\n",
		1
	);
	xtb_bt_new( SD_Dialog_win, "ExctY", SD_sdds_sl_fun, (xtb_data) &exact_Y_axis, aF(SNEYA_F) );
	xtb_describe_s( aF(SNEYA_F),
		" Always print Y axis values exactly where they should be\n"
		" (Prevents round-off errors but can cause weird behaviour)\n"
		" When VCatY is selected, prevents the use of nearest-neighbour categories\n"
		" in case an axis-value is not represented\n",
		1
	);
	xtb_bt_new( SD_Dialog_win, "VCatX", SD_sdds_sl_fun, (xtb_data) &ValCat_X_axis, aF(SNVCXA_F) );
	xtb_describe_s( aF(SNVCXA_F), " Activate a Value<>Category X axis (use *VAL_CAT_X*)\n"
		" Shift-click for options\n", 1);
	xtb_bt_new( SD_Dialog_win, "VCatG", SD_sdds_sl_fun, (xtb_data) &ValCat_X_grid, aF(SNVCXG_F) );
	xtb_describe_s( aF(SNVCXG_F), " Draw a vertical grid for the Value<>Category X axis\n", 1);
	xtb_ti_new( SD_Dialog_win, "", D_SN, MAXCHBUF, SD_dfn_fun, (xtb_data) &ValCat_X_levels, aF(SNVCXL_F) );
	xtb_describe_s( aF(SNVCXL_F), " Over how many lines the X ValCategories are drawn\n", 1);
	xtb_bt_new( SD_Dialog_win, "VCatY", SD_sdds_sl_fun, (xtb_data) &ValCat_Y_axis, aF(SNVCYA_F) );
	xtb_describe_s( aF(SNVCYA_F), " Activate a Value<>Category Y axis (use *VAL_CAT_Y*)\n"
		" Shift-click for options\n", 1);

	xtb_bt_new( SD_Dialog_win, "All:", SD_sdds_sl_fun, (xtb_data) &show_all_ValCat_I, aF(SNSAVCI_F) );
	xtb_describe_s( aF(SNSAVCI_F), " Show all intensity categories (*VAL_CAT_I*) on the Intensity legend\n", 1);
	xtb_bt_new( SD_Dialog_win, "VCatI", SD_sdds_sl_fun, (xtb_data) &ValCat_I_axis, aF(SNVCIA_F) );
	xtb_describe_s( aF(SNVCIA_F), " Activate a Value<>Category Intensity legend (use *VAL_CAT_I*)\n"
		" Shift-click for options\n", 1);

	xtb_ti_new(SD_Dialog_win, "", DFN_LEN, SD_dfn_fun, (xtb_data) &set_xmin, aF(XMIN_F) );
	xtb_describe_s( aF(XMIN_F), "Left bound of X-axis\n See also describtion of the [nobb] button\n", 1);
	xtb_ti_new(SD_Dialog_win, "", DFN_LEN, SD_dfn_fun, (xtb_data) &set_ymin, aF(YMIN_F) );
	xtb_describe_s( aF(YMIN_F), "Left bound of Y-axis\nNot used in polar mode\n See also describtion of the [nobb] button\n", 1);
/*     xtb_bk_new( SD_Dialog_win, (int)(2.5* XFontWidth(dialogFont.font)), 3, aF(BOUNDSSEP_F));	*/
	xtb_bt_new(SD_Dialog_win, "nobb", SD_sdds_sl_fun, (xtb_data) &nobb_coordinates, aF(NOBB_F) );
	xtb_describe_s( aF(NOBB_F),
		"When set, the axes-bounds fields show the bounds\n"
		" used for drawing the axes when no rectangular border\n"
		" is to be drawn. These nobb bounds can be specified\n"
		" independently for X and Y axes; set to NaN to desactivate\n"
		" a given range. When nobb bounds are set, this button's shadow\n"
		" changes from black to dark red to indicate the fact.\n",
		1
	);
	xtb_ti_new(SD_Dialog_win, "", DFN_LEN, SD_dfn_fun, (xtb_data) &set_xmax, aF(XMAX_F) );
	xtb_describe_s( aF(XMAX_F), "Right bound of X-axis\n See also describtion of the [nobb] button\n", 1);
	xtb_ti_new(SD_Dialog_win, "", DFN_LEN, SD_dfn_fun, (xtb_data) &set_ymax, aF(YMAX_F) );
	xtb_describe_s( aF(YMAX_F), "Highest bound of Y-axis\n See also describtion of the [nobb] button\n\n", 1);
	xtb_bt_new(SD_Dialog_win, "user", SD_sdds_sl_fun, (xtb_data) &user_coordinates, aF(USER_F) );
/* 	xtb_bt_set( AF(USER_F).win, theWin_Info->win_geo.user_coordinates, NULL);	*/
	xtb_describe_s( aF(USER_F), "Are these coordinates specified by the user\n(by zooming/settings dialogue), or\n"
		"automatic (keeping log_zero within the bounds)?\n"
		"Set automatically, unless when holding Mod1 when accepting new bound",
		1
	);
	xtb_bt_new(SD_Dialog_win, "exchng", SD_swap_pure, (xtb_data) NULL, aF(SWAPPURE_F) );
	xtb_describe_s( aF(SWAPPURE_F), "Exchange bounds (data with padding)\nwith pure_bounds (data without padding)", 1);
	xtb_bt_new(SD_Dialog_win, "proc", SD_sdds_sl_fun, (xtb_data) &process_bounds, aF(PROCB_F) );
/* 	xtb_bt_set( AF(PROCB_F).win, theWin_Info->process_bounds, NULL);	*/
	xtb_describe_s( aF(PROCB_F),
		"Whether or not to process the axes-bounds (min,max)\nwith the *DATA_PROCESS* and *TRANSFORM_[XY]* filters\n"
		"Unset => axes show original/user-selected data ranges\n"
		"Hold down the Mod1 key to prevent automatic redraw with new setting\n",
		1
	);
	xtb_bt_new(SD_Dialog_win, "raw", SD_sdds_sl_fun, (xtb_data) &raw_display, aF(RAWD_F) );
/* 	xtb_bt_set( AF(RAWD_F).win, theWin_Info->raw_display, NULL);	*/
	xtb_describe_s( aF(RAWD_F), "Whether or not to apply the\n*TRANSFORM_?* and *DATA_PROCESS* filters\n", 1);
	xtb_bt_new(SD_Dialog_win, "trax", SD_sdds_sl_fun, (xtb_data) &transform_axes, aF(TRAX_F) );
/* 	xtb_bt_set( AF(TRAX_F).win, theWin_Info->transform_axes, NULL);	*/
	xtb_describe_s( aF(TRAX_F), "Whether or not to process the axes numbers\nwith the *TRANSFORM_[XY]* filters\n"
		"Set => axes show transformed values\n"
		"Unset => axes show \"real\" values\n"
		"Hold down the Mod1 key to prevent automatic redraw with new setting\n",
		1
	);

	xtb_to_new(SD_Dialog_win, "Step:", XTB_TOP_LEFT, &dialogFont.font, NULL, aF(SNIFLBL_F) );
	AF(SNIFLBL_F).tabset= 1;
	xtb_to_new(SD_Dialog_win, "Scale:", XTB_TOP_LEFT, &dialogFont.font, NULL, aF(SNS2LBL_F) );
	AF(SNS2LBL_F).tabset= 1;
	xtb_to_new(SD_Dialog_win, "Bias:", XTB_TOP_LEFT, &dialogFont.font, NULL, aF(SNBTLBL_F) );
	AF(SNBTLBL_F).tabset= 1;
	xtb_to_new(SD_Dialog_win, "log_zero:", XTB_TOP_LEFT, &dialogFont.font, NULL, aF(SNLZLBL_F) );
	AF(SNLZLBL_F).tabset= 1;
	xtb_bt_new(SD_Dialog_win, "x", SD_sdds_sl_fun, (xtb_data) &lz_sym_x, aF(SNLZX_F) );
/* 	xtb_bt_set( AF(SNLZX_F).win, theWin_Info->lz_sym_x, NULL);	*/
	xtb_describe_s( aF(SNLZX_F), "Use user symbol for the substitute value\n log_zero_x for x=0\n", 1);
	xtb_bt_new( SD_Dialog_win, "-", SD_sdds_sl_fun, (xtb_data) &log_zero_x_mFlag, aF(SNLZXMI_F) );
	xtb_describe_s( aF(SNLZXMI_F), "log_zero_x at minimal X coordinate",1 );
	xtb_bt_new( SD_Dialog_win, "+", SD_sdds_sl_fun, (xtb_data) &log_zero_x_mFlag, aF(SNLZXMA_F) );
	xtb_describe_s( aF(SNLZXMA_F), "log_zero_x at maximal X coordinate",1 );
/* 	xtb_bt_set( AF(SNLZXMI_F).win, (theWin_Info->log_zero_x_mFlag< 0), NULL);	*/
	xtb_bt_set( AF(SNLZXMA_F).win, (theWin_Info->log_zero_x_mFlag> 0), NULL);

	xtb_bt_new(SD_Dialog_win, "y", SD_sdds_sl_fun, (xtb_data) &lz_sym_y, aF(SNLZY_F) );
/* 	xtb_bt_set( AF(SNLZY_F).win, theWin_Info->lz_sym_y, NULL);	*/
	xtb_describe_s( aF(SNLZY_F), "Use user symbol for the substitute value\n log_zero_y for y=0\n", 1);
	xtb_bt_new( SD_Dialog_win, "-", SD_sdds_sl_fun, (xtb_data) &log_zero_y_mFlag, aF(SNLZYMI_F) );
	xtb_describe_s( aF(SNLZYMI_F), "log_zero_y at minimal Y coordinate", 1 );
	xtb_bt_new( SD_Dialog_win, "+", SD_sdds_sl_fun, (xtb_data) &log_zero_y_mFlag, aF(SNLZYMA_F) );
	xtb_describe_s( aF(SNLZYMA_F), "log_zero_y at maximal Y coordinate", 1 );
/* 	xtb_bt_set( AF(SNLZYMI_F).win, (theWin_Info->log_zero_y_mFlag< 0), NULL);	*/
	xtb_bt_set( AF(SNLZYMA_F).win, (theWin_Info->log_zero_y_mFlag> 0), NULL);
	{ char buf[64];

		xtb_ti_new(SD_Dialog_win, "", DFN_LEN, SD_dfn_fun, (xtb_data) &Xincr_factor, aF(SNXIF_F) );
		xtb_describe_s( aF(SNXIF_F), "Step factor for X-axis numbers\n", 1);

		xtb_ti_new(SD_Dialog_win, "", DFN_LEN, SD_dfn_fun, (xtb_data) &Yincr_factor, aF(SNYIF_F) );
		xtb_describe_s( aF(SNYIF_F), "Step factor for Y-axis numbers\n", 1);

		xtb_ti_new(SD_Dialog_win, "", DFN_LEN, SD_dfn_fun, (xtb_data) &Xscale2, aF(SNS2X_F) );
		xtb_describe_s( aF(SNS2X_F), "Scale factor for X-values\nScaling takes place before other transformations\n", 1);
/* 			xtb_disable( AF(SNS2X_F).win );	*/
		xtb_ti_new(SD_Dialog_win, "", DFN_LEN, SD_dfn_fun, (xtb_data) &Yscale2, aF(SNS2Y_F) );
/* 			xtb_disable( AF(SNS2Y_F).win );	*/
		xtb_describe_s( aF(SNS2Y_F),
				"Scale factor for Y-values\n" \
				"Scaling takes place before other transformations\n" \
				"Set to zero to calculate scale factor that scales\n" \
				"all Y>= 1.0 or Y<=-1\n",
				1
		);
		xtb_ti_new(SD_Dialog_win, "", DFN_LEN, SD_dfn_fun, (xtb_data) &Xbias_thres, aF(SNBTX_F) );
		xtb_describe_s( aF(SNBTX_F), "Axis-delta below which translation over average takes place\n", 1);
		xtb_ti_new(SD_Dialog_win, "", DFN_LEN, SD_dfn_fun, (xtb_data) &Ybias_thres, aF(SNBTY_F) );
		xtb_describe_s( aF(SNBTY_F), "Axis-delta below which translation over average takes place\n", 1);
		   /* 20010721: *** verify whether the following distinction is still of use, and/or needs porting
		    \ to places where polarFlag changes!
			*/
		if( !theWin_Info->polarFlag ){
			xtb_ti_new(SD_Dialog_win, "", DFN_LEN, SD_dfn_fun, (xtb_data) &powXFlag, aF(SNTPX_F) );
		}
		else{
			xtb_ti_new(SD_Dialog_win, "", DFN_LEN, SD_dfn_fun, (xtb_data) &powAFlag, aF(SNTPX_F) );
		}
		xtb_describe_s( aF(SNTPX_F), "Power factor for pow(x,<power>) transformation\nor cos(x) and sin(x) in polar mode\n", 1);
		d3str( buf, d3str_format, theWin_Info->log_zero_x );
		xtb_ti_new(SD_Dialog_win, "", DFN_LEN, SD_dfn_fun, (xtb_data) &log_zero_x, aF(SNLZXV_F) );
/* 		xtb_ti_set( AF(SNLZXV_F).win, buf, NULL );	*/
		xtb_describe_s( aF(SNLZXV_F), "log_zero_x value to substitute\nfor x=0 and log X-axis\n", 1);
		xtb_ti_new(SD_Dialog_win, "", DFN_LEN, SD_dfn_fun, (xtb_data) &radix, aF(SNTPB_F) );
		xtb_describe_s( aF(SNTPB_F), "The goniometric base (radix) for polar plots, vector plotting, etc.\n", 1);
		xtb_ti_new(SD_Dialog_win, "", DFN_LEN, SD_dfn_fun, (xtb_data) &radix_offset, aF(SNTPO_F) );
		xtb_describe_s( aF(SNTPO_F),
			"The goniometric base offset (radix_offset) for polar plots, vector plotting, etc.\n"
			" 0 (default): 0 degrees is rightwards; 90 (for radix 360): 0 degrees points upwards, etc.\n",
			1
		);
		xtb_ti_new(SD_Dialog_win, "", DFN_LEN, SD_dfn_fun, (xtb_data) &powYFlag, aF(SNTPY_F) );
		d3str( buf, d3str_format, theWin_Info->log_zero_y );
		AF(SNTPY_F).tabset= 1;
		xtb_ti_new(SD_Dialog_win, "", DFN_LEN, SD_dfn_fun, (xtb_data) &log_zero_y, aF(SNLZYV_F) );
/* 		xtb_ti_set( AF(SNLZYV_F).win, buf, NULL );	*/
		AF(SNLZYV_F).tabset= 1;
	}
	xtb_ti_new( SD_Dialog_win, "", MAX(10,SD_strlen(theWin_Info->log_zero_sym_x)), MAXCHBUF, SD_df_fun, (xtb_data) log_zero_sym_x, aF(SNLZXS_F) );
/* 	xtb_ti_set( AF(SNLZXS_F).win, theWin_Info->log_zero_sym_x, NULL );	*/
	xtb_describe_s( aF(SNLZXS_F), "Symbol to use for the substitute value\n log_zero_x for x=0\n", 1);
	xtb_ti_new( SD_Dialog_win, "", MAX(10,SD_strlen(theWin_Info->log_zero_sym_x)), MAXCHBUF, SD_df_fun, (xtb_data) log_zero_sym_y, aF(SNLZYS_F) );
/* 	xtb_ti_set( AF(SNLZYS_F).win, theWin_Info->log_zero_sym_y, NULL );	*/
	AF(SNLZYS_F).tabset= 1;

    xtb_bt_new(SD_Dialog_win, "Ok [Return]", ok_fun, (xtb_data) SD_info, aF(OK_F));
	xtb_describe( aF(OK_F), "Close this dialogue, redrawing if necessary\nHold <Mod1> to delete the dialog\n");
    xtb_bt_new(SD_Dialog_win, "Redraw [^R]", SD_redraw_fun, (xtb_data) 0, aF(REDRAW_F));
	xtb_describe( aF(REDRAW_F), "Redraw the graph window belonging to this dialogue,\nshowing new settings\n"
		" Hold Alt (Meta/Mod1) key to show 1st drawn set's settings\n"
		" Hold Shift key to redraw in XSynchronised mode\n"
		" Use right button to (map and) raise the target window first.\n"
	);
    xtb_bt_new(SD_Dialog_win, "Auto", SD_redraw_fun, (xtb_data) &autoredraw, aF(AUTOREDRAW_F));
	xtb_describe( aF(AUTOREDRAW_F),
		"When set, the active window is redrawn automatically for any change that\n"
		" sets the \"to-be-redrawn\" flag. Note that some settings always do this, unless\n"
		" they were changed with the <Mod1> key held.\n"
	);
    xtb_bt_new(SD_Dialog_win, "Update [^^]", SD_update, (xtb_data) 0, aF(UPDATE_F));
	xtb_describe( aF(UPDATE_F), "update settings without redrawing\n");
/*     xtb_bt_new(SD_Dialog_win, "Print  [^P]", print_fun, (xtb_data) 0, aF(PRINT_F));	*/
/* 	xtb_describe( aF(PRINT_F), "Close this dialogue, and open the settings dialogue\n");	*/
    xtb_bt_new(SD_Dialog_win, "Close", SD_quit_fun, (xtb_data) 0, aF(QUIT_F));
	xtb_describe( aF(QUIT_F), "Shift-Mod1-click to close the owning window, or quit if the Controlkey is pressed too\n");

    xtb_bt_new(SD_Dialog_win, "Hist", SD_process_hist, (xtb_data) 0, aF(PHIST_F));
	xtb_describe( aF(PHIST_F), "Popup a menu with the history of processes\n<Mod1>-T\n");

    xtb_bt_new(SD_Dialog_win, "Args", SD_option_hist, (xtb_data) 0, aF(PARGS_F));
	xtb_describe( aF(PARGS_F), "Popup a menu with the history of commandline options\n");

    xtb_bt_new(SD_Dialog_win, "Fnc", SD_help_fnc, (xtb_data) 0, aF(HELP_F));
	xtb_describe( aF(HELP_F), "Popup a menu with all fascanf functions & variables\n<Mod1>-?\n"
		"Hold down the Mod1 key to forcedly rebuild (refresh) the menu\n"
	);

    xtb_bt_new(SD_Dialog_win, "Vars [^_,^/]", display_ascanf_variables_h, (xtb_data) 0, aF(VARIABLES_F));
	xtb_describe( aF(VARIABLES_F), "Popup a menu with the currently defined fascanf variables\n");

	SD_Dialog.frames= LAST_F;
	SD_Dialog.framelist= &local_sd_af;

      /* 
       * Now place elements - could make this one expression but pcc
       * is too wimpy.
       */
/* 	update_SD_size();	*/
/* 	format_SD_Dialog( frame, 1 );	*/
/* 	set_changeables(0,True);	*/

    xtb_bk_new( frame->win, 1, 1, aF(BAR_F));
	xtb_describe( aF(BAR_F), "Below this line (most) fields are set-specific\n");

    xtb_bk_new( frame->win, 1, D_VPAD, aF(VBAR_F));
    xtb_bk_new( frame->win, D_HPAD, 1, aF(TBAR_F));

	XSelectInput(disp, SD_Dialog_win, 
		/* VisibilityChangeMask|*/ExposureMask|StructureNotifyMask|ButtonPressMask|ButtonReleaseMask|KeyPressMask
	);

    diag_cursor = XCreateFontCursor(disp, XC_question_arrow );
    fg_color.pixel = normPixel;
    XQueryColor(disp, cmap, &fg_color);
    bg_color.pixel = bgPixel;
    XQueryColor(disp, cmap, &bg_color);
    XRecolorCursor(disp, diag_cursor, &fg_color, &bg_color);
    XDefineCursor(disp, SD_Dialog_win, diag_cursor);
    *OKbtn = AF(OK_F);
    *REDRAWbtn = AF(REDRAW_F);

	get_data_legend_buf();
	xtb_ti_set( SD_info->sn_legend, data_legend_buf, (xtb_data) 0 );
	xtb_init(disp, screen, normPixel, bgPixel, dialogFont.font, dialog_greekFont.font, False );
}


#define SH_W	5
#define SH_H	5

int Handle_SD_Event( XEvent *evt, int *handled, xtb_hret *xtb_return, int *level, int handle_others )
{
  char keys[4];
  KeySym keysymbuffer[4];
  int rd= theWin_Info->redraw, nbytes, keysyms;
  extern xtb_frame HO_Dialog, HO_okbtn;
  extern xtb_hret HO_ok_fun();
  extern char ps_comment[1024];

	if( Exitting ){
		return(1);
	}

	if( SD_Dialog.destroy_it ){
		_CloseSD_Dialog( &SD_Dialog, True );
		return(0);
	}

	if( debugFlag ){
		fprintf( StdErr, "Handle_SD_Event(%d,\"%s\") #%lu handled=%d, handle_others=%d\n",
			level, event_name(evt->type), evt->xany.serial,
			*handled, handle_others
		);
		fflush( StdErr );
	}
	if( evt->xany.type== KeyPress || evt->xany.type== KeyRelease || evt->xany.type== ButtonPress ||
		evt->xany.type== ButtonRelease || evt->xany.type== MotionNotify
	){
		xtb_modifier_state= xtb_Mod2toMod1( 0xff & evt->xbutton.state );
		xtb_button_state= 0xff00 & evt->xbutton.state;
	}
	else{
		xtb_modifier_state= 0;
		xtb_button_state= 0;
	}

	if( data_sn_number!= dsnn ){
		set_changeables(2,True);
	}

	switch( evt->type){
		case ClientMessage:{
			if( SD_Dialog.win== evt->xclient.window && evt->xclient.data.l[0]== wm_delete_window &&
				strcmp( XGetAtomName(disp, evt->xclient.message_type), "WM_PROTOCOLS")== 0
			){
				_CloseSD_Dialog( &SD_Dialog, True );
				  /* 990928: should be here?!	*/
				*handled= 1;
			}
			break;
		}
		case UnmapNotify:
			if( evt->xany.window== SD_Dialog.win ){
				SD_Dialog.mapped= 0;
				  /* 990928: should be here?!	*/
				*handled= 1;
			}
			break;
		case MapNotify:
			if( evt->xany.window== SD_Dialog.win ){
				SD_Dialog.mapped= 1;
				set_changeables(2,True);
				  /* 990928: should be here?!	*/
				*handled= 1;
			}
			break;
		case ConfigureNotify:
			if( evt->xany.window== SD_Dialog.win ){
			  XConfigureEvent *e= (XConfigureEvent*) evt;
			  int width= e->width, height= e->height;
			  Window dummy;
				if( !update_SD_size() ){
				  Boolean pw_ok;
					if( SD_Dialog.width!= width+ 2* D_BRDR || SD_Dialog.height!= height+ 2* D_BRDR ){
					  Boolean jr= JustResized;
						format_SD_Dialog( &SD_Dialog, 1 );
						if( !jr && (SD_Dialog.width!= width+ 2* D_BRDR || SD_Dialog.height!= height+ 2* D_BRDR) ){
							xtb_error_box( theWin_Info->window,
								"Sorry, Dialog-sizes are fixed, depending\n"
								"on area required for the buttons etc.\n",
								"Note"
							);
							JustResized= False;
							pw_ok= False;
						}
						else{
							pw_ok= True;
						}
					}
					else{
					  /* probably just a translation. No need to check and see if the size is (still) correct	*/
						pw_ok= False;
					}
					XTranslateCoordinates( disp, SD_Dialog.win, RootWindow(disp, screen),
							  0, 0, &e->x, &e->y, &dummy
					);
					if( debugFlag ){
						fprintf( StdErr, "SD_Dialog %dx%d+%d+%d ConfigureEvent %dx%d+%d+%d-%d ok=%d\n",
							SD_Dialog.width, SD_Dialog.height, SD_Dialog.x_loc, SD_Dialog.y_loc,
							e->width, e->height, e->x, e->y, WM_TBAR, pw_ok
						);
					}
					e->y-= WM_TBAR;
					if( pw_ok || SD_Dialog.x_loc!= e->x || SD_Dialog.y_loc!= e->y ){
						pw_centre_on_X= e->x+ width/2+ e->border_width;
						pw_centre_on_Y= e->y+ height/2+ e->border_width;
						SD_Dialog.x_loc= e->x;
						SD_Dialog.y_loc= e->y;
					}
				}
				*handled= 1;
			}
			  /* fall through to Expose handler	*/
		case Expose:
			if( evt->xany.window== SD_Dialog.win ){
			  XPoint line[3];
			  GC lineGC;
			  extern GC xtb_set_gc();

				line[0].x= 0;
				line[0].y= (short) SD_Dialog.height- 2* D_BRDR- 1;
				line[1].x= (short) SD_Dialog.width- 2* D_BRDR- 1;
				line[1].y= (short) SD_Dialog.height- 2* D_BRDR- 1;
				line[2].x= (short) SD_Dialog.width- 2* D_BRDR- 1;
				line[2].y= 0;
				lineGC= xtb_set_gc( SD_Dialog.win, xtb_back_pix, xtb_light_pix, dialogFont.font->fid);
				XSetLineAttributes( disp, lineGC, 2, LineSolid, CapButt, JoinMiter);
				XDrawLines(disp, SD_Dialog.win, lineGC, line, 3, CoordModeOrigin);

				line[0].x= line[1].x= line[1].y= line[2].y= 1;
				line[0].y= (short) SD_Dialog.height- 2* D_BRDR- 1;
				line[2].x= (short) SD_Dialog.width- 2* D_BRDR- 1;
				lineGC= xtb_set_gc( SD_Dialog.win, xtb_norm_pix, xtb_light_pix, dialogFont.font->fid);
				XSetLineAttributes( disp, lineGC, 2, LineSolid, CapButt, JoinMiter);
				XDrawLines(disp, SD_Dialog.win, lineGC, line, 3, CoordModeOrigin);

				if( evt->xexpose.send_event && evt->type== Expose ){
				  int i;
				  Window wi= evt->xexpose.window;
				  int w= evt->xexpose.width, h= evt->xexpose.height;
					if( debugFlag ){
						fprintf( StdErr,
							"Handle_SD_Event(): resending event (#%ld, %s, s_e=%d) for window %ld\n",
							evt->xany.serial, event_name(evt->type), evt->xany.send_event, evt->xany.window
						);
					}
					for( i= 0; i< LAST_F && theWin_Info; i++ ){
						evt->xexpose.window= AF(i).win;
						evt->xexpose.width= AF(i).width;
						evt->xexpose.height= AF(i).height;
						XSendEvent( disp, evt->xexpose.window, True, ExposureMask, evt );
						if( SD_Dialog.win && SD_Dialog.framelist ){
							xtb_dispatch(disp, SD_Dialog.win, SD_Dialog.frames, *SD_Dialog.framelist, evt);
						}
					}
					evt->xexpose.window= wi;
					evt->xexpose.width= w;
					evt->xexpose.height= h;
				}

				*handled= 1;
			}
			break;
		case ButtonPress:
			if( evt->xany.window== SD_Dialog.win && CheckMask(evt->xbutton.state,ControlMask) ){
				xtb_error_box( theWin_Info->window,
					"Customise the appearance of the graph to your liking\n"
					"Control-Click on the fields to pop-up a short description\n",
					"Settings Dialog"
				);
/* 				*handled= 1;	*/
			}
			break;
		case ButtonRelease:
			if( evt->xany.window== SD_Dialog.win ){
				XSetInputFocus( disp, PointerRoot, RevertToParent, CurrentTime);
				*handled= 1;
			}
			break;
		case KeyPress:{
		  xtb_frame *frame= xtb_lookup_frame(evt->xany.window);
			*handled= 0;
			nbytes = XLookupString(&evt->xkey, keys, 4,
						   (KeySym *) 0, (XComposeStatus *) 0
					);
			keysymbuffer[0]= XLookupKeysym( (XKeyPressedEvent*) &evt->xkey, 0);
			if( evt->xany.window== SD_Dialog.win || (frame && frame->parent== SD_Dialog.win) ){
				sprintf( ps_comment, "%s\n%% Handle_SD_Event(): %s; mask=%s; keys[%d]",
					ps_comment, XKeysymToString(keysymbuffer[0]), xtb_modifiers_string(evt->xbutton.state),
					nbytes
				);
				if( keysymbuffer[0]!= NoSymbol ){
					for( keysyms= 1; keysyms< nbytes && keysyms< 4; keysyms++){
						keysymbuffer[keysyms]= XLookupKeysym( (XKeyPressedEvent*) &evt->xkey, 0);
					}

					if( keysymbuffer[0]== XK_Return ){
						if( !CheckMask( xtb_modifier_state, ControlMask|Mod1Mask) ){
							ok_fun( okbtn.win, 0, ok_info);
							*handled= 1;
							return(1);
						}
					}
					else if( keysymbuffer[0]== XK_KP_Enter && HO_okbtn.win ){
					  void *HO_ok_info;
						if( !CheckMask( xtb_modifier_state, ControlMask|Mod1Mask) ){
							xtb_bt_get(HO_okbtn.win, &HO_ok_info);
							HO_ok_fun( HO_okbtn.win, 0, HO_ok_info );
							*handled= 1;
						}
					}
					else if( keysymbuffer[0]== 'a' ){
						if( CheckExclMask( xtb_modifier_state, Mod1Mask) ){
						  int aa= xtb_bt_get( AF(SNATA_F).win, NULL);
							apply_to_all= aa;
							SD_sdds_sl_fun( AF(SNATA_F).win, aa, (xtb_data) &apply_to_all );
							*handled= 1;
						}
					}
					else if( keysymbuffer[0]== 'm' ){
						if( CheckExclMask( xtb_modifier_state, Mod1Mask) ){
						  int aa= xtb_bt_get( AF(SNATM_F).win, NULL);
							apply_to_marked= aa;
							SD_sdds_sl_fun( AF(SNATM_F).win, aa, (xtb_data) &apply_to_marked );
							*handled= 1;
						}
					}
					else if( keysymbuffer[0]== 'h' ){
						if( CheckExclMask( xtb_modifier_state, Mod1Mask) ){
						  int aa= xtb_bt_get( AF(SNATH_F).win, NULL);
							apply_to_hlt= aa;
							SD_sdds_sl_fun( AF(SNATH_F).win, aa, (xtb_data) &apply_to_hlt );
							*handled= 1;
						}
					}
					else if( keysymbuffer[0]== 'n' ){
						if( CheckExclMask( xtb_modifier_state, Mod1Mask) ){
						  int aa= xtb_bt_get( AF(SNATN_F).win, NULL);
							apply_to_new= aa;
							SD_sdds_sl_fun( AF(SNATN_F).win, aa, (xtb_data) &apply_to_new );
							*handled= 1;
						}
					}
					else if( keysymbuffer[0]== 's' ){
						if( CheckExclMask( xtb_modifier_state, Mod1Mask) ){
						  int aa= xtb_bt_get( AF(SNATS_F).win, NULL);
							apply_to_src= aa;
							SD_sdds_sl_fun( AF(SNATS_F).win, aa, (xtb_data) &apply_to_src );
							*handled= 1;
						}
					}
					else if( keysymbuffer[0] == '4' && CheckExclMask(xtb_modifier_state, ShiftMask|ControlMask) ){
						display_ascanf_statbins( theWin_Info );
						*handled= 1;
					}
					else if( evt->xany.window== SD_Dialog.win && (keysymbuffer[0]== XK_Right || keysymbuffer[0]== XK_Left ||
						keysymbuffer[0]== XK_Up || keysymbuffer[0]== XK_Down)
					){
						evt->xkey.window= theWin_Info->window;
						XSendEvent( disp, theWin_Info->window, False, KeyPressMask, evt );
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
							SD_cycle_focus_button( keysymbuffer[0] );
							*handled= 1;
						}
					}
				}
			}
			else{
				nbytes= 0;
			}
			if( nbytes && !*handled ){
				  /* when *handled is set to 1 here, *handled keystrokes are
				   \ not processed further. This interferes a.o. with a
				   \ proper functioning of the TAB key (\t) which has
				   \ functions in several callback functions
				   */
				if( keys[0]== 0x04 ){
					ok_fun( okbtn.win, 0, ok_info);
					*handled= 1;
					return(1);
				}
				else if( keys[0]== 0x10 /* && evt->xany.window== SD_Dialog.win */ ){
				  /* ^P	*/
					print_it= 1;
					*handled= 0;
					return(1);
				}
				else if( keys[0]== 0x01 ){
				  /* ^A Draw all sets	*/
					sdds_rds_fun( AF(SNDAS_F).win, 1, rinfo);
					*handled= 1; /* ?	*/
				}
				else if( keys[0]== 0x13 ){
				  /* ^S Swap drawn/not drawn sets and redraw	*/
					sdds_rds_fun( AF(SNRDS_F).win, 1, rinfo);
					*handled= 1; /* ?	*/
				}
				else if( keys[0]== 0x12 ){
				  /* ^R	*/
					SD_redraw_fun( redrawbtn.win, 1, rinfo);
					*handled= 1;
				}
				else if( keys[0]== 0x1e ){
				  /* ^^	*/
					SD_update( AF(UPDATE_F).win, 1, rinfo);
					*handled= 1;
				}
				else if( keys[0]== 't' ){
					if( CheckMask( xtb_modifier_state, ShiftMask|Mod1Mask) ){
						SD_process_hist( AF(PHIST_F).win, 1, rinfo);
						*handled= 1;
					}
				}
				else if( keys[0]== '?' ){
					if( CheckMask( xtb_modifier_state, Mod1Mask) ){
						SD_help_fnc( AF(HELP_F).win, 1, rinfo);
						*handled= 1;
					}
				}
				else if( keys[0]== '*' && CheckMask( xtb_modifier_state, Mod1Mask ) ){
					SD_selectfun( AF(SNNRM_F).win, 1, rinfo);
				}
				else if( keys[0]== 0x1f ){
				  /* ^_ == ^/	*/
					display_ascanf_variables_h( AF(VARIABLES_F).win, 1, rinfo);
					*handled= 1;
				}
				else if( keys[0]== '\t' ){
				  Window w= get_text_box(1)->win;
				  int dummy;
					XGetInputFocus( disp, &w, &dummy );
					goto_next_text_box( &w );
				}
				else if( evt->xany.window== SD_Dialog.win && keys[0]== ' ' ){
					theWin_Info->halt= 1;
					*handled= 1;
				}
			}
			break;
		}
	}
	if( !*handled && theWin_Info ){
		if( SD_Dialog.win && SD_Dialog.framelist ){
			*xtb_return= xtb_dispatch(disp,SD_Dialog.win, SD_Dialog.frames, *SD_Dialog.framelist, evt);
		}
		if( SD_Dialog.destroy_it ){
			_CloseSD_Dialog( &SD_Dialog, True );
			return(0);
		}
		  /* All events are now handled from within _Handle_An_Event(). There is
		   \ therefore no longer need to call other handlers from this place: <handle_others>
		   \ should therefore be False always. This code will disappear in the near future,
		   \ when the new handling has proved to be bug-free.
		   */
		if( handle_others ){
			if( *xtb_return!= XTB_HANDLED && *xtb_return!= XTB_STOP && HO_Dialog.mapped> 0 ){
				Handle_HO_Event( evt, handled, xtb_return, level, 0 );
				if( Exitting ){
					return(1);
				}
			}
			if( *xtb_return!= XTB_HANDLED && *xtb_return!= XTB_STOP ){
			  int redrawn= theWin_Info->redrawn;
				_Handle_An_Event( evt, *level, 0, "settings_dialog" );
				if( Exitting ){
					return(1);
				}
				if( redrawn!= theWin_Info->redrawn ){
				  /* This is intended to prevent countless spurious calls
				   \ to set_changeables() that make it impossible to enter
				   \ any new value.
				   */
					set_changeables(2,True);
				}
			}
			else if( !XEventsQueued( disp, QueuedAfterFlush) ){
				set_changeables(1,True);
			}
		}
		else if( !XEventsQueued( disp, QueuedAfterFlush) ){
			set_changeables(1,True);
		}
	}
	else if( !XEventsQueued( disp, QueuedAfterFlush) ){
		set_changeables(1,True);
	}
	if( do_update_SD_size ){
		if( !update_SD_size() ){
			set_changeables(2,True);
		}
		else{
			set_changeables(2,True);
		}
	}
	if( theWin_Info ){
		if( theWin_Info->window && rd!= theWin_Info->redraw ){
			xtb_bt_set( redrawbtn.win, theWin_Info->redraw, NULL);
		}
	}
	set_HO_printit_win();

	return(0);
}

void CloseSD_Dialog( xtb_frame *dial )
{
	_CloseSD_Dialog( dial, False );
}

void _CloseSD_Dialog( xtb_frame *dial, Boolean delete )
{ XEvent evt;
  extern Window theSettingsWindow;
  extern LocalWin *theSettingsWin_Info;
  LocalWin *win_info= theWin_Info;
	if( disp && dial ){
	  int i;
		if( dial->destroy_it ){
			dial->destroy_it= 0;
			delete= True;
		}
		set_changeables(1,True);
		  /* Unset the global variables recording to which window the dialog
		   \ belongs.
		   */
		theSettingsWindow= 0;
		if( theSettingsWin_Info ){
			theSettingsWin_Info->SD_Dialog= NULL;
			theSettingsWin_Info= NULL;
		}
		if( delete ){
		  xtb_data info;
			for( i= 0; i< LAST_F; i++ ){
				if( sd_af[i].destroy ){
					(*(sd_af[i].destroy))( sd_af[i].win, &info );
					sd_af[i].win= 0;
					xfree( sd_af[i].framelist );
				}
			}
			xtb_unregister( dial->win, NULL );
			XDestroyWindow( disp, dial->win );
			dial->win= 0;
			dial->frames= 0;
			dial->framelist= NULL;
			xfree( data_legend_buf );
			dlb_len= 0;
			SD_LMaxBufSize= 0;
			xfree( SD_info );
		}
		else{
			XUnmapWindow( disp, dial->win );
		}
		while( XEventsQueued( disp, QueuedAfterFlush)> 0){
			XNextEvent( disp, &evt );
		}
		dial->mapped= 0;
		if( !Exitting && theWin_Info ){
			theWin_Info->SD_Dialog= NULL;
			if( theWin_Info->redraw ){
				theWin_Info->halt= 0;
				theWin_Info->draw_count= 0;
				RedrawNow( theWin_Info );
			}
		}
		  /* Unset the local LocalWin pointer	*/
		theWin_Info= NULL;
		xtb_XSync( disp, False);
		if( print_it && win_info && win_info->window ){
		  extern int PrintWindow();
			win_info->pw_placing= PW_CENTRE_ON;
			PrintWindow( win_info->window, win_info );
		}
	}
}

int settings_dialog(Window theWindow, LocalWin *win_info, char *prog, xtb_data cookie, char *title, char *in_title)
{ static Window dummy;
	static int level= 0;
	Window parent= win_info->window, root_win, win_win;
    XEvent evt;
    XWindowAttributes winInfo;
    XSizeHints hints;
	int win_x, win_y, mask, reexpose= False;

	if( !setNumber ){
		return(0);
	}

	print_it= 0;

	if( level ){
		xtb_error_box( theWin_Info->window, "Settings dialog already active\n", "Error" );
		return(0);
	}
	level++;

	theWin_Info= win_info;
	data_legend_buf= realloc( data_legend_buf, (1+LMAXBUFSIZE)* sizeof(char));
	dlb_len= LMAXBUFSIZE;
	win_info->SD_Dialog= &SD_Dialog;
	ascanf_window= win_info->window;

    if (!SD_Dialog.win) {
		make_SD_dialog(RootWindow(disp, screen), parent, prog, cookie,
				&okbtn, &redrawbtn, &SD_Dialog, title, in_title
		);
		(void) xtb_bt_get(okbtn.win, (xtb_data *) &ok_info);
		SD_ULabels= setNumber+ theWin_Info->ulabels;
    }
	else {
	  /* Change the button information */
		if( !title)
			XStoreName(disp, SD_Dialog.win, "Settings Dialog");
		else
			XStoreName(disp, SD_Dialog.win, title);
		xtb_to_set( AF(TITLE_F).win, in_title);
		(void) xtb_bt_get(okbtn.win, (xtb_data *) &ok_info);
		(void) xtb_bt_get(redrawbtn.win, (xtb_data *) &rinfo);
		xtb_bt_set( AF(OWM_F).win, overwrite_marks_Value(), NULL);
		ok_info->prog = prog;
		ok_info->cookie = cookie;
		  /* For gcc: LMAXBUFSIZE expands to a variable, which can be changed at
		   \ runtime. Thus, we need to ensure that our (ti-frame) buffers depending
		   \ on it are uptodate.
		   */
		if( SD_LMaxBufSize!= LMAXBUFSIZE ){
			  /* this will enforce update_SD_size() to take action	*/
			legend_width= 0;
			update_SD_size();
		}
		else{
			format_SD_Dialog( &SD_Dialog, 0 );
		}
		XRaiseWindow( disp, SD_Dialog.win);
		reexpose= True;
    }
	SD_Dialog.parent= parent;

	xtb_bt_get(redrawbtn.win, (xtb_data *) &rinfo);
	set_changeables(2,True);
	snn_fun( SD_info->sn_number, '\0', data_sn_number_buf, NULL);
    XGetWindowAttributes(disp, parent, &winInfo);
	switch( win_info->pw_placing ){
		case PW_PARENT:
		default:
			XTranslateCoordinates( disp, parent, RootWindow(disp, screen),
					  0, 0, &winInfo.x, &winInfo.y, &dummy
			);
			pw_centre_on_X= winInfo.x + winInfo.width/2;
			pw_centre_on_Y= winInfo.y + winInfo.height/2;
			break;
		case PW_MOUSE:
		  /* dialog comes up with filename box under mouse:	*/
			XQueryPointer( disp, SD_Dialog.win, &root_win, &win_win,
				&pw_centre_on_X, &pw_centre_on_Y, &win_x, &win_y, &mask
			);
			break;
		case PW_CENTRE_ON:
			break;
	}
	hints.x = pw_centre_on_X - SD_Dialog.width/2;
	hints.y = pw_centre_on_Y - SD_Dialog.height/2;
	CLIP( hints.x, 0, DisplayWidth(disp, screen) - SD_Dialog.width );
	CLIP( hints.y, 0, DisplayHeight(disp, screen) - SD_Dialog.height );
	XMoveWindow( disp, SD_Dialog.win, (int) hints.x, (int) hints.y);
	hints.flags = USPosition;
    XSetNormalHints(disp, SD_Dialog.win, &hints);
	if( !SD_Dialog.mapped ){
		XMapWindow(disp, SD_Dialog.win);
		SD_Dialog.mapped= 1;
	}
    XRaiseWindow(disp, SD_Dialog.win);
	{ int n= 0;
		  /* RJB: move the pointer so as to activate the first
		   * text_box.
		   */
		XWarpPointer( disp, SD_Dialog.win, get_text_box(0)->win, 0,0,0,0,
			(int) get_text_box(0)->width/2, (int) get_text_box(0)->height/2
		);
		while( XEventsQueued( disp, QueuedAfterFlush)> 0){
// 		  int handled= 0;
// 		  xtb_hret dum;
/* 			Handle_SD_Event( &evt, &handled, &dum, &level, False );	*/
			XNextEvent( disp, &evt );
			n++;
		}
		if( debugFlag ){
			fprintf( StdErr, "settings_dialog(): processed %d initial events\n", n );
		}
	}
	tabs_fun( 0,0,0,0 );
	update_SD_size();
	format_SD_Dialog( &SD_Dialog, 0 );
	set_changeables(0,True);
	if( SD_Dialog.mapped ){
		XMapRaised( disp, SD_Dialog.win );
	}
	if( reexpose ){
	  XEvent evt;
		evt.type= Expose;
		evt.xexpose.display= disp;
		evt.xexpose.x= 0;
		evt.xexpose.y= 0;
		evt.xexpose.width= SD_Dialog.width;
		evt.xexpose.height= SD_Dialog.height;
		evt.xexpose.window= SD_Dialog.win;
		evt.xexpose.count= 0;
		XSendEvent( disp, SD_Dialog.win, 0, ExposureMask, &evt);
	}
	level--;
	return(0);
}

int SimpleEdit( char *text, Sinc *result, char *errbuf )
{ char tnam[64]= "/tmp/XG-SimpleEdt-XXXXXX";
  int fd= mkstemp(tnam);
  int m= 0;
  FILE *fp, *rfp;
	  /* 20040127 another Mac OSX incompatibility: doing an fdopen(fd,"w+"), writing on it,
	   \ editing its content in the selected editor, and then reading it from that same FILE
	   \ pointer just does not work (somehow, fp remains at EOF despite a rewind command). Thus,
	   \ we are obliged to open a specific FILE *after* editing the contents, and *not* duplicating
	   \ fd...
	   */
	if( fd!= -1 && (fp= fdopen(fd,"w")) ){
	  char *editor= getenv( "XG_EDITOR" ), *command= NULL;
		if( text && *text ){
			print_string( fp, "", NULL, "", text );
			fflush(fp);
		}
		if( !editor ){
			editor= "xterm -e vi";
		}
		fclose(fp);
		if( (command= concat2( command, editor, " ", tnam, NULL )) ){
		  char *c;
		  ALLOCA( buf, char, SD_LMaxBufSize, buflen );
			system( command );
			xfree(command);
			XSync(disp, True);
			if( (rfp= fopen(tnam,"r")) ){
			   /* simple way to prevent returning a NULL result: */
				Sputs( "", result );
				while( !feof(rfp) && !ferror(rfp) && (c= fgets( buf, SD_LMaxBufSize, rfp)) ){
					Sputs( buf, result );
				}
				Sflush(result);
				fclose(rfp);
			}
			else{
				  /* Just re-use the generic error message code: user will understand that the reading didn't work out */
				unlink(tnam);
				goto simpleedit_error;
			}
		}
		else{
			if( errbuf ){
				sprintf( errbuf,
					"SimpleEdit() (%s): construct command to edit %s (%s)\n",
					text, 
					tnam, serror()
				);
			}
			else{
				fprintf( StdErr,
					"SimpleEdit() (%s): construct command to edit %s (%s)\n",
					text, 
					tnam, serror()
				);
			}
		}
		unlink(tnam);
	}
	else{
simpleedit_error:;
		if( errbuf ){
			sprintf( errbuf,
				"SimpleEdit() (%s): can't open temp file %s (%s)\n",
				text, 
				tnam, serror()
			);
		}
		else{
			fprintf( StdErr,
				"SimpleEdit() (%s): can't open temp file %s (%s)\n",
				text, 
				tnam, serror()
			);
		}
	}
	return(m);
}

xtb_hret SimpleEdit_h( Window win, int bval, xtb_data info )
{ ALLOCA( errbuf, char, LMAXBUFSIZE, errbuf_len);
  ALLOCA( text, char, xtb_ti_length(xtb_input_dialog_inputfield,0)+ 1, text_len);
  Sinc sinc;
	Sinc_string_behaviour( &sinc, NULL, 0,0, SString_Dynamic );
	Sflush(&sinc);
	xtb_ti_get( xtb_input_dialog_inputfield, text, (xtb_data *) 0);
	errbuf[0]= '\0';
	SimpleEdit( text, &sinc, errbuf );
	if( sinc.sinc.string ){
		xtb_ti_set( xtb_input_dialog_inputfield, sinc.sinc.string, NULL);
		xfree( sinc.sinc.string );
	}
	if( errbuf[0] ){
		xtb_error_box( win, errbuf, "Message while editing text:" );
	}
	return( XTB_HANDLED );
}

int EditExpression( char *code, char *which, char *errbuf )
{ char tnam[64]= "/tmp/XG-EditEXPR-XXXXXX";
  int fd= mkstemp(tnam);
  int m= 0;
  FILE *fp, *rfp= NULL;
	if( fd!= -1 && (fp= fdopen(fd,"w")) ){
	  char *editor= getenv( "XG_EDITOR" ), *command= NULL;
		if( strncmp( which, "**", 2)== 0 ){
			fprintf( fp, "%s\n", which );
		}
		else{
			fprintf( fp, "%s\\n\n", which );
		}
		if( strncmp( &(which[strlen(which)-2]), "**", 2 )== 0 ){
		  char *endtag= XGstrdup(which);
			if( endtag ){
				endtag[1]= '!';
				endtag[strlen(endtag)-1]= '\0';
				memmove( &endtag[1], endtag, strlen(which)-1 );
				endtag[0]= '\n';
				if( code && *code ){
					print_string( fp, "", NULL, endtag, code );
				}
				else{
					fprintf( fp, "%s\n", endtag );
				}
				xfree(endtag);
			}
		}
		else if( code && *code ){
			print_string( fp, "", NULL, "", code );
		}
		fputs( "\n\n", fp );
		if( !editor ){
			editor= "xterm -e vi";
		}
		fclose(fp);
		if( (command= concat2( command, editor, " ", tnam, NULL )) ){
		 extern int ascanf_propagate_events;
		 int ape= ascanf_propagate_events;
			system( command );
			xfree(command);
			XSync(disp, True);
			ascanf_propagate_events= False;
			  /* 20040127: IncludeFile now opens a NULL filedescriptor */
			m= IncludeFile( theWin_Info, rfp, tnam, True, NULL );
			snn_fun( SD_info->sn_number, 0, data_sn_number_buf, NULL);
			ascanf_propagate_events= ape;
		}
		else{
			if( errbuf ){
				sprintf( errbuf,
					"EditExpression() (%s): construct command to edit %s (%s)\n",
					which, 
					tnam, serror()
				);
			}
			else{
				fprintf( StdErr,
					"EditExpression() (%s): construct command to edit %s (%s)\n",
					which, 
					tnam, serror()
				);
			}
		}
		unlink(tnam);
	}
	else{
		if( errbuf ){
			sprintf( errbuf,
				"EditExpression() (%s): can't open temp file %s (%s)\n",
				which, 
				tnam, serror()
			);
		}
		else{
			fprintf( StdErr,
				"EditExpression() (%s): can't open temp file %s (%s)\n",
				which, 
				tnam, serror()
			);
		}
	}
	return(m);
}

