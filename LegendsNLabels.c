/* 
vim:ts=4:sw=4:
 * xgraph - A Simple Plotter for X
 *
 * David Harrison
 * University of California,  Berkeley
 * 1986, 1987, 1988, 1989
 *
 * Please see copyright.h concerning the formal reproduction rights
 * of this software.

 \ 20031001
 \ 2nd incarnation of LegendsNLabels.c: here be legends and ULabels.
 */

#include "config.h"
IDENTIFY( "Here be legends and ULlabels" );

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <unistd.h>
#include <sys/param.h>
#include <math.h>
#include <string.h>
#ifndef _APOLLO_SOURCE
#	include <strings.h>
#endif
#ifdef _AUX_SOURCE
	extern int strncasecmp();
#endif

#ifdef _AUX_SOURCE
#	include <sys/types.h>
#endif

#include <signal.h>

#include <pwd.h>
#include <ctype.h>
#include "xgout.h"
#include "xgraph.h"
#include "xtb/xtb.h"
#include "hard_devices.h"
#include <X11/Xutil.h>
#include <X11/keysym.h>
#include <X11/Xatom.h>

#include "new_ps.h"
extern char ps_comment[1024];

#include <setjmp.h>

#define ZOOM
#define TOOLBOX

#ifndef MAXFLOAT
#define MAXFLOAT	HUGE
#endif

#define BIGINT		0xfffffff

#include "NaN.h"

#define GRIDPOWER 	10

#define CONTROL_D	'\004'
#define CONTROL_C	'\003'
#define TILDE		'~'

#define BTNPAD		1
#define BTNINTER	3

#ifndef MAX
#	define MAX(a,b)	((a) > (b) ? (a) : (b))
#endif
#ifndef MIN
#	define MIN(a,b)	((a) < (b) ? (a) : (b))
#endif
  /* return b if b<a and b>0, else return a	*/
#define MINPOS(a,b)	(((b)<=0 || (a) < (b))? (a) : (b))
#define MAXNEG(a,b)	(((b)<=0 && (b) > (a))? (b) : (a))
#define ABS(x)		(((x)<0)?-(x):(x))
#define SIGN(x)		(((x)<0)?-1:(x>0)?1:0)
#define ZERO_THRES	-18.0	/* used to be: (1.0e-7); gcc generated wrong float constant in DrawGridAndAxis ** RJB **	*/
#ifndef _AUX_SOURCE
/* #	define zero_thres 1e-18	*/
#	define zero_thres DBL_MIN
#endif

#include <float.h>
#include "ascanf.h"
#include "XXseg.h"
#include "Elapsed.h"
#include "XG_noCursor"
#include <errno.h>
#include <sys/stat.h>
#include <X11/cursorfont.h>

#include "fdecl.h"
#include "copyright.h"

#define nsqrt(x)	(x<0.0 ? 0.0 : sqrt(x))
#define sqr(x)		((x)*(x))

#define ISCOLOR		(wi->dev_info.dev_flags & D_COLOR)

#define DEFAULT_PIXVALUE(set) 	((set) % MAXATTR)
#define DEFAULT_LINESTYLE(set) ((set)%MAXATTR)
#define MAX_LINESTYLE			MAXATTR /* ((bwFlag)?(MAXATTR-1):(MAXSETS/MAXATTR)-1) 	*/
#define LINESTYLE(set)	(AllSets[set].linestyle)
#define ELINESTYLE(set)	((AllSets[set].elinestyle>=0)?AllSets[set].elinestyle:LINESTYLE(set))
#define LINEWIDTH(set)	fabs(AllSets[set].lineWidth)
#define ELINEWIDTH(set)	((AllSets[set].elineWidth>=0)?AllSets[set].elineWidth:LINEWIDTH(set))
#define DEFAULT_MARKSTYLE(set) ((set)+1)
#define PIXVALUE(set)	AllSets[set].pixvalue, AllSets[set].pixelValue
#define MARKSTYLE(set)	(ABS(AllSets[set].markstyle)-1)

#define COLMARK(set) ((set) / MAXATTR)

#define BWMARK(set) \
((set) % MAXATTR)

#define NORMSIZEX	600
#define NORMSIZEY	400
#define NORMASP		(((double)NORMSIZEX)/((double)NORMSIZEY))
#define MINDIM		100

extern int XGTextWidth();
extern void ExitProgramme(), Restart(), Restart_handler(), Dump_handler();

extern void init_X();
#ifdef TOOLBOX
extern void do_error();
#endif

extern double psm_base, psm_incr, psdash_power;
extern int psm_changed, psMarkers, internal_psMarkers;
extern double Xdpi;

extern char *xgraph_NameBuf;
extern int xgraph_NameBufLen;

extern double overlap_legend_tune, highlight_par[];
extern int highlight_mode, highlight_npars;

extern RGB *xg_IntRGB;

extern int use_X11Font_length, _use_gsTextWidth, used_gsTextWidth, prev_used_gsTextWidth;
extern double *ascanf_Counter, *ascanf_counter, *ascanf_numPoints, *ascanf_setNumber, *ascanf_TotalSets;

extern double TRANSFORMED_x, TRANSFORMED_y, TRANSFORMED_ldy, TRANSFORMED_hdy;

#define __DLINE__	(double)__LINE__

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

#define NUMSETS	setNumber

static UserLabel *Target_UL= NULL;

char *ULabelTypeNames[UL_types+1]= { "RL", "HL", "VL", "<invalid>" };

ULabelTypes Parse_ULabelType( char ULtype[2] )
{ ULabelTypes type= UL_regular;
	if( *ULtype ){
	  unsigned short mask;
		mask= ( ( toupper(ULtype[0]) ) << 8 ) + ( toupper(ULtype[1]) );
		switch( mask ){
			case 'RL':
			default:
				type= UL_regular;
				break;
			case 'HL':
				type= UL_hline;
				break;
			case 'VL':
				type= UL_vline;
				break;
		}
	}
	return( type );
}

xtb_hret UL_lineWidth_h( Window win, int bval, xtb_data info )
{ char buf[256], *nbuf;
	sprintf( buf, "%g", Target_UL->lineWidth );
	if( (nbuf= 
		xtb_input_dialog_r( win, buf, 16, sizeof(buf)*sizeof(char), 
			"",
			"Enter the desired line width:",
			"", NULL, "", NULL, NULL, NULL )
		)
	){ double x;
		if( sscanf( nbuf, "%lf", &x )== 1 ){
			Target_UL->lineWidth= x;
		}
		if( nbuf!= buf ){
			xfree( nbuf );
		}
	}
	return( XTB_HANDLED );
}

static char *UL_Type_menu[]= {
		" Select one of the following label types (click on the first line of the type description):\n",
		"   RL: a regular/classical label, with text and possibly an \"arrow\"\n",
		"   HL: a simple horizontal line\n",
		"   VL: a simple vertical line\n"
};

xtb_hret UL_GetType_h( Window win, int bval, xtb_data info )
{
#if 0
	 char buf[256], *nbuf;
	strcpy( buf, ULabelTypeNames[Target_UL->type] );
	if( (nbuf= 
		xtb_input_dialog_r( win, buf, 16, sizeof(buf)*sizeof(char), 
			" RL for a regular label,\n HL for a simple horizontal line,\n VL for a simple vertical line\n"
				" (currently only the 1st letter is used)\n",
			"Enter a style:",
			"", NULL, "", NULL, NULL, NULL )
		)
	){
		switch( tolower( nbuf[0] ) ){
			case 'r':
			default:
				Target_UL->type= UL_regular;
				break;
			case 'h':
				Target_UL->type= UL_hline;
				break;
			case 'v':
				Target_UL->type= UL_vline;
				break;
		}
		if( nbuf!= buf ){
			xfree( nbuf );
		}
	}
#else
	{ char *sel= NULL, **vector_menu;
	  xtb_frame *menu= NULL;
	  int id, N= 0;
		vector_menu= xtb_CharArray( &N, False, sizeof(UL_Type_menu)/sizeof(char*), UL_Type_menu, 0 );
		switch( Target_UL->type ){
			case UL_regular:
			default:
				vector_menu[1][1]= '>';
				break;
			case UL_hline:
				vector_menu[2][1]= '>';
				break;
			case UL_vline:
				vector_menu[3][1]= '>';
				break;
		}
		id= xtb_popup_menu( win, vector_menu[0], "Select the desired type of the new label:", &sel, &menu);
		if( sel ){
		  int set;
			for( set= N-1; set> 0; set-- ){
				if( strstr( vector_menu[set], sel) ){
					switch( set ){
						case 3:
							Target_UL->type= UL_vline;
							break;
						case 2:
							Target_UL->type= UL_hline;
							break;
						case 1:
						default:
							Target_UL->type= UL_regular;
							break;
					}
					set= -1;
				}
			}
			if( set>= 0 ){
				Boing(100);
			}
		}
		xfree( sel );
		xtb_popup_delete( &menu );
		xfree( vector_menu[0] );
		xfree( vector_menu );
	}
#endif
	return( XTB_HANDLED );
}

UserLabel *Add_UserLabel( LocalWin *wi, char *labeltext, double x1, double y1, double x2, double y2,
	int point_label, DataSet *point_label_set, int point_l_nr, double point_l_x, double point_l_y,
	ULabelTypes type,
	int allow_name_trans, unsigned int mask_rtn_pressed, unsigned int mask_rtn_released,
	int noDialog
)
{ UserLabel *new= NULL;
	if( (new= (UserLabel*) calloc( sizeof(UserLabel), 1)) ){
		new->x1= x1;
		new->y1= y1;
		new->x2= x2;
		new->y2= y2;
		if( CheckMask( mask_rtn_released, ControlMask) ){
		  double _point_l_x= x1, _point_l_y= y1;
		  DataSet *_label_set;
		  int _point_l_nr;
			if( (_point_l_nr=
					Find_Point( wi, &_point_l_x, &_point_l_y, &_label_set, 0, NULL, True, True, True, True )
				)>= 0
			){
				new->set_link= _label_set->set_nr;
			}
			else{
				new->set_link= -1;
			}
		}
		else if( wi->add_label== -1 ){
		  Boolean all_marked, all_hlt, none_marked, none_hlt;
			check_marked_hlt( wi, &all_marked, &all_hlt, &none_marked, &none_hlt);
			  /* for the label definition, we make a little exception - all_.. and none_..
			   \ cannot be defined at the same time. When checking for drawing an existing
			   \ label, this is acceptable - like that both all_.. and none_.. labels will
			   \ be drawn when all sets are drawn.
			   */
			if( all_marked ){
				none_marked= False;
			}
			if( all_hlt ){
				none_hlt= False;
			}
			if( all_marked ){
				new->set_link= -2;
			}
			else if( all_hlt ){
				new->set_link= -3;
			}
			else if( none_marked ){
				new->set_link= -4;
			}
			else if( none_hlt ){
				new->set_link= -5;
			}
			else{
				new->set_link= -1;
			}
		}
		else{
			new->set_link= -1;
		}
		new->do_transform= (allow_name_trans ||
				!(wi->transform.x_len || wi->transform.y_len || wi->process.data_process_len)
		);
		new->rt_added= True;
		new->pixvalue= 0;
		new->pixelCName= NULL;
		new->next= NULL;
		new->label[0]= '\0';
		new->pixlinked= 0;
		new->pnt_nr= -1;
		new->type= type;
		new->lineWidth= (type== UL_vline || type== UL_hline)? 0 : axisWidth;
		  /* 20040907: */
		if( point_label_set ){
			new->set_link= point_label_set->set_nr;
		}
		if( point_label ){
			if( point_l_nr>= 0 && CheckMask( mask_rtn_pressed, Button3Mask)){
				if( CheckMask( mask_rtn_released, ControlMask) ){
					new->x1= point_l_x;
					new->y1= point_l_y;
					new->label[0]= '\0';
					update_LinkedLabel( wi, new, point_label_set, point_l_nr,
						CheckMask(mask_rtn_released, ShiftMask)
					);
					if( SCREENX( wi, Trans_X(wi, new->x1)) == SCREENX( wi, Trans_X(wi, new->x2)) ){
						new->x2= new->x1;
					}
					if( SCREENY( wi, Trans_Y(wi, new->y1, 0)) == SCREENY( wi, Trans_Y(wi, new->y2, 0)) ){
						new->y2= new->y1;
					}
					new->do_draw= 1;
#if 0
					new->next= wi->ulabel;
					wi->ulabel= new;
#else
					new->next= NULL;
					wi->ulabel= Install_ULabels( wi->ulabel, new, False );
#endif
					wi->ulabels+= 1;
					wi->redraw= 1;
				}
			}
		}
		else{
		  char *nbuf;
		  char pc[]= "#x01Request";
		  char header[512];
			wi->add_label= 0;
			xtb_bt_set( wi->label_frame.win, wi->add_label, NULL);

			if( labeltext ){
				strncpy( new->label, labeltext, sizeof(new->label)-1 );
			}
			else{
				sprintf( new->label, "Label #%d", wi->ulabels );
			}

			snprintf( header, sizeof(header), "Enter text for label\n %s; (%g,%g) <- (%g,%g)%s:",
				ULabelTypeNames[type],
				x1, y1, x2, y2,
				(point_label_set)? d2str(point_label_set->set_nr, " linked to set %g", NULL) : ""
			);
			Target_UL= new;
			if( noDialog ){
				nbuf= new->label;
			}
			else{
				nbuf= xtb_input_dialog( wi->window, new->label, 90, sizeof(new->label)/sizeof(char), 
					   header, parse_codes(pc),
					   False,
					   "lWidth", UL_lineWidth_h, "Type", UL_GetType_h, "Edit", SimpleEdit_h
			   );
			}
			if( nbuf ){
				new->do_draw= 1;
				switch( new->type ){
					case UL_hline:
/* 						set_NaN( new->x1 );	*/
						set_NaN( new->x2 );
						set_NaN( new->y2 );
						break;
					case UL_vline:
/* 						set_NaN( new->y1 );	*/
						set_NaN( new->x2 );
						set_NaN( new->y2 );
						break;
				}
#if 0
				new->next= wi->ulabel;
				wi->ulabel= new;
#else
				new->next= NULL;
				wi->ulabel= Install_ULabels( wi->ulabel, new, False );
#endif
				wi->ulabels+= 1;
				wi->redraw= 1;
				if( nbuf!= new->label ){
					xfree( nbuf );
				}
			}
			else{
				xtb_error_box( wi->window,
					"Can't create new label\nOperation Cancelled\nor Text-entry window wouldn't open\n",
					"Warning"
				);
				xfree( new);
			}
			Target_UL= NULL;
		}
		if( new ){
			if( new->x1== new->x2 && new->y1== new->y2 ){
				new->nobox= 1;
			}
			else{
				new->nobox= 0;
			}
		}
		wi->printed= 0;
	}
	else{
		xtb_error_box( wi->window, "Can't create new label", "Warning" );
	}
	return(new);
}

/* Install a list of UserLabels <src> "onto" the destination list, which may be empty.
 \ This consists of finding the end of a non-empty destination list, and then copying all source
 \ labels into newly allocated labels that are appended to the list in a way that is hardly
 \ more expensive than a simple prepending ('cons') operation.
 */
UserLabel *Install_ULabels( UserLabel *dst, UserLabel *src, int copy )
{ UserLabel *new= dst, *old= src;
	if( old && new ){
		  /* First, find the end of an existing destination label-list: */
		while( new->next ){
			new= new->next;
		}
	}
	if( copy ){
		while( old ){
			if( !new ){
				if( (dst= (UserLabel*) calloc( sizeof(UserLabel), 1)) ){
					new= dst;
					*new= *old;
					if( old->pixelCName ){
						new->pixelCName= XGstrdup(old->pixelCName);
					}
					new->old2= NULL;
				}
			}
			else{
				if( (new->next= (UserLabel*) calloc( sizeof(UserLabel), 1)) ){
					new= new->next;
					*new= *old;
					if( old->pixelCName ){
						new->pixelCName= XGstrdup(old->pixelCName);
					}
					new->old2= NULL;
				}
			}
			old= old->next;
		}
	}
	else if( !new ){
		dst= src;
	}
	else{
		new->next= src;
	}
	return( dst );
}

int Copy_ULabels( LocalWin *dst, LocalWin *src )
{
	dst->ulabel= Install_ULabels( dst->ulabel, src->ulabel, True );
	return( (dst->ulabels= src->ulabels) );
}

int Delete_ULabels( LocalWin *wi )
{ UserLabel *ul, *nul;
  int ulabels= 0;
	if( wi && (ul= wi->ulabel) ){
		ulabels= wi->ulabels;
		while( ul ){
			if( ul->free_buf ){
				xfree( ul->labelbuf);
			}
			nul= ul->next;
			xfree( ul );
			nul= ul;
		}
		wi->ulabel= NULL;
		wi->ulabels= 0;
	}
	return(ulabels);
}

/* Returns the colours in which <ul> should be drawn. Care should
 \ be taken to have the unmodified value in AllAttrs[0].pixelValue!
 */
Pixel ULabel_pixelValue( UserLabel *ul, Pixel *txtPixel )
{
	if( ul->pixvalue< 0 ){
		if( txtPixel ){
			*txtPixel= ul->pixelValue;
		}
		return( ul->pixelValue );
	}
	else{
		if( ul->set_link>= 0 && ul->set_link< setNumber && ul->pixlinked ){
		  DataSet *lset= &AllSets[ul->set_link];
			if( lset ){
				if( lset->pixvalue< 0 ){
					if( txtPixel ){
						*txtPixel= lset->pixelValue;
					}
					return( lset->pixelValue );
				}
				else{
					if( txtPixel ){
						*txtPixel= AllAttrs[lset->pixvalue].pixelValue;
					}
					return( AllAttrs[lset->pixvalue].pixelValue );
				}
			}
		}
		if( txtPixel ){
			*txtPixel= AllAttrs[0].pixelValue;
		}
		return( zeroPixel );
	}
}

char *ULabel_pixelCName( UserLabel *ul, int *type )
{
	if( ul->pixvalue< 0 ){
		if( type ){
			*type= 0;
		}
		return( ul->pixelCName );
	}
	else{
		if( ul->set_link>= 0 && ul->set_link< setNumber && ul->pixlinked ){
		  DataSet *lset= &AllSets[ul->set_link];
			if( lset ){
				if( lset->pixvalue< 0 ){
					if( type ){
						*type= -1;
					}
					return( lset->pixelCName );
				}
				else{
					if( type ){
						*type= -1;
					}
					return( AllAttrs[lset->pixvalue].pixelCName );
				}
			}
		}
		if( type ){
			*type= 1;
		}
		return( zeroCName );
	}
}

extern int StringCheck(char *, int, char *, int);
#define STRINGCHECK(s,max)	StringCheck(s,max,__FILE__,__LINE__)

char *ULabel_pixelCName2( UserLabel *ul, int *type )
{ int t= 0;
  char *cname;
  static char buf[256];
	if( !type ){
		type= &t;
	}
	cname= ULabel_pixelCName( ul, type );
	if( *type ){
		if( *type== -1 ){
			sprintf( buf, "linked [%s] (#%d)", cname, ul->set_link );
		}
		else{
			sprintf( buf, "default [%s]", cname );
		}
		STRINGCHECK( buf, sizeof(buf) );
		return( buf );
	}
	else{
		return( cname );
	}
}

int DrawIntensityLegend( LocalWin *wi, Boolean doit )
{ double minIntense, maxIntense, scaleIntense;
  int NC;
	if( !wi || wi->no_intensity_legend ){
		return(0);
	}
	if( wi->IntensityLegend.legend_needed< 0 && doit ){
		if( !wi->SS_I.count ){
			wi->IntensityLegend.legend_needed= 0;
			return(0);
		}
		else{
			wi->IntensityLegend.legend_needed= 1;
		}
	}
	if( wi->IntensityLegend.legend_needed && (NC= IntensityColourFunction.NColours) ){
	  char minVal[MAXAXVAL_LEN], maxVal[MAXAXVAL_LEN], valstr[MAXAXVAL_LEN];
	  int i, x, nx, ptx= -1, px= -1, plen= 0, minVallen, maxVallen, last_x;
	  ValCategory *vcat, *mivcat= NULL, *mavcat= NULL;
	  CustomFont *mivcatfont= NULL, *mavcatfont= NULL;
	  double fontScale, pixScale, all_valScale= 1;
	  LegendInfo *IL= &wi->IntensityLegend;
		if( IntensityColourFunction.range_set ){
			minIntense= IntensityColourFunction.range.min;
			maxIntense= IntensityColourFunction.range.max;
		}
		else{
			minIntense= wi->SS_I.min;
			maxIntense= wi->SS_I.max;
		}
		if( minIntense== maxIntense ){
			scaleIntense= 1;
		}
		else{
			scaleIntense= (maxIntense- minIntense)/ (NC- 1);
		}
		  /* else.. Get the min and max textual representations to determine the legend width	*/
		  /* 20020208: prepend a space to all but the first values to print. This ensures an adequate
		   \ separation.
		   */
		maxVal[0]= ' ';
		if( wi->ValCat_I_axis ){
			if( (mivcat= Find_ValCat( wi->ValCat_I, minIntense, NULL, NULL)) ){
				strncpy( minVal, mivcat->category, sizeof(minVal)-1 );
				mivcatfont= wi->ValCat_IFont;
			}
			else{
			  /* 20000824: pass False to use_real_value, as passing true would "undo" the effect
			   \ of a logXFlag or sqrtXFlag.
			   */
				WriteValue(wi, &minVal[0], minIntense, 0, 0, 0, 0, X_axis, False, scaleIntense, sizeof(minVal));
/* 					fprintf( StdErr, "%s\n", d2str( minIntense, NULL, NULL) );	*/
			}
			if( (mavcat= Find_ValCat( wi->ValCat_I, maxIntense, NULL, NULL)) ){
				strncpy( &maxVal[1], mavcat->category, sizeof(maxVal)-2 );
				mavcatfont= wi->ValCat_IFont;
			}
			else{
				WriteValue(wi, &maxVal[1], maxIntense, 0, 0, 0, 0, X_axis, False, scaleIntense, sizeof(maxVal)-1);
/* 					fprintf( StdErr, "%s\n", d2str( maxIntense, NULL, NULL) );	*/
			}
		}
		else{
			WriteValue(wi, &minVal[0], minIntense, 0, 0, 0, 0, X_axis, False, scaleIntense, sizeof(minVal));
/* 					fprintf( StdErr, "%s\n", d2str( minIntense, NULL, NULL) );	*/
			WriteValue(wi, &maxVal[1], maxIntense, 0, 0, 0, 0, X_axis, False, scaleIntense, sizeof(maxVal)-1);
/* 					fprintf( StdErr, "%s\n", d2str( maxIntense, NULL, NULL) );	*/
		}
		if( !use_X11Font_length ){
			minVallen= strlen(minVal)* wi->dev_info.axis_width;
			maxVallen= strlen(maxVal)* wi->dev_info.axis_width;
		}
		else{
			minVallen= XGTextWidth(wi, minVal, T_AXIS, mivcatfont );
			maxVallen= XGTextWidth(wi, maxVal, T_AXIS, mavcatfont );
		}
		if( mivcat ){
			mivcat->print_len= minVallen;
		}
		if( mavcat ){
			mavcat->print_len= maxVallen;
		}
		  /* Scale these widths for the actually used font	*/
		if( wi->ValCat_IFont && wi->ValCat_I && wi->ValCat_I_axis ){
			if( !_use_gsTextWidth && PS_STATE(wi)->Printing== PS_PRINTING ){
				fontScale= CustomFont_psWidth(wi->ValCat_IFont) / XFontWidth( wi->ValCat_IFont->XFont.font);
			}
			else{
				fontScale= 1;
			}
		}
		else{
			if( !_use_gsTextWidth ){
				fontScale= ((double)wi->dev_info.axis_width) / MAX( XFontWidth(axisFont.font), XFontWidth(axis_greekFont.font) );
			}
			else{
				fontScale= 1;
			}
		}
		minVallen= (int)((minVallen)* fontScale) + wi->dev_info.bdr_pad;
		maxVallen= (int)((maxVallen)* fontScale) + wi->dev_info.bdr_pad;
		if( wi->ValCat_I && wi->ValCat_I_axis ){
		  int totLen= minVallen+ maxVallen, len;
			vcat= wi->ValCat_I;
			while( vcat && vcat->min!= vcat->max ){
				valstr[0]= ' ';
				if( vcat->val!= minIntense && vcat->val!= maxIntense ){
					strncpy( &valstr[1], vcat->category, sizeof(valstr)-2 );
					if( !use_X11Font_length ){
						vcat->print_len= len= strlen(valstr)* wi->dev_info.axis_width;
					}
					else{
						vcat->print_len= len= XGTextWidth(wi, valstr, T_AXIS, wi->ValCat_IFont );
					}
					totLen+= (int)(len* fontScale) + wi->dev_info.bdr_pad;
				}
				vcat++;
			}
			all_valScale= (IL->legend_width= totLen);
		}
		else{
			IL->legend_width= (minVallen+ maxVallen+ wi->dev_info.bdr_pad);
			if( !(wi->ValCat_I && wi->ValCat_I_axis) ){
				IL->legend_width= MAX( (wi->XOppX- wi->XOrgX- 3* wi->dev_info.tick_len)/2, 2* IL->legend_width );
			}
		}
		  /* The legend should have space for at least the low and high intensity printed values,
		   \ (20020208) be at least 1/2 the plotting region's width,
		   \ but it should not be larger than the plotting region.
		   \ 20030930: min.width only when not ValCat_I.
		   */
		IL->legend_width= MIN( wi->XOppX- wi->XOrgX- 3* wi->dev_info.tick_len, IL->legend_width );
		  /* all_valScale: a scale factor that relates the 'obtained' width to the desired width */
		all_valScale= IL->legend_width/ all_valScale;
		pixScale= ((double) IL->legend_width)/ NC;

		IL->legend_lrx= IL->legend_frx= IL->legend_ulx+ IL->legend_width;
		  /* Height will be tick_len for now.	*/
		IL->legend_lry= IL->legend_uly+ wi->dev_info.tick_len;
		vcat= wi->ValCat_I;
		plen= 0;
		nx= x= IL->legend_ulx;
		last_x= IL->legend_lrx- maxVallen;

		  /* 20000209: verify if in spirsin.xg there can be some memory-overwrite between here and the
		   \ previous WriteValue invocation.
		   */

		if( doit ){
			wi->axis_stuff.I.last_index= 0;
		}

		for( i= 0; i< NC && doit; i++ ){
		  double intens= minIntense+ i* scaleIntense;
		  Pixel colour= IntensityColourFunction.XColours[i].pixel;
		  RGB *pscol= NULL;
		  int all_val, len;
		  XSegment line;
		  char *vstr;
			valstr[0]= ' ';
			IntensityColourFunction.last_read= i;
			if( IntensityColourFunction.exactRGB ){
				pscol= &(IntensityColourFunction.exactRGB[i]);
			}
			all_val= 0;
			if( wi->show_all_ValCat_I && wi->ValCat_I && wi->ValCat_I_axis ){
				if( vcat->min!= vcat->max || i== 0 || i== NC-1 ){
					if( i && i< NC-1 ){
						strncpy( &valstr[1], vcat->category, sizeof(valstr)- 2 );
						vstr= valstr;
					}
					x= (int) nx;
					nx= x+ (vcat->print_len* fontScale+ wi->dev_info.bdr_pad)* all_valScale;
					  /* 991209: increment x only when there are (still) categories	*/
					if( vcat->min!= vcat->max ){
						vcat++;
					}
				}
				all_val= 1;
			}
			else if( wi->ValCat_I_axis && (vcat= Find_ValCat( wi->ValCat_I, intens, NULL, NULL)) ){
				x= (int) nx;
				nx= x+ (vcat->print_len* fontScale+ wi->dev_info.bdr_pad)* all_valScale;
				strncpy( &valstr[1], vcat->category, sizeof(valstr)- 2 );
				vstr= valstr;
			}
			else{
				x= (int)( IL->legend_ulx+ i* pixScale);
				nx= x+ pixScale;
				vstr= NULL;
				WriteValue(wi, &valstr[1], intens, 0, 0, 0, 0, X_axis, False, scaleIntense, sizeof(valstr)-1);
/* 						fprintf( StdErr, "%s\n", d2str( valstr, NULL, NULL) );	*/
			}
			if( !i || x> px || i== NC-1 ){
				AddAxisValue( wi, &wi->axis_stuff.I, intens );
				psThisRGB= pscol;
				wi->dev_info.xg_rect( wi->dev_info.user_state,
					rect_diag2xywh( x, IL->legend_uly, nx+1, IL->legend_lry), 0, L_VAR, 0,
					-1, colour, 1, -1, colour, NULL
				);
#ifdef DEBUG
				  /* 20020216: this is more to debug the display/printing device than to debug our code...
				   \ It outputs hairlines at the onset of every rectangle filled with the current intensity
				   \ colour. For even entries, the hairline is drawn "below", for odd "above", theoretically
				   \ avoiding the possibility that the hairlines form a solid band of black.
				   \ WhenThis feature allows to verify whether the device renders the individual, exactly
				   \ specified colours correctly, or whether it reduces the colour resolution.
				   \ This only makes sense if we have access to the exact colours as requested -- and if
				   \ these *are* all unique.
				   */
				if( pscol ){
				  XSegment hl;
					hl.x1= hl.x2= x;
					if( i % 2 ){
						hl.y1= IL->legend_uly;
						hl.y2= hl.y1+ wi->dev_info.bdr_pad/2;
					}
					else{
						hl.y1= IL->legend_lry;
						hl.y2= hl.y1- wi->dev_info.bdr_pad/2;
					}
					wi->dev_info.xg_seg( wi->dev_info.user_state, 1, &hl, 0, L_VAR, 0, -1, black_pixel, NULL);
				}
#endif
				line.x1= line.x2= x;
				line.y1= IL->legend_lry;
				line.y2= line.y1+ wi->dev_info.bdr_pad/2;
				if( i== 0 ){
					wi->dev_info.xg_seg( wi->dev_info.user_state, 1, &line, 1, L_VAR, 0, -1, black_pixel, NULL);
					AxisValueCurrentLabelled( wi, &wi->axis_stuff.I, True );
					wi->dev_info.xg_text( wi->dev_info.user_state,
						x, IL->legend_lry+ wi->dev_info.bdr_pad, minVal,
						T_UPPERLEFT, T_AXIS, mivcatfont
					);
					plen= minVallen;
					ptx= x;
				}
				else if( i== NC-1 ){
					wi->dev_info.xg_seg( wi->dev_info.user_state, 1, &line, 1, L_VAR, 0, -1, black_pixel, NULL);
					if( x+ maxVallen> IL->legend_lrx ){
						x= last_x+ wi->dev_info.bdr_pad;
					}
					AxisValueCurrentLabelled( wi, &wi->axis_stuff.I, True );
					wi->dev_info.xg_text( wi->dev_info.user_state,
						x, IL->legend_lry+ wi->dev_info.bdr_pad, maxVal,
						T_UPPERLEFT, T_AXIS, mavcatfont
					);
					plen= maxVallen;
					ptx= x;
				}
				else{
				  CustomFont *cfnt;
					if( !vstr ){
						vstr= valstr;
						cfnt= NULL;
					}
					else{
						cfnt= wi->ValCat_IFont;
					}
					if( !use_X11Font_length ){
						len= strlen(vstr)* wi->dev_info.axis_width;
					}
					else{
						len= XGTextWidth(wi, vstr, T_AXIS, cfnt );
					}
					len= (int)(len* fontScale) + wi->dev_info.bdr_pad;
					if( all_val ){
						if( x+ len<= last_x ){
						  /* There will be no overlap with the (obligatory) last label.
						   \ (We can use the <= operator since the calculated length includes padding).
						   */
							wi->dev_info.xg_seg( wi->dev_info.user_state, 1, &line, 1, L_VAR, 0, -1, black_pixel, NULL);
							AxisValueCurrentLabelled( wi, &wi->axis_stuff.I, True );
							wi->dev_info.xg_text( wi->dev_info.user_state,
								x, IL->legend_lry+ wi->dev_info.bdr_pad, vstr,
								T_UPPERLEFT, T_AXIS, cfnt
							);
							plen= len;
							ptx= x;
						}
					}
					else if( x+ len/2 <= last_x ){
						if( (ptx== IL->legend_ulx && (x>= ptx+ plen+ len/2)) || (ptx> IL->legend_ulx && x>= ptx+ (plen+ len)/2) ){
						  /* There will be no overlap with the (obligatory) last label.
						   \ (We can use the <= operator since the calculated length includes padding).
						   */
							wi->dev_info.xg_seg( wi->dev_info.user_state, 1, &line, 1, L_VAR, 0, -1, black_pixel, NULL);
							AxisValueCurrentLabelled( wi, &wi->axis_stuff.I, True );
							wi->dev_info.xg_text( wi->dev_info.user_state,
								x, IL->legend_lry+ wi->dev_info.bdr_pad, vstr,
								T_TOP, T_AXIS, cfnt
							);
							plen= len;
							ptx= x;
						}
					}
				}
				px= x;
			}
				if( Handle_An_Event( wi->event_level, 1, "DrawIntensityLegend-" STRING(__LINE__), wi->window, 
						StructureNotifyMask|KeyPressMask|ButtonPressMask
					)
				){
					  /* 20000427: changed to False	*/
					XG_XSync( disp, False );
					return(-1);
				}
				if( wi->delete_it== -1 ){
					return(-1);
				}
				if( wi->redraw ){
					return(-1);
				}
				if( wi->halt ){
					doit= False;
				}

		}
		if( doit ){
			wi->dev_info.xg_rect( wi->dev_info.user_state,
				rect_diag2xywh( IL->legend_ulx- 1.25* wi->dev_info.var_width_factor/2, IL->legend_uly,
					IL->legend_lrx+ 1.25* wi->dev_info.var_width_factor/2, IL->legend_lry), 1.25, L_VAR, 0,
				-1, black_pixel, 0, -1, 0, NULL
			);
#ifdef DEBUG
			wi->dev_info.xg_rect( wi->dev_info.user_state,
				rect_diag2xywh( IL->legend_ulx- wi->dev_info.var_width_factor, IL->legend_uly,
					IL->legend_lrx+ wi->dev_info.var_width_factor, IL->legend_uly+ IL->legend_height), 1, L_VAR, 0,
				-1, black_pixel, 0, -1, 0, NULL
			);
#endif
		}
		return( 1 );
	}
	return(0);
}

int LegendLineWidth(LocalWin *wi, int idx )
{ double lw= 0;
	if( idx>=0 && idx< setNumber ){
		lw= LINEWIDTH( idx);
		if( AllSets[idx].barFlag> 0 ){
			lw= MAX(5, AllSets[idx].lineWidth);
		}
		if( wi->legend_line[idx].highlight || AllSets[idx].barFlag> 0 ){
			lw= HL_WIDTH(lw)+ AllSets[idx].lineWidth;
		}
	}
	return( (int) lw );
}

typedef struct ItemOfLegend{
	int idx, y;
} ItemOfLegend;

void DrawULabels( LocalWin *wi, int pass, int doit, int *prev_silent, void *dimension_data )
{ Boolean all_marked, all_hlt, none_marked, none_hlt;
  int bdr_pad= wi->dev_info.bdr_pad/2, lnr= 0;
	if( wi->no_ulabels ){
		return;
	}
	switch( pass ){
		case 0:
			if( doit ){
			  UserLabel *ul= wi->ulabel;
			  double asn= *ascanf_setNumber, anp= *ascanf_numPoints;
				if( ul ){
					check_marked_hlt( wi, &all_marked, &all_hlt, &none_marked, &none_hlt);
				}
				while( ul ){
					ul->draw_it= 0;
					if( strlen( ul->label) && ul->do_draw &&
						( ul->set_link== -1 ||
							(ul->set_link== -2 && all_marked ) ||
							(ul->set_link== -3 && all_hlt ) ||
							(ul->set_link== -4 && none_marked) ||
							(ul->set_link== -5 && none_hlt) ||
							(ul->set_link>= 0 && draw_set(wi, ul->set_link) )
						)
					){
					  double tx1, tx2, ty1, ty2;
					  int ok1= 1, ok2= 1, no_box
						/* , mark_inside1, mark_inside2, clipcode1, clipcode2	*/	;
					  int trans;
						if( ul->x1!= ul->x2 || ul->y1!= ul->y2 ){
/* 							ul->nobox= 0;	*/
						}
						else{
/* 							ul->nobox= 1;	*/
						}
						parse_codes( ul->label );
						if( ul->free_buf ){
							xfree( ul->labelbuf );
						}
						ul->labelbuf= ul->label;
						ul->free_buf= False;
						if( ul->set_link>= 0 && ul->pnt_nr>= 0 ){
						  DataSet *lset= &AllSets[ul->set_link];
						  int pnr= ul->pnt_nr;
							if( ul->x1!= lset->xvec[pnr] || ul->y1!= lset->yvec[pnr] || ul->eval!= lset->errvec[pnr] ){
								update_LinkedLabel( wi, ul, lset, pnr, ul->short_flag );
							}
						}
						else{
						  char *ntt, *parsed_end= NULL;
							if( (ntt= ParseTitlestringOpcodes( wi, ul->set_link, ul->label, &parsed_end )) ){
								ul->labelbuf= ntt;
								if( ntt[0]== '`' && parsed_end[-1]== '`' ){
									parsed_end[-1]= '\0';
									strncpy( ul->label, &ntt[1], sizeof(ul->label)/sizeof(char)- 1);
									ul->labelbuf= ul->label;
								}
								else{
									ul->free_buf= True;
								}
							}
						}
						tx1= ul->x1, tx2= ul->x2, ty1= ul->y1, ty2= ul->y2;
						trans= (ul->do_transform)? 0 : -1;
						*ascanf_setNumber= ul->set_link;
						*ascanf_numPoints= 2;
						no_box= (tx1== tx2) && (ty1== ty2);
						  /* Make sure that the unconstrained co-ordinate is within the current drawing area: */
						switch( ul->type ){
							case UL_hline:
								tx1= (wi->hiX - wi->loX)/2;
								break;
							case UL_vline:
								ty1= (wi->hiY - wi->loY)/2;
								break;
						}
						do_transform( wi, ul->label, __DLINE__, "DrawULabels(label,1)", &ok1, NULL,
							&tx1, NULL, NULL, &ty1,
							NULL, NULL, NULL, NULL, 1, -1, 1.0, 1.0, 1.0, trans, 0,
							!((wi->raw_display && wi->raw_once>= 0) || (wi->raw_once< 0 && wi->raw_val))
						);
						ul->tx1= TRANSFORMED_x;
						ul->ty1= TRANSFORMED_y;
						if( ul->type== UL_regular ){
							do_transform( wi, ul->label, __DLINE__, "DrawULabels(label,2)", &ok2, NULL,
								&tx2, NULL, NULL, &ty2,
								NULL, NULL, NULL, NULL, 1, -1, 1.0, 1.0, 1.0, trans, 0,
								!((wi->raw_display && wi->raw_once>= 0) || (wi->raw_once< 0 && wi->raw_val))
							);
							ul->tx2= TRANSFORMED_x;
							ul->ty2= TRANSFORMED_y;
						}
						if( ok1 && ok2 ){
						  Pixel color0= AllAttrs[0].pixelValue;
							switch( ul->type ){
								case UL_regular:
								default:
									ul->line.x1= SCREENX(wi, tx1);
									ul->line.y1= SCREENY(wi, ty1);
									ul->line.x2= SCREENX(wi, tx2);
									ul->line.y2= SCREENY(wi, ty2);
									break;
								case UL_hline:
									  /* Just a horizontal line to mark a Y level */
									ul->line.x1= wi->XOrgX;
									ul->line.y2= ul->line.y1= SCREENY(wi, ty1);
									ul->line.x2= wi->XOppX;
									break;
								case UL_vline:
									  /* Just a vertical line to mark an X value */
									ul->line.x2= ul->line.x1= SCREENX(wi, tx1);
									ul->line.y1= wi->XOrgY;
									ul->line.y2= wi->XOppY;
									break;
							}
							AllAttrs[0].pixelValue= ULabel_pixelValue( ul, NULL);
							if( ul->type== UL_regular && !no_box ){
								if( ul->set_link>= 0 && wi->legend_line[ul->set_link].highlight ){
									HighlightSegment( wi, ul->set_link, 1, &ul->line, HL_WIDTH(ul->lineWidth), L_VAR );
								}
								wi->dev_info.xg_seg(wi->dev_info.user_state,
										 1, &ul->line, ul->lineWidth, L_VAR,
										 0, 0, 0, NULL
								);
								wi->dev_info.xg_dot(wi->dev_info.user_state,
											ul->line.x1, ul->line.y1,
											P_DOT, 0, 0, 0, 0, NULL
								);
							}
							AllAttrs[0].pixelValue= color0;
							ul->draw_it= 1;
						}
					}
					ul= ul->next;
				}
				*ascanf_setNumber= asn;
				*ascanf_numPoints= anp;
			}
			else{
				*prev_silent= wi->dev_info.xg_silent( wi->dev_info.user_state, True );
			}
			break;
		case 1:
			if( doit ){
			  UserLabel *ul= wi->ulabel;
			  double asn= *ascanf_setNumber, anp= *ascanf_numPoints;
			  XSegment label_box[5];
				while( ul ){
					if( ul->set_link>= 0 && AllSets[ul->set_link].numPoints<= 0 ){
						ul->label[0]= '\0';
						if( ul->free_buf ){
							ul->free_buf= False;
							xfree( ul->labelbuf );
						}
					}
					if( strlen( ul->label) &&
						( ul->set_link== -1 ||
							(ul->set_link== -2 && all_marked ) ||
							(ul->set_link== -3 && all_hlt ) ||
							(ul->set_link== -4 && none_marked) ||
							(ul->set_link== -5 && none_hlt) ||
							(ul->set_link>= 0 && draw_set(wi, ul->set_link) )
						)
					){
					  double tx1= ul->tx1, ty1= ul->ty1, tx2= ul->tx2, ty2= ul->ty2;
					  int no_box, label_len;
					  int maxFontWidth= XFontWidth(legendFont.font);
						*ascanf_setNumber= ul->set_link;
						*ascanf_numPoints= 2;
						no_box= (tx1== tx2) && (ty1== ty2);
						if( ul->draw_it ){
						  char *name, *line;
						  Pixel color0= AllAttrs[0].pixelValue;
						  int width, height, lines= 0, cx, cy;
							AllAttrs[0].pixelValue= ULabel_pixelValue( ul, &textPixel );
							if( ul->type== UL_regular && !no_box ){
							  /* The "arrow-marker" (a dot) is drawn once more.	*/
								wi->dev_info.xg_dot(wi->dev_info.user_state,
											ul->line.x1, ul->line.y1,
											P_DOT, 0, 0, 0, 0, NULL
								);
							}
							else if( ul->type== UL_hline || ul->type== UL_vline ){
								  /* 20040830: not sure if we want highlighting for this type! */
								if( ul->set_link>= 0 && wi->legend_line[ul->set_link].highlight ){
									HighlightSegment( wi, ul->set_link, 1, &ul->line, HL_WIDTH(axisWidth), L_VAR );
								}
								wi->dev_info.xg_seg(wi->dev_info.user_state,
										 1, &ul->line, ul->lineWidth, L_VAR,
										 0, 0, 0, NULL
								);
							}
							if( ul->type== UL_regular ){
								if( !xgraph_NameBuf || strlen(ul->labelbuf)> xgraph_NameBufLen ){
									xfree( xgraph_NameBuf );
									xgraph_NameBuf= XGstrdup( ul->labelbuf );
									xgraph_NameBufLen= strlen( xgraph_NameBuf );
								}
								else{
									strcpy( xgraph_NameBuf, ul->labelbuf );
								}
								name= xgraph_NameBuf;
								if( xtb_has_greek( ul->labelbuf) ){
									maxFontWidth= MAX(maxFontWidth, XFontWidth(legend_greekFont.font) );
								}
								  /* Find the printing width of the label	*/
								label_len= 0;
								while( name && xtb_getline( &name, &line ) ){
								 int llen;
									lines+= 1;
									if( !use_X11Font_length ){
										llen = strlen( line );
									}
									else{
										llen= XGTextWidth( wi, line, T_LEGEND, NULL );
									}
									label_len= MAX( label_len, llen);
								}
								if( !use_X11Font_length ){
									label_len= label_len* wi->dev_info.legend_width;
								}
								else if( !_use_gsTextWidth ){
									label_len= (int)((label_len)* ((double)wi->dev_info.legend_width/ maxFontWidth ));
								}
								label_len+= wi->dev_info.bdr_pad;
								if( ul->vertical ){
									width= increment_height( wi, ul->labelbuf, '\n');
									height= label_len;
								}
								else{
									width= label_len;
									height= increment_height( wi, ul->labelbuf, '\n');
								}
								  /* Create a box to write the label in	*/
								{ int pad= (ul->nobox && no_box)? 0 : 1;
									if( ul->vertical ){
										label_box[0].x1 = ul->line.x2 - pad* bdr_pad- width/1.75;
										label_box[0].y2= label_box[0].y1= ul->line.y2- pad* wi->dev_info.bdr_pad/ 1.5- height/2;
										label_box[0].x2= label_box[0].x1+ width* 2/1.75 + pad* bdr_pad;
										label_box[2].y1= label_box[0].y1+ height + pad* wi->dev_info.bdr_pad/ 1.5;
									}
									else{
										label_box[0].x1 = ul->line.x2 - pad* wi->dev_info.bdr_pad/ 1.5- width/2;
										label_box[0].y2= label_box[0].y1= ul->line.y2- pad* bdr_pad- height/1.75;
										label_box[0].x2= label_box[0].x1+ width + pad* wi->dev_info.bdr_pad/ 1.5;
										label_box[2].y1= label_box[0].y1+ height* 2/1.75 + pad* bdr_pad;
									}
									label_box[2].x1= label_box[1].x2= label_box[1].x1= label_box[0].x2;
									label_box[1].y1= label_box[0].y2;
									label_box[1].y2= label_box[2].y2= label_box[2].y1;
									label_box[2].x2= label_box[0].x1;
									label_box[3].x1= label_box[2].x2;
									label_box[3].y1= label_box[2].y2;
									label_box[3].x2= label_box[0].x1;
									label_box[3].y2= label_box[0].y1;
									label_box[4].x1= label_box[3].x2;
									label_box[4].y2= label_box[4].y1= label_box[3].y2;
									label_box[4].x2= label_box[4].x1+ pad* wi->dev_info.bdr_pad;
								}
								memcpy( ul->box, label_box, sizeof(ul->box) );
								  /* 20010904: always draw some sort of a box - either erasing just the
								   \ area the text will occupy, or also the frame with shadow when so
								   \ requested.
								   */
								/* if( !no_box ) */{
								  Pixel lcolor0= AllAttrs[0].pixelValue;
								  int i, trans;
								  double shadowWidth= (ul->lineWidth+ zeroWidth)/2;
									sprintf( ps_comment, "label box #%d", lnr);
									if( !ul->nobox ){
										  /* Draw a shade in the "gray colour"
										   \ For this, the legendbox is temporarily
										   \ translated over (shadowWidth,shadowWidth)
										   */
										if( PS_STATE(wi)->Printing== PS_PRINTING){
											trans= (int) (0.5+ shadowWidth* wi->dev_info.var_width_factor - 1);
										}
										else{
											trans= (int) (0.5+ shadowWidth);
										}
										AllAttrs[0].pixelValue= black_pixel;
										for( i= 0; i< 5; i++ ){
											label_box[i].x1+= trans;
											label_box[i].y1+= trans;
											label_box[i].x2+= trans;
											label_box[i].y2+= trans;
										}
										wi->dev_info.xg_rect( wi->dev_info.user_state,
											rect_xsegs2xywh( 5, label_box), shadowWidth, L_VAR, 0,
											0, AllAttrs[0].pixelValue, 1, 0, AllAttrs[0].pixelValue, NULL
										);
										for( i= 0; i< 5; i++ ){
											label_box[i].x1-= trans;
											label_box[i].y1-= trans;
											label_box[i].x2-= trans;
											label_box[i].y2-= trans;
										}
										AllAttrs[0].pixelValue= lcolor0;
									}
									wi->dev_info.xg_clear( wi->dev_info.user_state, label_box[0].x1, label_box[0].y1,
										label_box[2].x1 - label_box[0].x1, label_box[2].y1 - label_box[0].y1, 0, 0
									);
									if( !ul->nobox ){
										wi->dev_info.xg_seg(wi->dev_info.user_state,
												 5, label_box, ul->lineWidth, L_VAR,
												 0, 0, 0, NULL
										);
									}
								}
								use_textPixel= 1;
								name= xgraph_NameBuf;
								strcpy( name, ul->labelbuf);
								if( ul->vertical ){
									sprintf( ps_comment, "Label strings #%d; dimensions wxh=%dx%d",
										lnr, width, height
									);
								}
								else{
									sprintf( ps_comment, "Label strings #%d; dimensions wxh=(%d+%d=%d)x%d",
										lnr, width- wi->dev_info.bdr_pad, wi->dev_info.bdr_pad, width, height
									);
								}
								if( ul->nobox && no_box ){
									  /* Centre the text relative to the drawn invisible box, using the
									   \ alignment functionality provided by the xg_text method. Correct
									   \ for the number of lines in the text.
									   */
									cx= (label_box[2].x1+ label_box[0].x1)/2;
									cy= (label_box[2].y1+ label_box[0].y1)/2- (lines-1)* wi->dev_info.legend_height/2;
								}
								while( name && xtb_getline( &name, &line ) ){
								  int xx, yy;
									if( ul->vertical ){
									  static double xs= 1.25;
	/* 
									  static double xs= 1.775;
										ps_old_font_offsets= True;
	 */
										wi->dev_info.xg_text(wi->dev_info.user_state,
											(xx= ul->line.x2- width/xs), (yy= ul->line.y2+ height/2- bdr_pad* 2),
											line, T_VERTICAL, T_LEGEND, NULL
										);
#if DEBUG==2
										{ XSegment l1, l2;
											l1.x1= xx- width/2, l1.x2= xx+ width/2;
											l1.y1= l1.y2= yy;
											l2.x1= l2.x2= xx;
											l2.y1= yy- height/2, l2.y2= yy+ height/2;
											wi->dev_info.xg_seg(wi->dev_info.user_state,
													 1, &l1, 0, L_VAR,
													 0, 0, 0, NULL
											);
											wi->dev_info.xg_seg(wi->dev_info.user_state,
													 1, &l2, 0, L_VAR,
													 0, 0, 0, NULL
											);
										}
#endif
										ul->line.x2+= wi->dev_info.legend_height;
									}
									else{
										ps_old_font_offsets= True;
										if( ul->nobox && no_box ){
											wi->dev_info.xg_text(wi->dev_info.user_state,
												cx, cy,
												line, T_CENTER, T_LEGEND, NULL
											);
										}
										else{
											wi->dev_info.xg_text(wi->dev_info.user_state,
												(xx= ul->line.x2- width/2), (yy= ul->line.y2- height/2),
												line, T_UPPERLEFT, T_LEGEND, NULL
											);
										}
										if( debugFlag && debugLevel== -2 ){
											ps_old_font_offsets= True;
											wi->dev_info.xg_text(wi->dev_info.user_state,
												ul->line.x2+ bdr_pad, ul->line.y2,
												line, T_CENTER, T_LEGEND, NULL
											);
										}
#if DEBUG==2
										{ XSegment l1, l2;
											l1.x1= xx- width/2, l1.x2= xx+ width/2;
											l1.y1= l1.y2= yy;
											l2.x1= l2.x2= xx;
											l2.y1= yy- height/2, l2.y2= yy+ height/2;
											wi->dev_info.xg_seg(wi->dev_info.user_state,
													 1, &l1, 0, L_VAR,
													 0, 0, 0, NULL
											);
											wi->dev_info.xg_seg(wi->dev_info.user_state,
													 1, &l2, 0, L_VAR,
													 0, 0, 0, NULL
											);
										}
#endif
										ul->line.y2+= wi->dev_info.legend_height;
										cy+= wi->dev_info.legend_height;
									}
								}
							}
							AllAttrs[0].pixelValue= color0;
							use_textPixel= 0;
						}
					}
					if( ul->free_buf ){
						xfree( ul->labelbuf );
						ul->free_buf= False;
					}
					ul= ul->next;
					lnr+= 1;
				}
				*ascanf_setNumber= asn;
				*ascanf_numPoints= anp;
			}
			else{
				wi->dev_info.xg_silent( wi->dev_info.user_state, wi->silenced || *prev_silent );
			}
			break;
	}
}

int DrawLegend( LocalWin *wi, int doit, void *dimension_data )
/*
 * This draws a legend of the data sets displayed.  Only those that
 * will fit are drawn.
 \ 990413 NVB (Nota Very Bene :)):
 \ For the time being, I personally find it more important to keep the grouping of sets
 \ together, than to show the sets in the legend in the same order as they are drawn.
 */
{
#if /* defined(_HPUX_SOURCE) && */ defined(__GNUC__)
  /* MAXSETS macro for MaxSets : a variable. In gcc, int spot[MaxSets] works!!	
   \ (dynamic array allocation)
   */
    ItemOfLegend spots[NUMSETS];
#else
	static ItemOfLegend *spots= NULL;
	static int spots_size= 0;
#endif
    int idx, spot, lineLen, oneLen, legend_len= 0, last_file_Y= 0, drawn_last= -1;
	int spot_id, first= -1, last= -1, ps_mark_scale, mh;
	XSegment leg_line, file_line, legend_box[5];
	char *prev_fileName, *fileName;
	int *prev_fn_shown= NULL, *fn_shown= NULL;
	int bdr_pad= wi->dev_info.bdr_pad/2, ps_width;
	int prev_silent;
	DataSet *this_set;

	if( wi->show_overlap== 1 && 
		!((wi->raw_display && wi->raw_once>= 0) || (wi->raw_once< 0 && wi->raw_val))
	){
		overlap(wi);
	}

	if( wi->legend_type== 1 ){
		return( DrawLegend2( wi, doit, dimension_data) );
	}

	TitleMessage( wi, "Legends" );

#ifdef DEBUG
	if( !wi ){
		fprintf( StdErr, "DrawLegend() called with NULL argument\n" );
		return(0);
	}
#endif
#if /* defined(_HPUX_SOURCE) && */ defined(__GNUC__)
	if( debugFlag){
		fprintf( StdErr, "DrawLegend(): spots[%d]\n", sizeof(spots)/sizeof(struct ItemOfLegend) );
	}
#else
	if( !spots || NUMSETS> spots_size ){
		xfree( spots);
		if( !(spots= (struct ItemOfLegend*) calloc( NUMSETS, sizeof(struct ItemOfLegend) )) ){
			fprintf( StdErr, "DrawLegend(): cannot allocate spots[%d]: %s\n",
				NUMSETS
			);
			spots_size= 0;
		}
		else{
			spots_size= NUMSETS;
		}
	}
#endif
	  /* First draw all lines of any labels as they should by the lowest structures	*/
	DrawULabels( wi, 0, doit, &prev_silent, dimension_data );
	if( PS_STATE(wi)->Printing== PS_PRINTING){
	  /* in new_ps.c:
	   \ PS_MARK * ui->baseWidth = PS_MARK * rd(VDPI/POINTS_PER_INCH*BASE_WIDTH)
	   */
		ps_mark_scale= PS_MARK * PS_STATE(wi)->baseWidth;
		ps_width= PS_DATA_WBASE * PS_STATE(wi)->baseWidth;
	}
	else{
		ps_mark_scale= PS_MARK* BASE_WIDTH* (Xdpi/POINTS_PER_INCH)/ 0.9432624113475178; /* 1;	*/
		ps_width= PS_DATA_WBASE* BASE_WIDTH* (Xdpi/POINTS_PER_INCH)/ 0.9432624113475178; /* 1;	*/
	}

	if( wi->no_legend ){
		goto finish_ULabels;
	}

	for( first= 0; first< setNumber; first++ ){
		if( draw_set(wi, first) && wi->numVisible[first] && AllSets[first].show_legend &&
			strcmp(AllSets[first].setName, "*NOLEGEND*") && strncmp( AllSets[first].setName, "*NOPLOT*", 8)
		){
			break;
		}
	}
	spot = wi->legend_uly+ (LegendLineWidth(wi,first)-1);
	lineLen = 0;
	  /* First pass checks what must be done, and how much vertical space it will take */
	for( idx = 0;  idx < NUMSETS;  idx++ ){
		this_set= &AllSets[idx];
		if( (this_set->numPoints > 0)
/* 			&& (spot + wi->dev_info.legend_height + 2 < wi->XOppY)	*/
		){
		  int mw, psw= (ps_width* (LegendLineWidth(wi,idx)-1))/ 2;
			if( (this_set->markFlag && !this_set->pixelMarks) ||
				(idx< NUMSETS-1 && AllSets[idx+1].markFlag && !AllSets[idx+1].pixelMarks) ||
				(idx && AllSets[idx-1].markFlag && !AllSets[idx-1].pixelMarks)
			){
				MarkerSizes( wi, idx, ps_mark_scale, &mw, &mh );
				mw*= 2;
			}
			else{
				mw= mh= 0;
			}
			  /* add width of the postscript marker to the x position	*/
			leg_line.x1 = wi->legend_ulx + mw;
			if( this_set->markFlag && ! this_set->pixelMarks ){
				leg_line.x1+= mark_w;
			}
			if( !strcmp( this_set->setName, "*NOLEGEND*") ){
				this_set->show_legend= 0;
			}
			if( !strncmp( this_set->setName, "*NOPLOT*", 8) ){
				this_set->draw_set= 0;
				wi->draw_set[idx]= 0;
			}
			if( draw_set( wi, idx) && this_set->show_legend && wi->numVisible[idx] ){
			  char *ntt, *parsed_end= NULL;
					/* Meets the criteria */
				if( first== -1 ){
					first= idx;
				}
				last= idx;
				spot+= mh+ psw;
				if( this_set->free_nameBuf ){
					xfree( this_set->nameBuf );
				}
				this_set->nameBuf= this_set->setName;
				this_set->free_nameBuf= False;
				if( (ntt= ParseTitlestringOpcodes( wi, idx, this_set->setName, &parsed_end )) ){
					this_set->nameBuf= ntt;
					if( ntt[0]== '`' && parsed_end[-1]== '`' ){
						parsed_end[-1]= '\0';
						xfree( this_set->setName );
						this_set->setName= strdup( &ntt[1] );
						xfree( this_set->nameBuf );
						this_set->nameBuf= this_set->setName;
					}
					else{
						this_set->free_nameBuf= True;
					}
				}
				oneLen = strlen(this_set->nameBuf)+ 1;
				legend_len+= oneLen+ 1;
				if( oneLen > lineLen) lineLen = oneLen;
				wi->legend_line[idx].low_y= spot- psw- wi->dev_info.legend_pad;
				spot+= increment_height( wi, this_set->nameBuf, '\n');
				legend_box[2].y1= spot;
				wi->legend_line[idx].high_y= spot;
				spot += 2 + bdr_pad+ psw + mh;
			}
			else{
				wi->legend_line[idx].low_y= wi->legend_line[idx].high_y= -1;
			}
		}
		this_set->filename_shown= 0;
	}
	if( wi->show_overlap ){
	  int op= 0;
		if( wi->overlap_buf && strlen(wi->overlap_buf) ){
			spot+= increment_height( wi, wi->overlap_buf, '\n') + 2 + bdr_pad+ (ps_width* (LegendLineWidth(wi,idx-1)-1))/ 2 + mh;
			op+= 1;
		}
		if( wi->overlap2_buf && strlen(wi->overlap2_buf) ){
			spot+= increment_height( wi, wi->overlap2_buf, '\n');
			if( !op ){
				spot+= 2 + bdr_pad+ (ps_width* (LegendLineWidth(wi,idx-1)-1))/ 2 + mh;
			}
		}
		legend_box[2].y1= spot;
	}
	legend_box[2].y1+= mh /* + wi->dev_info.legend_height */ + bdr_pad;
	wi->legend_lry= legend_box[2].y1;
	wi->legend_length= legend_len;

/* 	lineLen = lineLen * wi->dev_info.legend_width;	*/
	spot = wi->legend_uly+ (LegendLineWidth(wi,first)-1);

	if( doit && lineLen && first>= 0 ){
	  int correctX= 0, correctY= 0;
		legend_box[0].x1 = wi->legend_ulx - wi->dev_info.bdr_pad;
		if( (AllSets[first].markFlag && !AllSets[first].pixelMarks) ){
			MarkerSizes( wi, first, ps_mark_scale, &correctX, &correctY );
		}
		if( AllSets[first].show_llines && (!AllSets[first].noLines || AllSets[first].barFlag> 0 ) ){
			correctX= MAX( correctX, LegendLineWidth(wi,first)/2);
			correctY= MAX( correctY, LegendLineWidth(wi,first)/2);
		}
		legend_box[0].y2= legend_box[0].y1= wi->legend_uly- bdr_pad- correctY;
		legend_box[0].x2= wi->legend_frx + wi->dev_info.bdr_pad;
		legend_box[2].x1= legend_box[1].x2= legend_box[1].x1= legend_box[0].x2;
		legend_box[1].y1= legend_box[0].y2;
		legend_box[1].y2= legend_box[2].y2= legend_box[2].y1;
		legend_box[2].x2= legend_box[0].x1;
		legend_box[3].x1= legend_box[2].x2;
		legend_box[3].y1= legend_box[2].y2;
		legend_box[3].x2= legend_box[0].x1;
		legend_box[3].y2= legend_box[0].y1;
		legend_box[4].x1= legend_box[3].x2;
		legend_box[4].y2= legend_box[4].y1= legend_box[3].y2;
		legend_box[4].x2= legend_box[4].x1+ wi->dev_info.bdr_pad;
		strcpy( ps_comment, "legend box");
		if( !wi->no_legend_box ){
		  Pixel color0= AllAttrs[0].pixelValue;
		  int i, trans;
		  double shadowWidth= (axisWidth+ zeroWidth)/2;
			  /* Draw a shade in the "gray colour"
			   \ For this, the legendbox is temporarily
			   \ translated over (shadowWidth,shadowWidth)
			   */
			if( PS_STATE(wi)->Printing== PS_PRINTING){
/* 				trans= shadowWidth* PS_DATA_WBASE * PS_STATE(wi)->baseWidth - 2;	*/
				trans= (int) (shadowWidth* wi->dev_info.var_width_factor - 2+ 0.5);
			}
			else{
				trans= (int) (0.5+ shadowWidth);
			}
			AllAttrs[0].pixelValue= black_pixel;
			for( i= 0; i< 5; i++ ){
				legend_box[i].x1+= trans;
				legend_box[i].y1+= trans;
				legend_box[i].x2+= trans;
				legend_box[i].y2+= trans;
			}
/* 
			wi->dev_info.xg_seg(wi->dev_info.user_state,
					 5, legend_box, shadowWidth, L_VAR,
					 0, 0, 0, NULL
			);
 */
			wi->dev_info.xg_rect( wi->dev_info.user_state,
				rect_xsegs2xywh( 5, legend_box), shadowWidth, L_VAR, 0,
				0, AllAttrs[0].pixelValue, 1, 0, AllAttrs[0].pixelValue, NULL
			);
			for( i= 0; i< 5; i++ ){
				legend_box[i].x1-= trans;
				legend_box[i].y1-= trans;
				legend_box[i].x2-= trans;
				legend_box[i].y2-= trans;
			}
			AllAttrs[0].pixelValue= color0;
		}
		if( !wi->no_legend_box ){
			wi->dev_info.xg_clear( wi->dev_info.user_state, legend_box[0].x1, legend_box[0].y1,
				legend_box[2].x1 - legend_box[0].x1, legend_box[2].y1 - legend_box[0].y1, 0, 0
			);
			wi->dev_info.xg_seg(wi->dev_info.user_state,
					 5, legend_box, axisWidth, L_AXIS,
					 0, 0, 0, NULL
			);
		}
	}
	spot = wi->legend_uly+ (LegendLineWidth(wi,first)-1);
	lineLen = 0;
	  /* second pass draws the text */
	{  char *name= NULL;
		for( idx = 0;  idx < NUMSETS;  idx++ ){
			this_set= &AllSets[idx];
			if( (this_set->numPoints > 0)
/*	 			&& (spot + wi->dev_info.legend_height + 2 < wi->XOppY)	*/
			){
			  int mw, psw= (ps_width* (LegendLineWidth(wi,idx)- 1))/ 2;
				if( (this_set->markFlag && !this_set->pixelMarks) ||
					(idx< NUMSETS-1 && AllSets[idx+1].markFlag && !AllSets[idx+1].pixelMarks) ||
					(idx && AllSets[idx-1].markFlag && !AllSets[idx-1].pixelMarks)
				){
					MarkerSizes( wi, idx, ps_mark_scale, &mw, &mh );
					mw*= 2;
				}
				else{
					mw= mh= 0;
				}
				  /* add width of the postscript marker to the x position	*/
				leg_line.x1 = wi->legend_ulx + mw;
				if( this_set->markFlag && ! this_set->pixelMarks ){
					leg_line.x1+= mark_w;
				}
				if( !strcmp( this_set->setName, "*NOLEGEND*") ){
					this_set->show_legend= 0;
				}
				if( !strncmp( this_set->setName, "*NOPLOT*", 8) ){
					this_set->draw_set= 0;
					wi->draw_set[idx]= 0;
				}
				if( doit && draw_set(wi, idx) && this_set->show_legend && wi->numVisible[idx] ){
				  char *line;
				  Boolean hl= False;
					if( !xgraph_NameBuf || strlen(this_set->nameBuf)> xgraph_NameBufLen ){
						xfree( xgraph_NameBuf );
						xgraph_NameBuf= XGstrdup(this_set->nameBuf);
						xgraph_NameBufLen= strlen(xgraph_NameBuf);
					}
					else{
						strcpy( xgraph_NameBuf, this_set->nameBuf );
					}
					name= xgraph_NameBuf;
						/* Meets the criteria */
					spot+= mh+ psw;
					while( name && xtb_getline( &name, &line ) ){
						oneLen = strlen(line)+ 1;
						if( oneLen > lineLen) lineLen = oneLen;
						if( wi->legend_line[idx].highlight && !hl && highlight_par[1] ){
						  Pixel color0= AllAttrs[0].pixelValue;
						  XSegment highlight;
							highlight.x1 = wi->legend_ulx + mw;
							highlight.x2 = wi->legend_lrx;
							highlight.y1 = MIN(wi->legend_line[idx].high_y,wi->legend_line[idx].low_y);
							highlight.y2 = MAX(wi->legend_line[idx].high_y,wi->legend_line[idx].low_y);
							AllAttrs[0].pixelValue=
								(wi->legend_line[idx].pixvalue< 0)? wi->legend_line[idx].pixelValue : highlightPixel;
							wi->dev_info.xg_clear( wi->dev_info.user_state, highlight.x1, highlight.y1,
								highlight.x2 - highlight.x1, highlight.y2 - highlight.y1, 1, 0
							);
							AllAttrs[0].pixelValue= color0;
							hl= True;
							textPixel= (this_set->pixvalue< 0)? this_set->pixelValue : 
								AllAttrs[this_set->pixvalue].pixelValue;
							use_textPixel= 1;
						}
						ps_old_font_offsets= 1;
						wi->dev_info.xg_text(wi->dev_info.user_state,
							leg_line.x1+ wi->dev_info.bdr_pad/2, spot+2, line,
							T_UPPERLEFT, T_LEGEND, NULL
						);
						  /* we currently take use_textPixel as a once-only switch, so we don't
						   \ save/restore its state!
						   */
						use_textPixel= 0;
						if( debugFlag ){
							fprintf( StdErr, "DrawLegend(): setName \"%s\", file#%d @(%d,%d)\n",
								line, this_set->fileNumber, leg_line.x1, spot+2
							);
							fflush( StdErr );
						}
						spot+= wi->dev_info.legend_height;
					}
/* 					wi->legend_line[idx].high_y= spot;	*/
					legend_box[2].y1= spot;
					spot += 2 + bdr_pad+ psw + mh;
				}
			}
		}
	}
	if( doit && wi->show_overlap ){
	  int op= 0;
		if( wi->overlap_buf && strlen(wi->overlap_buf) ){
			spot+= mh+ (ps_width* (LegendLineWidth(wi,idx-1)- 1))/ 2;
			wi->dev_info.xg_text(wi->dev_info.user_state,
				wi->legend_ulx+ wi->dev_info.bdr_pad/2, spot+2, wi->overlap_buf,
				T_UPPERLEFT, T_LEGEND, NULL
			);
			op+= 1;
		}
		if( wi->overlap2_buf && strlen(wi->overlap2_buf) ){
			spot+= (op)? wi->dev_info.legend_height : mh+ (ps_width* (LegendLineWidth(wi,idx-1)- 1))/ 2;
			wi->dev_info.xg_text(wi->dev_info.user_state,
				wi->legend_ulx+ wi->dev_info.bdr_pad/2, spot+2, wi->overlap2_buf,
				T_UPPERLEFT, T_LEGEND, NULL
			);
		}
	}

	spot = wi->legend_uly+ (LegendLineWidth(wi,first)-1);

	fileName= NULL;

	  /* third pass draws the lines and the fileNames.	*/
	for( spot_id= 0, idx = 0;  idx < NUMSETS;  idx++ ){
		this_set= &AllSets[idx];
		if( (this_set->numPoints > 0)
/* 			&& (spot + wi->dev_info.legend_height + 2 < wi->XOppY)	*/
		){
		  int mw, psw= (ps_width* (LegendLineWidth(wi,idx)- 1))/ 2;
			if( (this_set->markFlag && !this_set->pixelMarks) ||
				(idx< NUMSETS-1 && AllSets[idx+1].markFlag && !AllSets[idx+1].pixelMarks) ||
				(idx && AllSets[idx-1].markFlag && !AllSets[idx-1].pixelMarks)
			){
				MarkerSizes( wi, idx, ps_mark_scale, &mw, &mh );
			}
			else{
				mw= mh= 0;
			}
			if( !strcmp( this_set->setName, "*NOLEGEND*") ){
				this_set->show_legend= 0;
			}
			if( !strncmp( this_set->setName, "*NOPLOT*", 8) ){
				this_set->draw_set= 0;
				wi->draw_set[idx]= 0;
			}
			if( draw_set( wi, idx) && this_set->show_legend && wi->numVisible[idx] ){
				leg_line.x1 = wi->legend_ulx + mw;
				spot+= mh;
				if( this_set->markFlag && ! this_set->pixelMarks ){
					leg_line.x1+= mark_w/ 2;
				}
				leg_line.x2 = wi->legend_lrx;
				file_line.x1= leg_line.x2;
				file_line.x2= leg_line.x2;
				leg_line.y1 = leg_line.y2 = spot - wi->dev_info.legend_pad;
				if( doit ){
					if( this_set->show_llines ){
						if( wi->legend_line[idx].highlight || (this_set->barFlag> 0 && this_set->barType==2) ){
							HighlightSegment( wi, idx, 1, &leg_line, HL_WIDTH(this_set->lineWidth), L_VAR );
						}
						wi->dev_info.xg_seg(wi->dev_info.user_state,
							1, &leg_line, LINEWIDTH(idx), L_VAR,
							LINESTYLE(idx), PIXVALUE(idx), NULL
						);
					}
					if( this_set->markFlag && ! this_set->pixelMarks ){
						wi->dev_info.xg_dot(wi->dev_info.user_state,
							leg_line.x1, leg_line.y1,
							P_MARK, MARKSTYLE(idx), PIXVALUE(idx), idx, NULL
						);

					}
					if( debugFlag ){
						fprintf( StdErr, "DrawLegend(): setName \"%s\", line (%d,%d-%d)\n",
							this_set->nameBuf, leg_line.x1, leg_line.y1, leg_line.y2
						);
						fflush( StdErr );
					}
				}
				if( spots){
					spots[spot_id].idx= idx;
					spots[spot_id].y= leg_line.y1;
				}
				prev_fileName= fileName;
				prev_fn_shown= fn_shown;
				fileName= (wi->labels_in_legend)? this_set->YUnits : this_set->fileName;
				fn_shown= &(this_set->filename_shown);
			}
			if( wi->new_file[idx] || (idx== first && idx== last) ){
				if( spots && drawn_last>= 0 ){
					file_line.y1= spots[last_file_Y].y;
					file_line.y2= spots[drawn_last].y;
					if( !prev_fileName ){
						if( wi->labels_in_legend ){
							prev_fileName= AllSets[(idx)? idx-1 : idx].YUnits;
						}
						else{
							prev_fileName= AllSets[(idx)? idx-1 : idx].fileName;
						}
						prev_fn_shown= &(AllSets[(idx)? idx-1 : idx].filename_shown);
					}
					if( draw_set(wi, spots[drawn_last].idx) && AllSets[spots[drawn_last].idx].show_legend &&
						wi->numVisible[spots[drawn_last].idx]
					){
						if( doit ){
							wi->dev_info.xg_seg(wi->dev_info.user_state,
								1, &file_line, axisWidth, L_AXIS,
								0, 0, 0, NULL
							);
							if( wi->filename_in_legend> 0 ){
								wi->dev_info.xg_text(wi->dev_info.user_state,
									file_line.x1+wi->dev_info.bdr_pad, (file_line.y1 + file_line.y2)/ 2,
									prev_fileName,
									T_LEFT, T_LEGEND, NULL
								);
							}
							if( debugFlag ){
								fprintf( StdErr, "DrawLegend(): set #%d: new file \"%s\" (old set #%d-#%d) (%d-%d)\n",
									idx, this_set->fileName,
									last_file_Y, drawn_last,
									spots[last_file_Y].y, spots[drawn_last].y
								);
							}
						}
						*prev_fn_shown= 1;
						last_file_Y= spot_id;
						drawn_last= -1;
					}
					else{
						if( doit && debugFlag ){
							fprintf( StdErr, "DrawLegend(): hidden set #%d: new file \"%s\" (old set #%d-#%d) (%d-%d)\n",
								idx, this_set->fileName,
								last_file_Y, drawn_last,
								spots[last_file_Y].y, spots[drawn_last].y
							);
						}
						last_file_Y= spot_id;
					}
				}
				  /* 950919: we must update prev_fileName here too; after all, this is used
				   \ for showing.
				   */
				prev_fileName= fileName;
				prev_fn_shown= fn_shown;
				fileName= (wi->labels_in_legend)? this_set->YUnits : this_set->fileName;
				fn_shown= &(this_set->filename_shown);
			}
			if( draw_set(wi, idx) && this_set->show_legend && wi->numVisible[idx] ){
				spot += 2 + increment_height( wi, this_set->nameBuf, '\n') +
					bdr_pad+ 2*psw + mh;
				drawn_last= spot_id;
				spot_id+= 1;
			}
			else if( debugFlag){
				fprintf( StdErr, "DrawLegend(): set #%d *NOLEGEND*\n", idx);
			}
		}
	}
	if( spots && drawn_last!= -1 ){
		if( drawn_last<= 0 ){
			drawn_last= spot_id- 1;
		}
		file_line.y1= spots[last_file_Y].y;
		file_line.y2= spots[drawn_last].y;
		if( doit ){
			wi->dev_info.xg_seg(wi->dev_info.user_state,
					 1, &file_line, axisWidth, L_AXIS,
					 0, 0, 0, NULL
			);
		}
		if( wi->filename_in_legend>0 ){
			if( !fileName ){
				fileName= (wi->labels_in_legend)? AllSets[drawn_last].YUnits : AllSets[drawn_last].fileName; 
				fn_shown= &(AllSets[drawn_last].filename_shown);
			}
			if( fileName && doit ){
				wi->dev_info.xg_text(wi->dev_info.user_state,
					file_line.x1+wi->dev_info.bdr_pad, (file_line.y1 + file_line.y2)/ 2,
					fileName,
					T_LEFT, T_LEGEND, NULL
				);
				if( debugFlag ){
					fprintf( StdErr, "DrawLegend(): last set #%d: new file \"%s\" (old set #%d-#%d) (%d-%d)\n",
						idx, fileName,
						last_file_Y, drawn_last,
						spots[last_file_Y].y, spots[drawn_last].y
					);
				}
			}
			*fn_shown= 1;
		}
	}
finish_ULabels:;
	DrawULabels( wi, 1, doit, &prev_silent, dimension_data );
	TitleMessage( wi, NULL );
	return(1);
}

int DrawLegend2( LocalWin *wi, int doit, void *dimension_data )
/*
 * This draws a legend of the data sets displayed.  Only those that
 * will fit are drawn.
 */
{ ALLOCA( legend_entry_pos, ItemOfLegend, setNumber, spots_size);
    int idx, Yposition, lineLen, oneLen, legend_len= 0, last_file_Y= -1, drawn_last= -1;
	int entry_id, first_drawn= -1, first= -1, last= -1, ps_mark_scale, mh;
	XSegment leg_line, file_line[3], legend_box[5];
	char *prev_fileName= NULL, *fileName= NULL;
	int *prev_fn_shown= NULL, *fn_shown= NULL, groups= 0;
	int bdr_pad= wi->dev_info.bdr_pad/2, ps_width, showlines= 0;
	int prev_silent;
	DataSet *this_set= NULL;
	double bbw= wi->hard_devices[PS_DEVICE].dev_legend_size;
	LegendDimensions *dim= dimension_data;
	int first_markerHeight= -1, markerWidth= -1, tempSize, text_lx, max_left_pos= -1;

#ifdef DEBUG
	if( !wi ){
		fprintf( StdErr, "DrawLegend2() called with NULL argument\n" );
		return(0);
	}
#endif
	if( !doit ){
		if( dim ){
		  /* This means that we are going to determine the legend's dimensions; not
		   \ just its height. For ease of calculations, we initialise everything to
		   \ 0, including the few legend_.. fields in the wi structure that are directly
		   \ used to determine drawing co-ordinates.
		   */
			dim->maxFontWidth= XFontWidth( legendFont.font );
			dim->first_markerHeight= 0;
			dim->markerWidth= 0;
			dim->maxName= dim->maxFile= 0;
			dim->overlap_size= 0;
			wi->legend_uly= wi->legend_ulx= 0;
			wi->legend_lrx= wi->legend_frx= 0;
			tempSize= 0;
		}
	}

	if( debugFlag ){
		fprintf( StdErr, "DrawLegend2(0x%lx,doit=%s,dim=0x%lx)\n", wi, (doit)? "True" : "False", dim );
	}

	TitleMessage( wi, "Legends" );

	  /* First draw all lines of any labels as they should by the lowest structures	*/
	DrawULabels( wi, 0, doit, &prev_silent, dimension_data );
	if( PS_STATE(wi)->Printing== PS_PRINTING){
	  /* in new_ps.c:
	   \ PS_MARK * ui->baseWidth = PS_MARK * rd(VDPI/POINTS_PER_INCH*BASE_WIDTH)
	   */
		ps_mark_scale= PS_MARK * PS_STATE(wi)->baseWidth;
		ps_width= PS_DATA_WBASE * PS_STATE(wi)->baseWidth;
	}
	else{
		ps_mark_scale= PS_MARK* BASE_WIDTH* (Xdpi/POINTS_PER_INCH)/ 0.9432624113475178; /* 1;	*/
		ps_width= PS_DATA_WBASE* BASE_WIDTH* (Xdpi/POINTS_PER_INCH)/ 0.9432624113475178; /* 1;	*/
	}
	if( wi->no_legend ){
		goto finish_ULabels;
	}
	Yposition = wi->legend_uly+ bdr_pad;
	lineLen = 0;
	  /* First pass checks what must be done, and how much vertical space it will take */
	for( entry_id= 0, idx = 0;  idx < NUMSETS;  idx++ ){
		if( (AllSets[idx].numPoints > 0)
/* 			&& (Yposition + wi->dev_info.legend_height + 2 < wi->XOppY)	*/
		){
		  int mw, psw= (ps_width* (LegendLineWidth(wi,idx)-1))/ 2;
			this_set= &AllSets[idx];
			  /* 20001216: In all occurences below where a call to MarkerSizes() was issued
			   \ only when the set actually had marks, this has been changed to a call
			   \ in any case. I.e., there is always a call to MarkerSizes() to determine the
			   \ dimensions of a marker. To keep the old code available for a while, this
			   \ has been effected by ensuring that the if always evaluates to true.
			   */
			if( 1 || (this_set->markFlag && !this_set->pixelMarks) ||
				(idx< NUMSETS-1 && AllSets[idx+1].markFlag && !AllSets[idx+1].pixelMarks) ||
				(idx && AllSets[idx-1].markFlag && !AllSets[idx-1].pixelMarks)
			){
				MarkerSizes( wi, idx, ps_mark_scale, &mw, &mh );
			}
			else{
				mw= ps_mark_scale * psm_base;
				mh= ps_mark_scale * psm_base;
			}
			if( !strcmp( this_set->setName, "*NOLEGEND*") ){
				this_set->show_legend= 0;
			}
			if( !strncmp( this_set->setName, "*NOPLOT*", 8) ){
				this_set->draw_set= 0;
				wi->draw_set[idx]= 0;
			}
			if( draw_set( wi, idx) && this_set->show_legend && wi->numVisible[idx] ){
			  int h, lineHeight, totalHeight, Ypos;
			  char *ntt, *parsed_end= NULL;

				if( this_set->free_nameBuf ){
					xfree( this_set->nameBuf );
				}
				this_set->nameBuf= this_set->setName;
				this_set->free_nameBuf= False;
				if( (ntt= ParseTitlestringOpcodes( wi, idx, this_set->setName, &parsed_end )) ){
					this_set->nameBuf= ntt;
					if( ntt[0]== '`' && parsed_end[-1]== '`' ){
						parsed_end[-1]= '\0';
						xfree( this_set->setName );
						this_set->setName= strdup( &ntt[1] );
						xfree( this_set->nameBuf );
						this_set->nameBuf= this_set->setName;
						this_set->free_nameBuf= False;
					}
					else{
						this_set->free_nameBuf= True;
					}
				}

				if( this_set->barFlag ){
				  double bHeight= wi->bar_legend_dimension_weight[1]* HL_WIDTH(bbw), bbHeight;
				  double bWidth= wi->bar_legend_dimension_weight[0]* mw+ wi->dev_info.bdr_pad;
					if( LINEWIDTH(idx)> bHeight ){
						bHeight= LINEWIDTH(idx);
					}
					  /* 20001216: for vertical spacing, scale with wi->bar_legend_dimension_weight[2]	*/
					bbHeight= bHeight*= wi->dev_info.var_width_factor;
					bHeight*= wi->bar_legend_dimension_weight[2];
					if( wi->legend_line[idx].highlight ){
					  double hlw= HL_WIDTH(this_set->lineWidth)* wi->dev_info.var_width_factor;
						if( bbHeight+ hlw> bHeight ){
							bHeight= bbHeight+ hlw;
						}
					}
					mh= MAX( (int)( bHeight+ 1 )/ 2, mh);
					mw= MAX( (int)( bWidth+ 1)/2, mw);
				}
					/* Meets the criteria */
				if( first== -1 ){
					first_drawn= entry_id;
					first= idx;
				}
				last= idx;
				psw= MAX( psw, mh);
				  /* We always calculate the max. markerwidth, to be able to left-align
				   \ the text entries.
				   */
				if( mw> markerWidth ){
					markerWidth= mw;
				}
				if( !doit && dim ){
					if( first_markerHeight<= 0 ){
/* 						first_markerHeight= MAX(mw, psw);	*/
						first_markerHeight= MAX(mw, mh);
					}
					dim->markerWidth= markerWidth;
				}
				if( (!this_set->noLines /* || this_set->barFlag> 0 */) && this_set->show_llines ){
					showlines+= 1;
				}
				if( showlines ){
					leg_line.x1= wi->legend_ulx+ LEG2_LINE_LENGTH(wi,markerWidth)+ wi->dev_info.bdr_pad;
				}
				else{
					leg_line.x1= wi->legend_ulx+ 2* markerWidth+ wi->dev_info.bdr_pad;
				}
				  /* Determine the right-most position at which text must be left-aligned:	*/
				if( leg_line.x1> max_left_pos ){
					max_left_pos= leg_line.x1;
				}

				  /* lineHeight: the half-height of a single-line legend entry, or the
				   \ height above.
				   */
				lineHeight= MAX( wi->dev_info.legend_height/2, psw)+ wi->dev_info.legend_pad;
				h= increment_height( wi, this_set->nameBuf, '\n');
				totalHeight= h+ wi->dev_info.legend_pad;

				Ypos= Yposition+ lineHeight/* + bdr_pad/2 */;
				wi->legend_line[idx].low_y= Ypos;
				  /* Add some padding. This can be more generous when there is some form of
				   \ marker (mark, bar) next to the text that is higher than the text itself.
				   */
				if( psw> wi->dev_info.legend_height/2 ){
					wi->legend_line[idx].low_y-= wi->dev_info.legend_pad+ wi->dev_info.legend_height;
				}
				else{
					wi->legend_line[idx].low_y-= wi->dev_info.legend_pad+ wi->dev_info.legend_height/2;
				}
				  /* Store the y-co-ordinate where this legend entry must be shown	*/
				if( legend_entry_pos){
					legend_entry_pos[entry_id].idx= idx;
					legend_entry_pos[entry_id].y= Ypos;
					entry_id+= 1;
				}
				  /* This is the lower co-ordinate of the text zone that may need highlighting.	*/
				wi->legend_line[idx].high_y= Ypos+ totalHeight;
				  /* Add some padding to that, too	*/
				if( psw<= h/2 ){
					wi->legend_line[idx].high_y-= wi->dev_info.legend_height/2;
				}

				oneLen = (this_set->nameBuf)? strlen(this_set->nameBuf)+ 1 : 1;
				legend_len+= oneLen+ 1;
				if( oneLen > lineLen) lineLen = oneLen;

				  /* Determine where this entry ends (max. Y co-ordinate)	*/
				if( h> psw ){
					totalHeight= h- wi->dev_info.legend_height/2+ bdr_pad/2+ wi->dev_info.legend_pad;
				}
				else{
					totalHeight= psw+ wi->dev_info.legend_pad;
				}
				Yposition= MAX( wi->legend_line[idx].high_y+ bdr_pad/2, Ypos+ totalHeight);

				if( !doit && dim && !dim->first_markerHeight ){
					dim->first_markerHeight= first_markerHeight /* MAX( text_hheight, first_markerHeight ) */;
				}
				  /* Keep track of the vertical extents of the legendbox to be able to draw the box
				   \ around it.
				   */
				legend_box[2].y1= Yposition;
			}
			else{
				wi->legend_line[idx].low_y= wi->legend_line[idx].high_y= -1;
			}
		}
		AllSets[idx].filename_shown= 0;
	}

	if( doit && first!= -1 && 
		((wi->legend_ulx== wi->legend_frx) || (!wi->legend_placed && max_left_pos> wi->dev_info.area_w) )
	){
		if( debugFlag ){
			fprintf( StdErr, "DrawLegend2(): zero width legend and/or outside canvas: let's see if a redraw changes that!\n" );
		}
		wi->redraw= True;
		return(0);
	}
	if( wi->show_overlap ){
	  int op= 0;
		if( wi->overlap_buf && strlen(wi->overlap_buf) ){
			Yposition+= increment_height( wi, wi->overlap_buf, '\n') + 2* wi->dev_info.var_width_factor + bdr_pad+
				(ps_width* (LegendLineWidth(wi,idx-1)-1))/ 2 + mh;
			op+= 1;
		}
		if( wi->overlap2_buf && strlen(wi->overlap2_buf) ){
			Yposition+= increment_height( wi, wi->overlap2_buf, '\n');
			if( !op ){
				Yposition+= 2* wi->dev_info.var_width_factor + bdr_pad+ (ps_width* (LegendLineWidth(wi,idx-1)-1))/ 2 + mh;
			}
		}
		legend_box[2].y1= Yposition;
	}

	legend_box[2].y1+= bdr_pad;

	wi->legend_lry= legend_box[2].y1;
	wi->legend_length= legend_len;

/* 	lineLen = lineLen * wi->dev_info.legend_width;	*/
	Yposition = wi->legend_uly+ (LegendLineWidth(wi,first)-1);

	if( doit && lineLen && first>= 0 ){
		legend_box[0].x1 = wi->legend_ulx - wi->dev_info.bdr_pad;
		legend_box[0].y2= legend_box[0].y1= wi->legend_uly- bdr_pad;
		legend_box[0].x2= wi->legend_frx + wi->dev_info.bdr_pad;
		legend_box[2].x1= legend_box[1].x2= legend_box[1].x1= legend_box[0].x2;
		legend_box[1].y1= legend_box[0].y2;
		legend_box[1].y2= legend_box[2].y2= legend_box[2].y1;
		legend_box[2].x2= legend_box[0].x1;
		legend_box[3].x1= legend_box[2].x2;
		legend_box[3].y1= legend_box[2].y2;
		legend_box[3].x2= legend_box[0].x1;
		legend_box[3].y2= legend_box[0].y1;
		legend_box[4].x1= legend_box[3].x2;
		legend_box[4].y2= legend_box[4].y1= legend_box[3].y2;
		legend_box[4].x2= legend_box[4].x1+ wi->dev_info.bdr_pad;
		strcpy( ps_comment, "legend box");
		if( !wi->no_legend_box ){
		  Pixel color0= AllAttrs[0].pixelValue;
		  int i, trans;
		  double shadowWidth= (axisWidth+ zeroWidth)/2;
			  /* Draw a shade in the "gray colour"
			   \ For this, the legendbox is temporarily
			   \ translated over (shadowWidth,shadowWidth)
			   */
			if( PS_STATE(wi)->Printing== PS_PRINTING){
/* 				trans= shadowWidth* PS_DATA_WBASE * PS_STATE(wi)->baseWidth - 2;	*/
				trans= (int) (0.5+ shadowWidth* wi->dev_info.var_width_factor - 2);
			}
			else{
				trans= (int) (0.5+ shadowWidth);
			}
			AllAttrs[0].pixelValue= black_pixel;
			for( i= 0; i< 5; i++ ){
				legend_box[i].x1+= trans;
				legend_box[i].y1+= trans;
				legend_box[i].x2+= trans;
				legend_box[i].y2+= trans;
			}
/* 
			wi->dev_info.xg_seg(wi->dev_info.user_state,
					 5, legend_box, shadowWidth, L_VAR,
					 0, 0, 0, NULL
			);
 */
			wi->dev_info.xg_rect( wi->dev_info.user_state,
				rect_xsegs2xywh( 5, legend_box), shadowWidth, L_VAR, 0,
				0, AllAttrs[0].pixelValue, 1, 0, AllAttrs[0].pixelValue, NULL
			);
			for( i= 0; i< 5; i++ ){
				legend_box[i].x1-= trans;
				legend_box[i].y1-= trans;
				legend_box[i].x2-= trans;
				legend_box[i].y2-= trans;
			}
			AllAttrs[0].pixelValue= color0;
/* 		}	*/
/* 		if( !wi->no_legend_box ){	*/
			wi->dev_info.xg_clear( wi->dev_info.user_state, legend_box[0].x1, legend_box[0].y1,
				legend_box[2].x1 - legend_box[0].x1, legend_box[2].y1 - legend_box[0].y1, 0, 0
			);
			wi->dev_info.xg_seg(wi->dev_info.user_state,
					 5, legend_box, axisWidth, L_AXIS,
					 0, 0, 0, NULL
			);
		}
		if( debugFlag && PS_STATE(wi)->Printing== X_DISPLAY ){
			wi->dev_info.xg_dot(wi->dev_info.user_state,
						wi->legend_ulx, wi->legend_uly,
						P_PIXEL, 0, -1, zeroPixel ^ normPixel, 0, NULL
			);
		}
	}
	lineLen = 0;

	  /* second pass draws the text */
	{  char *name= NULL;
		for( entry_id= 0, idx = 0;  idx < NUMSETS;  idx++ ){
			if( (AllSets[idx].numPoints > 0)
/*	 			&& (Yposition + wi->dev_info.legend_height + 2 < wi->XOppY)	*/
			){
			  int mw;
				this_set= &AllSets[idx];
				if( 1 || (this_set->markFlag && !this_set->pixelMarks) ||
					(idx< NUMSETS-1 && AllSets[idx+1].markFlag && !AllSets[idx+1].pixelMarks) ||
					(idx && AllSets[idx-1].markFlag && !AllSets[idx-1].pixelMarks)
				){
					MarkerSizes( wi, idx, ps_mark_scale, &mw, &mh );
					  /* 20001213: quick hack to avoid too wide legends	*/
					if( wi->legend_ulx+ LEG2_LINE_LENGTH(wi,markerWidth)> wi->dev_info.area_w ){
					  double ms= this_set->markSize;
						set_NaN(this_set->markSize);
						MarkerSizes( wi, idx, ps_mark_scale, &mw, &mh );
						this_set->markSize= ms;
					}
				}
				else{
					mw= 2* ps_mark_scale * psm_base;
					mh= ps_mark_scale * psm_base;
				}
				  /* add width of the postscript marker to the x position	*/
				if( showlines ){
					leg_line.x1 = wi->legend_ulx+ LEG2_LINE_LENGTH(wi,markerWidth)+ wi->dev_info.bdr_pad;
				}
				else{
					  /* 20001218: I think there must be 2* markWidth here (too)	*/
					leg_line.x1 = wi->legend_ulx+ 2* markerWidth+ wi->dev_info.bdr_pad;
				}
				  /* 20030224: apparently we override the above calculated values here: */
					leg_line.x1= max_left_pos;
				if( !strcmp( this_set->setName, "*NOLEGEND*") ){
					this_set->show_legend= 0;
				}
				if( !strncmp( this_set->setName, "*NOPLOT*", 8) ){
					this_set->draw_set= 0;
					wi->draw_set[idx]= 0;
				}
				if( draw_set(wi, idx) && this_set->show_legend && wi->numVisible[idx] ){
				  char *line;
				  Boolean hl= False;
					if( this_set->nameBuf ){
						if( !xgraph_NameBuf || strlen(this_set->nameBuf)> xgraph_NameBufLen ){
							xfree( xgraph_NameBuf );
							xgraph_NameBuf= XGstrdup(this_set->nameBuf);
							xgraph_NameBufLen= strlen(xgraph_NameBuf);
						}
						else{
							strcpy( xgraph_NameBuf, this_set->nameBuf );
						}
					}
					name= xgraph_NameBuf;
						/* Meets the criteria */
					Yposition= legend_entry_pos[entry_id].y- wi->dev_info.legend_pad;
					entry_id+= 1;
					while( name && xtb_getline( &name, &line ) ){
					  XSegment highlight;
						oneLen = strlen(line)+ 1;
						if( oneLen > lineLen) lineLen = oneLen;
						highlight.x1 = leg_line.x1;
						highlight.y1 = MIN(wi->legend_line[idx].high_y,wi->legend_line[idx].low_y);
						highlight.x2 = wi->legend_lrx;
						highlight.y2 = MAX(wi->legend_line[idx].high_y,wi->legend_line[idx].low_y);
						if( wi->legend_line[idx].highlight && !hl && highlight_par[1] ){
							if( doit ){
							  Pixel color0= AllAttrs[0].pixelValue;
								AllAttrs[0].pixelValue=
									(wi->legend_line[idx].pixvalue< 0)? wi->legend_line[idx].pixelValue : highlightPixel;
								wi->dev_info.xg_clear( wi->dev_info.user_state, highlight.x1, highlight.y1,
									highlight.x2 - highlight.x1, highlight.y2 - highlight.y1, 1, 0
								);
								AllAttrs[0].pixelValue= color0;
								hl= True;
								textPixel= (this_set->pixvalue< 0)? this_set->pixelValue : 
									AllAttrs[this_set->pixvalue].pixelValue;
								use_textPixel= True;
							}
						}
						if( doit ){

							if( debugFlag && PS_STATE(wi)->Printing== X_DISPLAY ){
								wi->dev_info.xg_rect( wi->dev_info.user_state,
									rect_xsegs2xywh( 1, &highlight), 0, L_VAR, 0,
									PIXVALUE(idx), 0, 0, 0, NULL
								);
							}

							wi->dev_info.xg_text(wi->dev_info.user_state,
								leg_line.x1+ wi->dev_info.bdr_pad/3, Yposition, line,
								T_LEFT /* T_UPPERLEFT */, T_LEGEND, NULL
							);
							use_textPixel= False;
							if( debugFlag ){
								fprintf( StdErr, "DrawLegend2(): setName \"%s\", file#%d @(%d,%d)\n",
									line, this_set->fileNumber, leg_line.x1, Yposition+2
								);
								fflush( StdErr );
							}
							Yposition+= wi->dev_info.legend_height;
						}
					 	else if( dim ){
						 int len;
							if( xtb_has_greek( line) ){
								dim->maxFontWidth= MAX(dim->maxFontWidth, XFontWidth(legend_greekFont.font) );
							}
							if( !use_X11Font_length ){
								len = strlen(line);
							}
							else{
								len= XGTextWidth( wi, line, T_LEGEND, NULL );
							}
							tempSize= MAX( tempSize, len);
							if( !dim->first_markerHeight ){
							  int text_height= (Yposition- highlight.y1);
								dim->first_markerHeight= MAX( text_height, first_markerHeight );
							}
						}
					}
					if( doit ){
						legend_box[2].y1= Yposition;
					}
					else if( dim ){
						text_lx= leg_line.x1+ wi->dev_info.bdr_pad/2;
						if( tempSize > dim->maxName){
							dim->maxName = tempSize;
						}
						if( text_lx> wi->legend_lrx ){
							wi->legend_lrx= text_lx;
						}
					}
				}
			}
		}
	}
	if( !doit && dim ){
		if( wi->textrel.used_gsTextWidth<= 0 && PS_STATE(wi)->Printing!= X_DISPLAY ){
		  /* scale the screen-based lenght estimate with the ratio of PS length estimate over
		   \ default screen length estimate (XFontWidth(); used when !use_X11Font_length)
		   */
			dim->maxName= (int)((dim->maxName)* ((double)wi->dev_info.legend_width/ dim->maxFontWidth )) + wi->dev_info.bdr_pad;
		}
		wi->legend_lrx+= dim->maxName;
		if( !dim->first_markerHeight ){
			dim->first_markerHeight= first_markerHeight;
		}
	}
	if( wi->show_overlap ){
		if( doit ){
		  int op= 0;
			if( wi->overlap_buf && strlen(wi->overlap_buf) ){
				Yposition+= bdr_pad+ mh+ (ps_width* (LegendLineWidth(wi,idx-1)- 1))/ 2;
				wi->dev_info.xg_text(wi->dev_info.user_state,
					wi->legend_ulx+ wi->dev_info.bdr_pad/2, Yposition+2* wi->dev_info.var_width_factor, wi->overlap_buf,
					T_UPPERLEFT, T_LEGEND, NULL
				);
				op+= 1;
			}
			if( wi->overlap2_buf && strlen(wi->overlap2_buf) ){
				Yposition+= (op)? increment_height( wi, wi->overlap_buf, '\n') :
					bdr_pad+ mh+ (ps_width* (LegendLineWidth(wi,idx-1)- 1))/ 2;
				wi->dev_info.xg_text(wi->dev_info.user_state,
					wi->legend_ulx+ wi->dev_info.bdr_pad/2, Yposition+2* wi->dev_info.var_width_factor, wi->overlap2_buf,
					T_UPPERLEFT, T_LEGEND, NULL
				);
			}
		}
		else if( dim ){
		  char *o1= (wi->overlap_buf)? wi->overlap_buf : "";
		  char *o2= (wi->overlap2_buf)? wi->overlap2_buf : "";
			if( !use_X11Font_length ){
				dim->overlap_size = strlen( o1 );
				dim->overlap_size = MAX( dim->overlap_size, strlen( o2 ) );
			}
			else{
				dim->overlap_size= XGTextWidth( wi, o1, T_LEGEND, NULL );
				dim->overlap_size= MAX( dim->overlap_size, XGTextWidth( wi, o2, T_LEGEND, NULL ));
			}
			if( wi->textrel.used_gsTextWidth<= 0 && PS_STATE(wi)->Printing!= X_DISPLAY ){
			  /* scale the screen-based lenght estimate with the ratio of PS length estimate over
			   \ default screen length estimate (XFontWidth(); used when !use_X11Font_length)
			   */
				dim->overlap_size= (int)((dim->overlap_size)* ((double)wi->dev_info.legend_width/ dim->maxFontWidth )) +
					wi->dev_info.bdr_pad;
			}
			dim->overlap_size+= (1+ overlap_legend_tune + SIGN(overlap_legend_tune))* wi->dev_info.bdr_pad;
		}
	}

	fileName= NULL;

	  /* third pass draws the lines and the fileNames.	*/
	for( entry_id= 0, idx = 0;  idx < NUMSETS;  idx++ ){
		if( (AllSets[idx].numPoints > 0)
/* 			&& (Yposition + wi->dev_info.legend_height + 2 < wi->XOppY)	*/
		){
		  int mw, psw= (ps_width* (LegendLineWidth(wi,idx)- 1))/ 2;
			this_set= &AllSets[idx];
			if( 1 || (this_set->markFlag && !this_set->pixelMarks) ||
				(idx< NUMSETS-1 && AllSets[idx+1].markFlag && !AllSets[idx+1].pixelMarks) ||
				(idx && AllSets[idx-1].markFlag && !AllSets[idx-1].pixelMarks)
			){
				MarkerSizes( wi, idx, ps_mark_scale, &mw, &mh );
				  /* 20001213: quick hack to avoid to wide legends	*/
				if( wi->legend_ulx+ LEG2_LINE_LENGTH(wi,markerWidth)> wi->dev_info.area_w ){
				  double ms= this_set->markSize;
					set_NaN(this_set->markSize);
					MarkerSizes( wi, idx, ps_mark_scale, &mw, &mh );
					this_set->markSize= ms;
				}
			}
			else{
				mw= ps_mark_scale * psm_base;
				mh= ps_mark_scale * psm_base;
			}
			if( !strcmp( this_set->setName, "*NOLEGEND*") ){
				this_set->show_legend= 0;
			}
			if( !strncmp( this_set->setName, "*NOPLOT*", 8) ){
				this_set->draw_set= 0;
				wi->draw_set[idx]= 0;
			}
			psw= MAX( mh, psw);
			if( draw_set( wi, idx) && this_set->show_legend && wi->numVisible[idx] ){
			  double minIntense, maxIntense, scaleIntense;
			  Pixel colorx= (this_set->pixvalue< 0)? this_set->pixelValue : AllAttrs[this_set->pixvalue].pixelValue;
			  int respix= False;
				  /* 20001222: Setting the intensity colour was done here	*/
				leg_line.x1 = wi->legend_ulx;
				Yposition= legend_entry_pos[entry_id].y;
				leg_line.x2 = leg_line.x1+ LEG2_LINE_LENGTH(wi,markerWidth);
						leg_line.x2= max_left_pos- wi->dev_info.bdr_pad;
				  /* 20001213: quick hack to avoid taking up too much space:	*/
				if( !wi->legend_placed && mw> 0 ){
					while( leg_line.x2> wi->dev_info.area_w ){
						leg_line.x2-= mw;
					}
				}
				file_line[0].x2= file_line[1].x1= wi->legend_lrx+ bdr_pad;
				file_line[2].x1= file_line[1].x2= wi->legend_lrx+ bdr_pad;
				file_line[2].x2= file_line[0].x1= file_line[0].x2- bdr_pad;
				leg_line.y1 = leg_line.y2 = Yposition - wi->dev_info.legend_pad;
				if( doit ){
				  double bxpos= (leg_line.x2+ leg_line.x1)/ 2;
					  /* This draws the line/bar/marker display	*/
					if( (!this_set->noLines || this_set->barFlag> 0) && this_set->show_llines ){
						if( this_set->barFlag> 0 ){
						  double lw= wi->bar_legend_dimension_weight[1]* HL_WIDTH(bbw), bHeight, hlw;
						  double bWidth= wi->bar_legend_dimension_weight[0]* mw+ wi->dev_info.bdr_pad;
						  XRectangle rec;
						  XSegment bar_line;
							if( LINEWIDTH(idx)> lw ){
								lw= LINEWIDTH(idx);
							}
/* 							if( wi->legend_line[idx].highlight ){	*/
/* 								lw+= 2* HL_WIDTH(this_set->lineWidth);	*/
/* 							}	*/
							bHeight= lw* wi->dev_info.var_width_factor;
							if( wi->legend_line[idx].highlight ){
								hlw= HL_WIDTH(this_set->lineWidth);
							}
							bar_line.x1= bxpos;
							bar_line.x2= bxpos;
							bar_line.y1= leg_line.y1+ bHeight/ 2;
							bar_line.y2= bar_line.y1- bHeight;
							switch( this_set->barType ){
#ifdef HOOK_NOT_BAR
								case 5:{
								  XSegment hook[2];
									hook[1].x1= hook[0].x2= hook[0].x1= bar_line.x1- bWidth/2;
									hook[1].x2= hook[1].x1+ bWidth;
									goto leg_hook_continue;
								case 6:
									hook[1].x1= hook[0].x2= hook[0].x1= bar_line.x1+ bWidth/2;
									hook[1].x2= hook[1].x1- bWidth;
leg_hook_continue:;
									hook[0].y1= bar_line.y1;
									hook[1].y2= hook[1].y1= hook[0].y2= bar_line.y2;
									if( wi->legend_line[idx].highlight ){
										HighlightSegment( wi, idx, 2, hook, hlw, L_VAR );
									}
									wi->dev_info.xg_seg(wi->dev_info.user_state,
												2, hook,
												LINEWIDTH(idx), L_VAR,
												LINESTYLE(idx), PIXVALUE(idx), NULL);
									break;
								}
#else
								case 5:
								case 6:{
									if( this_set->barType== 5 ){
										bxpos+= bWidth/2;
									}
									else{
										bxpos-= bWidth/2;
									}
									bWidth*= 2;
									 /* fall through to default case: */
								}
#endif
								default:
									bar_line.x1= bxpos;
									bar_line.x2= bxpos;
									rec= *rect_xywh( bxpos- bWidth/2,leg_line.y1- bHeight/2, bWidth, bHeight );
									Draw_Bar( wi, &rec, &bar_line, (double) bWidth/ wi->dev_info.var_width_factor,
										this_set->barType, L_VAR, this_set, idx, -1,
										this_set->lineWidth, LINESTYLE(idx), this_set->lineWidth, 0,
										minIntense, maxIntense, scaleIntense, colorx, respix
									);
									break;
							}
						}
						else{
							if( wi->legend_line[idx].highlight ){
								HighlightSegment( wi, idx, 1, &leg_line, HL_WIDTH(this_set->lineWidth), L_VAR );
							}
							wi->dev_info.xg_seg(wi->dev_info.user_state,
								1, &leg_line, LINEWIDTH(idx), L_VAR,
								LINESTYLE(idx), PIXVALUE(idx), NULL
							);
						}
					}
					if( this_set->markFlag ){
						  /* 20001222: setting intensity colour is now done here, where it is needed:	*/
						if( doit && wi->error_type[idx]== INTENSE_FLAG ){
							if( IntensityColourFunction.NColours> 1 ){
								if( IntensityColourFunction.range_set ){
									minIntense= IntensityColourFunction.range.min;
									maxIntense= IntensityColourFunction.range.max;
								}
								else{
									minIntense= wi->SS_I.min;
									maxIntense= wi->SS_I.max;
								}
								scaleIntense= (IntensityColourFunction.NColours- 1)/ (maxIntense- minIntense);
								Retrieve_IntensityColour( wi, this_set, this_set->av_error,
									minIntense, maxIntense, scaleIntense, &colorx, &respix
								);
								psThisRGB= xg_IntRGB;
							}
						}
						switch( this_set->pixelMarks ){
							case 2:
								wi->dev_info.xg_dot(wi->dev_info.user_state,
									(int) bxpos, leg_line.y1,
									P_DOT, 0, PIXVALUE(idx), idx, NULL
								);
								break;
							case 1:
								wi->dev_info.xg_dot(wi->dev_info.user_state,
									(int) bxpos, leg_line.y1,
									P_PIXEL, 0, PIXVALUE(idx), idx, NULL
								);
								break;
							case 0:
								/* Distinctive markers */
								wi->dev_info.xg_dot(wi->dev_info.user_state,
									(int) bxpos, leg_line.y1,
									P_MARK, MARKSTYLE(idx), PIXVALUE(idx), idx, NULL
								);
								break;
						}
						if( respix ){
							if( this_set->pixvalue< 0 ){
								this_set->pixelValue= colorx;
							}
							else{
								AllAttrs[this_set->pixvalue].pixelValue= colorx;
							}
						}
					}
					if( debugFlag ){
						fprintf( StdErr, "DrawLegend2(): setName \"%s\", line (%d,%d-%d)\n",
							this_set->nameBuf, leg_line.x1, leg_line.y1, leg_line.y2
						);
						fflush( StdErr );
					}
				}
				prev_fileName= fileName;
				prev_fn_shown= fn_shown;
				fileName= (wi->labels_in_legend)? this_set->YUnits : this_set->fileName;
				fn_shown= &(this_set->filename_shown);
			}
			if( wi->new_file[idx] || (idx== first && idx== last) ){
				if( legend_entry_pos && drawn_last>= 0 ){
				  int dl= legend_entry_pos[drawn_last].idx, fly;
					if( last_file_Y< 0 ){
						last_file_Y= first_drawn;
					}
					fly= legend_entry_pos[last_file_Y].idx;
					file_line[0].y1= file_line[0].y2= file_line[1].y1= wi->legend_line[fly].low_y;
					file_line[2].y1= file_line[2].y2= file_line[1].y2= wi->legend_line[dl].high_y;
					if( !prev_fileName ){
						if( wi->labels_in_legend ){
							prev_fileName= AllSets[(idx)? idx-1 : idx].YUnits;
						}
						else{
							prev_fileName= AllSets[(idx)? idx-1 : idx].fileName;
						}
						prev_fn_shown= &(AllSets[(idx)? idx-1 : idx].filename_shown);
					}
					if( !doit && dim ){
						if( wi->labels_in_legend && !fileName ){
							if( idx ){
								fileName= AllSets[idx].YUnits= XGstrdup( AllSets[idx-1].YUnits );
							}
						}
						if( wi->labels_in_legend && !AllSets[idx].XUnits ){
							if( idx ){
								AllSets[idx].XUnits= XGstrdup( AllSets[idx-1].XUnits );
							}
						}
					}
					if( draw_set(wi, dl) && AllSets[dl].show_legend &&
						wi->numVisible[dl]
					){
						if( doit ){
							wi->dev_info.xg_seg(wi->dev_info.user_state,
								3, file_line, axisWidth, L_AXIS,
								0, 0, 0, NULL
							);
							if( wi->filename_in_legend> 0 ){
								wi->dev_info.xg_text(wi->dev_info.user_state,
									file_line[1].x1+wi->dev_info.bdr_pad, (file_line[1].y1 + file_line[1].y2)/ 2,
									prev_fileName,
									T_LEFT, T_LEGEND, NULL
								);
							}
							if( debugFlag ){
								fprintf( StdErr, "DrawLegend2(): set #%d: new file \"%s\" (old set #%d-#%d) (%d-%d)\n",
									idx, this_set->fileName,
									last_file_Y, drawn_last,
									file_line[1].y1, file_line[1].y2
								);
							}
						}
						else if( dim ){
							if( !use_X11Font_length ){
								tempSize = strlen(fileName);
							}
							else{
								tempSize= XGTextWidth( wi, fileName, T_LEGEND, NULL );
							}
							if( tempSize > dim->maxFile ){
								dim->maxFile = tempSize;
							}
							if( wi->filename_in_legend ){
								file_line[1].x1+= wi->dev_info.bdr_pad;
								if( file_line[1].x1> wi->legend_frx ){
									wi->legend_frx= file_line[1].x1;
								}
							}
						}
						*prev_fn_shown= 1;
						last_file_Y= entry_id;
						drawn_last= -1;
					}
					else{
						if( doit && debugFlag ){
							fprintf( StdErr, "DrawLegend2(): hidden set #%d: new file \"%s\" (old set #%d-#%d) (%d-%d)\n",
								idx, this_set->fileName,
								last_file_Y, drawn_last,
								legend_entry_pos[last_file_Y].y, legend_entry_pos[drawn_last].y
							);
						}
						last_file_Y= entry_id;
					}
				}
				  /* 950919: we must update prev_fileName here too; after all, this is used
				   \ for showing.
				   */
				prev_fileName= fileName;
				prev_fn_shown= fn_shown;
				fileName= (wi->labels_in_legend)? this_set->YUnits : this_set->fileName;
				fn_shown= &(this_set->filename_shown);
				groups+= 1;
			}
			if( draw_set(wi, idx) && this_set->show_legend && wi->numVisible[idx] ){
			  int h= increment_height( wi, this_set->nameBuf, '\n');
				Yposition += 2 + MAX(h, psw) + bdr_pad;
				drawn_last= entry_id;
				entry_id+= 1;
			}
			else if( debugFlag){
				fprintf( StdErr, "DrawLegend2(): set #%d *NOLEGEND*\n", idx);
			}
		}
	}
	  /* 990416: don't draw the grouping line if all sets are in the same group
	   \ (last_file_Y< 0?!), and no filenames or labelnames are to be shown.
	   */
	if( legend_entry_pos && drawn_last!= -1 && (groups> 1 || wi->filename_in_legend> 0) ){
	  int dl, fly;
		if( drawn_last<= 0 ){
			drawn_last= entry_id- 1;
		}
		dl= legend_entry_pos[drawn_last].idx;
		if( last_file_Y< 0 ){
			last_file_Y= first_drawn;
		}
		fly= legend_entry_pos[last_file_Y].idx;
/* 		file_line.y1= legend_entry_pos[last_file_Y].y;	*/
/* 		file_line.y2= legend_entry_pos[drawn_last].y;	*/
		file_line[0].y1= file_line[0].y2= file_line[1].y1= wi->legend_line[fly].low_y;
		file_line[2].y1= file_line[2].y2= file_line[1].y2= wi->legend_line[dl].high_y;
		if( doit ){
			wi->dev_info.xg_seg(wi->dev_info.user_state,
					 3, file_line, axisWidth, L_AXIS,
					 0, 0, 0, NULL
			);
		}
		if( wi->filename_in_legend>0 ){
			if( !fileName ){
				fileName= (wi->labels_in_legend)? AllSets[dl].YUnits : AllSets[dl].fileName; 
				fn_shown= &(AllSets[dl].filename_shown);
			}
			if( fileName ){
				if( doit ){
					wi->dev_info.xg_text(wi->dev_info.user_state,
						file_line[1].x1+wi->dev_info.bdr_pad, (file_line[1].y1 + file_line[1].y2)/ 2,
						fileName,
						T_LEFT, T_LEGEND, NULL
					);
					if( debugFlag ){
						fprintf( StdErr, "DrawLegend2(): last set #%d: new file \"%s\" (old set #%d-#%d) (%d-%d)\n",
							idx, fileName,
							last_file_Y, drawn_last,
							file_line[1].y1, file_line[1].y2
						);
					}
				}
				else if( dim ){
					if( !use_X11Font_length ){
						tempSize = strlen(fileName);
					}
					else{
						tempSize= XGTextWidth( wi, fileName, T_LEGEND, NULL );
					}
					if( tempSize > dim->maxFile ){
						dim->maxFile = tempSize;
					}
					if( wi->filename_in_legend ){
						file_line[1].x1+= wi->dev_info.bdr_pad;
						if( file_line[1].x1> wi->legend_frx ){
							wi->legend_frx= file_line[1].x1;
						}
					}
				}
			}
			  /* 20010105: move following line upwards within fileName!=NULL scope?	*/
			*fn_shown= 1;
		}
	}
	if( !doit && dim ){
		if( !wi->legend_frx ){
			wi->legend_frx= wi->legend_lrx- 0*wi->dev_info.bdr_pad/ 1.5;
		}
		if( wi->filename_in_legend ){
			if( wi->textrel.used_gsTextWidth<= 0 && PS_STATE(wi)->Printing!= X_DISPLAY ){
			  /* scale the screen-based lenght estimate with the ratio of PS length estimate over
			   \ default screen length estimate (XFontWidth(); used when !use_X11Font_length)
			   */
				dim->maxFile= (int)((dim->maxFile)* ((double)wi->dev_info.legend_width/ dim->maxFontWidth )) + wi->dev_info.bdr_pad;
			}
			wi->legend_frx+= dim->maxFile;
		}
	}
	if( wi->show_overlap ){
		if( !doit && dim ){
			if( wi->legend_ulx+ dim->overlap_size+ wi->dev_info.bdr_pad> wi->legend_frx ){
				wi->legend_frx= wi->legend_ulx+ dim->overlap_size+ wi->dev_info.bdr_pad;
			}
		}
	}
finish_ULabels:;
	DrawULabels( wi, 1, doit, &prev_silent, dimension_data );
	TitleMessage( wi, NULL);
	GCA();
	return(1);
}


