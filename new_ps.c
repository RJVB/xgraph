/*
vim:ts=4:sw=4:
 */
/*
 * Postscript output for xgraph
 *
 * Rick Spickelmier
 * David Harrison
 */

#include "config.h"
IDENTIFY( "PostScript device code" );

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include "xgout.h"
#include <ctype.h>

#include "hard_devices.h"

#include <string.h>
#ifndef _APOLLO_SOURCE
#	include <strings.h>
#endif

#include "xgraph.h"
#include "NaN.h"

#include "copyright.h"

/*
 * Working macros
 */

#define OUT		fprintf
#define PS(str)		OUT(psFile, str)
#define PSU(str)	OUT(ui->psFile, str)
#define MAX(a, b)	((a) > (b) ? (a) : (b))
#define SWAP(a,b,type)	{type c= (a); (a)= (b); (b)= c;}

#include "xfree.h"

extern Pixmap dotMap ;

#ifndef MIN
#	define MIN(a,b)	(((a)>(b))?(b):(a))
#endif
#ifndef MAX
#	define MAX(a,b)	(((a)<(b))?(b):(a))
#endif

/*
 * Globals
 */

int internal_psMarkers= 17, psMarkers= 17;

static double PS_scale;		/* devs/micron */
char ps_comment[1024];
static char *ps_comment_buf= NULL;

extern int _use_gsTextWidth;

int legend_number, stroked= 0;

static double PSscale;

extern int ps_xpos, ps_ypos;
extern double ps_scale;
double graph_width= 0, graph_height= 0;
double ps_l_offset= 15.0, ps_b_offset= 15.0;

/*
 * Externals and forwards
 */

#ifndef __STDC__
	extern char *malloc(), *calloc();
#endif
static void psScale(), psFonts(), psMarks(), psText(), psClear(), psSeg(), psRect(), psPolyg(), psArc(), psDot(), psEnd();

/*
 * Local structures and definitions
 */

#include "new_ps.h"

#include "fdecl.h"

static long minY, minX, maxX, maxY;
Boolean mXSet= False, mYSet= False;
int psEPS= False, psDSC= True, psSetPage= False;
double psSetPage_width= -1, psSetPage_height= -1;
double psSetHeight_corr_factor= 1, psSetWidth_corr_factor= 1;

static void psboundsX( int x )
{
	if( !mXSet ){
		minX= maxX= x;
		mXSet= True;
	}
	else{
		minX= MIN(minX, x);
		maxX= MAX(maxX, x);
	}
}

static void psboundsY( int y )
{
	if( !mYSet ){
		minY= maxY= y;
		mYSet= True;
	}
	else{
		minY= MIN(minY, y);
		maxY= MAX(maxY, y);
	}
}

static void psbounds( int x, int y )
{
	psboundsX(x), psboundsY(y);
}

static int _IX( psUserInfo *ui, int x, int y)
{
/* 	if( psEPS ){	*/
/* 		SWAP(x,y, int);	*/
/* 	}	*/
	psboundsX(x);
	return(x);
}

#define IX(v,w)	_IX(ui,v,w)

static int _IY(psUserInfo *ui, int x, int y)
{
/* 	if( psEPS ){	*/
/* 		SWAP(x, y, int );	*/
/* 	}	*/
	y= ui->height_devs - y;
	psboundsY(y);
	return(y);
}

#define IY(v,w)	_IY(ui,v,w)


int rd(double dbl)
/* Short and sweet rounding function */
{
    if (dbl < 0.0) {
		return ((int) (dbl - 0.5));
    } else {
		return ((int) (dbl + 0.5));
    }
}

int ps_coloured= 0, ps_font_reencode= 1;
static int pssoft_coloured= 1;

/* 20020216: this is a pointer to an RGB entry containing a colour specification that may
 \ be different from the RGB associated with <pixel> (in PSconvert_colour()), e.g. because
 \ the X server didn't yield the requested RGB, but something close (or gamma-corrected).
 \ This pointer is sticky during the externally visible calls (xg_XXXX methods in the xgOut outInfo
 \ structure), but is reset before these exit. Thus, the user's code will have to cache the
 \ desired value, and re-initialise psThisRGB when a sequence of drawing commands should make
 \ use of that particular colour.
 \ NB: this mechanism ensures that intended RGB values can make it into the PostScript code,
 \ independent of what colour the X server substituted for them. It does not in any sense
 \ guarantee that the printed outcome is correct. It also means that any gamma correction applied
 \ by the X server (for its own display...) will not afflict the printed output - which may be
 \ an advantage, or not.
 */
RGB *psThisRGB= NULL;

char *PSconvert_colour( FILE *fp, Pixel pixel)
{  double avcolour;
   XColor rgb;
   static char cbuf[3*64+8];

	if( psThisRGB ){
		rgb.red= psThisRGB->red;
		rgb.green= psThisRGB->green;
		rgb.blue= psThisRGB->blue;
	}
	else{
		rgb.pixel= pixel;
		XQueryColor( disp, cmap, &(rgb) );
	}
	if( pssoft_coloured || ps_coloured ){
		sprintf( cbuf, "%s %s %s",
			d2str( rgb.red/ 65535.0, NULL, NULL), 
			d2str( rgb.green/ 65535.0, NULL, NULL),
			d2str( rgb.blue/ 65535.0, NULL, NULL)
		);
	}
	else{
		avcolour= xtb_PsychoMetric_Gray( &rgb );
		if( debugFlag && fp ){
			fprintf( fp, "%% PSconvert_colour(R=%ld,G=%ld,B=%ld)= psychom. PS gray %g\n",
				(long) rgb.red, (long) rgb.green, (long) rgb.blue, avcolour/ 65535.0
			);
		}
		d2str( avcolour/65535.0, "%g", cbuf);
	}
	return( cbuf );
}

static int width_devs, bdr_pad, legend_height, legend_width;

static LocalWin *theWin_Info;
extern LocalWin *ActiveWin;

extern char PrintTime[];
extern xtb_frame HO_Dialog;

extern FILE *StdErr;

int psSilent( char *user_state, Boolean silent)
{ struct userInfo *st= (struct userInfo*) user_state;
  int ret;
	if( !st ){
		return(0);
	}
	ret= st->silent;
	if( silent ){
		strcpy( ps_comment, "PS output off" );
	}
	else{
		strcpy( ps_comment, "PS output on" );
	}
	st->silent= silent;
	return( ret );
}

#define STRINGCHECK(s,max)	StringCheck(s,max,__FILE__,__LINE__)

double psm_incr= 0.75, psm_base= 1.5;
int psm_changed= 0, ps_page_nr= 1;

static double transl_compensation= 25.82;

static int fg_ccol, fg_called= 0, newfile;
static Pixel fg_cpix;
static RGB *fg_rgb= NULL;

extern int use_textPixel;
extern Pixel textPixel;

double Font_Width_Estimator= FONT_WIDTH_EST, _Font_Width_Estimator;
XGStringList *PSSetupIncludes;

int CustomFont_psWidth( CustomFont *cf )
{ static void *lcf= NULL;
  static int w;
	if( cf ){
		if( cf!= lcf ){
		  double font_size= cf->PSPointSize * INCHES_PER_POINT * VDPI;
			w= rd( font_size * _Font_Width_Estimator );
		}
		lcf= cf;
	}
	else{
		lcf= NULL;
		w= 0;
	}
	return(w);
}

int CustomFont_psHeight( CustomFont *cf )
{ static void *lcf= NULL;
  static int h;
	if( cf ){
		if( cf!= lcf ){
		  double font_size= cf->PSPointSize * INCHES_PER_POINT * VDPI;
			h= rd( font_size );
		}
		lcf= cf;
	}
	else{
		lcf= NULL;
		h= 0;
	}
	return( h );
}

int psLineLength= 0;

static void _psText( FILE *fp, char *Text, Boolean standout, char *fn, double size, int socolour,
	char *wrapstr, int is_comment, int escape, int delimit
)
{
	if( !is_comment ){
		if( standout ){
			OUT( fp, "cp /%s %lg (", fn, size);
		}
		else if( delimit ){
			if( debugFlag && socolour>= 0 ){
				OUT( fp, "(%s) print ( width is ) print (%s) stringwidth pop buf cvs print\n"
					"(; unscaled: ) print (%s) stringwidth pop %lg mul graph-scale mul buf cvs print (\\n) print\n",
					Text, Text, Text, PSscale
				);
			}
			OUT( fp, "(");
		}
	}
	while( *Text ){
		if( *Text!= 0x7f ){
			switch( *Text ){
				case '\n':
					fputc( *Text, fp );
					psLineLength= 0;
					break;
				case '(':
				case ')':
					if( escape ){
				case '\\':
						fputc( '\\', fp);
					}
					psLineLength+= 1;
					  /* no break!	*/
				default:
					fputc( *Text, fp);
					psLineLength+= 1;
					break;
			}
		}
		if( psLineLength>= 255 ){
			if( !wrapstr && !is_comment ){
				fputc( '\\', fp );
			}
			fputc( '\n', fp );
			if( wrapstr ){
				fputs( wrapstr, fp );
				psLineLength= strlen(wrapstr);
			}
			else{
				if( is_comment ){
					fputs( "%%+ ", fp );
					psLineLength= 4;
				}
				else{
					psLineLength= 0;
				}
			}
		}
		Text++;
	}
	if( !is_comment ){
		if( standout ){
			OUT( fp, ") %s",
				PSconvert_colour( NULL, (use_textPixel)? textPixel : (xg_text_highlight)? black_pixel : white_pixel )
			);
			OUT( fp, " %s standout\n",
				PSconvert_colour( NULL, AllAttrs[socolour].pixelValue)
			);
		}
		else if( delimit ){
			if( socolour>= 0 ){
				OUT( fp, ") fgColour show\n");
			}
			else{
				OUT( fp, ")" );
			}
		}
	}
}

void psPrint( FILE *fp, char *text, char *wrapstr, int is_comment )
{
	if( text ){
		_psText( fp, text, False, "", 0, 0, wrapstr, is_comment, True, True );
	}
}

void PSprint_string( FILE *fp, char *header, char *wrapheader, char *trailer, char *string, int is_comment )
{
	if( count_char( string, '\n')< 1 ){
/* 		fprintf( fp, "%s%s", string, trailer );	*/
		if( header ){
			_psText( fp, header, False, "", 0, 0, NULL, is_comment, False, True );
		}
		psPrint( fp, string, wrapheader, is_comment );
	}
	else{
	  char *Buf= XGstrdup( string), *buf= Buf, *c;
		while( xtb_getline( &buf, &c ) ){
			if( c!= Buf ){
				fputs( "\n", fp);
				psLineLength= 0;
			}
/* 			fprintf( fp, "%s%s", wrapheader, c );	*/
			if( header ){
				_psText( fp, header, False, "", 0, 0, NULL, is_comment, False, True );
			}
			else if( c!= Buf ){
				_psText( fp, wrapheader, False, "", 0, 0, NULL, is_comment, False, True );
			}
			psPrint( fp, c, wrapheader, is_comment );
		}
/* 		fprintf( fp, "%s", trailer );	*/
		xfree( Buf );
	}
	_psText( fp, trailer, False, "", 0, 0, NULL, is_comment, False, True );
	fflush( fp );
}

int Write_ps_comment( FILE *fp )
{ int len;
  char *combuf= (ps_comment_buf)? ps_comment_buf : ps_comment;
	if( (len= strlen(combuf)) ){
		if( combuf== ps_comment && len> sizeof(ps_comment)/sizeof(char) ){
			fprintf( StdErr, "Write_ps_comment(): possibly fatal error: length %d > %d\n\"%s\"\n",
				len, sizeof(ps_comment)/sizeof(char), combuf
			);
			fflush(StdErr);
		}
		if( len> 10 ){
			XStoreName( disp, HO_Dialog.win, combuf );
			xtb_XSync( disp, False );
		}
		fputs("% PScomment: ", fp);
		if( combuf[0]== 'X' && combuf[1]== 'G' ){
		  char *c;
			  /* Cut after the process-id: there the read_buffer is stored, which
			   \ can contain invalid characters for PS.
			   */
			if( (c= strstr( combuf, "$=")) ){
				while( *c && !isspace(*c) ){
					c++;
				}
				*c= '\0';
				fputs( "[windowtitle truncated] ", fp );
			}
		}
		PSprint_string( fp, "%  ", "%  ", "\n", combuf, True );
		combuf[0]= '\0';
	}
	fflush( fp );
	xfree( ps_comment_buf );
	return(len);
}


static void DoPSIncludes( FILE *psFile, char *caller )
{ FILE *fp;
  XGStringList *li= PSSetupIncludes;
  char buf[512];
	while( li ){
		if( (fp= fopen( li->text, "r")) ){
			fprintf( psFile, "\n%%%%IncludeFile %s\n", li->text );
			while( fgets( buf, 511, fp ) ){
				fputs( buf, psFile );
			}
			fprintf( psFile, "%%%%EndFile %s\n", li->text );
			fclose(fp);
		}
		else{
			fprintf( StdErr, "%s: can't open PSSetupInclude file \"%s\" (%s)\n",
				caller,
				li->text, serror()
			);
		}
		li= li->next;
	}
}

extern int use_gsTextWidth;

#include <pwd.h>
#include <sys/types.h>

char PS_greek_font[]= "Symbol";
int ps_previous_dimensions= False, ps_pages= 0;
double ps_hbbox[4];

unsigned long PSdrawCounter= 0;

static unsigned short psNw= 1, psNh= 1;
unsigned short psNgraph_wide= 1, psNgraph_high= 1;
unsigned short psGraph_x_idx= 0, psGraph_y_idx= 0;

/*ARGSUSED*/
int psInit( FILE *psFile, int width, int height, int orient,
	char *tf, double ts, char *lef, double les, char *laf, double las, char *af, double as,
	LocalWin *win_info, char errmsg[ERRBUFSIZE], int initFile
)
/*
 * The basic coordinate system is points (roughly 1/72 inch).
 * However,  most laser printers can do much better than that.
 * We invent a coordinate system based on VDPI dots per inch.
 * This goes along the long side of the page.  The long side
 * of the page is LDIM inches in length,  the short side
 * SDIM inches in length.  We call this unit a `dev'.
 * We map `width' and `height' into devs.
 */
{ double font_size;
  char hnam[80], dnam[80], *linfo= NULL, title[128], *Window_Name;
  xgOut *outInfo= &(win_info->dev_info);			/* Returned device info   */
  struct userInfo *ui= (struct userInfo*) outInfo->user_state;
  extern char XGraphBuildString[];
  LocalWin *wi= win_info;
  int axis_height;
  extern char UPrintFileName[], *PrintFileName;
  char *pf= (UPrintFileName[0])? UPrintFileName : (PrintFileName)? PrintFileName : "";

    if( !ui ){
		ui = (struct userInfo *) calloc( 1, sizeof(struct userInfo));
	}
    ui->psFile = psFile;
    ui->currentTextStyle = PS_NO_TSTYLE;
    ui->currentDashStyle = PS_NO_DSTYLE;
    ui->currentWidth = PS_NO_WIDTH;
    ui->currentLStyle = PS_NO_LSTYLE;
    ui->title_family = tf;
    ui->title_size = ts;
    ui->label_family = laf;
    ui->label_size = las;
    ui->legend_family = lef;
    ui->legend_size = les;
    ui->axis_family = af;
    ui->axis_size = as;
      /* Roughly,  one-eighth a point in devs */
    ui->baseWidth = rd( VDPI / POINTS_PER_INCH * BASE_WIDTH );
	ui->dev_info= outInfo;
	ui->truncated= 0;

    PS_scale = VDPI / MICRONS_PER_INCH;

	_Font_Width_Estimator= (use_gsTextWidth)? 1 : Font_Width_Estimator;
/* 	_Font_Width_Estimator= Font_Width_Estimator;	*/

    outInfo->dev_flags = 0;
    outInfo->area_w = rd( ((double) width) * PS_scale );
    outInfo->area_h = rd( ((double) height) * PS_scale );
    ui->height_devs = outInfo->area_h;
	width_devs= outInfo->area_w;
    bdr_pad= outInfo->bdr_pad = rd( PS_BDR_PAD * VDPI );
    outInfo->axis_pad = rd( PS_AXIS_PAD * VDPI );
    outInfo->legend_pad = rd( PS_LEG_PAD * VDPI );
    outInfo->tick_len = rd( PS_TICK_LEN * VDPI );
    outInfo->errortick_len = rd( PS_TICK_LEN * VDPI/1.414 );

    /* Font estimates */
    font_size = as * INCHES_PER_POINT * VDPI;
    outInfo->axis_height = rd( font_size );
    outInfo->axis_width = rd( font_size * _Font_Width_Estimator );

    font_size = les * INCHES_PER_POINT * VDPI;
    legend_height= outInfo->legend_height = rd( font_size );
    legend_width= outInfo->legend_width = rd( font_size * _Font_Width_Estimator );

    font_size = las * INCHES_PER_POINT * VDPI;
    outInfo->label_height = rd( font_size );
    outInfo->label_width = rd( font_size * _Font_Width_Estimator );

    font_size = ts * INCHES_PER_POINT * VDPI;
    outInfo->title_height = rd( font_size );
    outInfo->title_width = rd( font_size * _Font_Width_Estimator );

    outInfo->max_segs = PS_MAX_SEGS;

	axis_height= outInfo->axis_height;
	if( win_info->ValCat_XFont && win_info->ValCat_X && win_info->ValCat_X_axis ){
		axis_height= MAX( axis_height, CustomFont_psHeight( win_info->ValCat_XFont ) );
	}
    outInfo->xname_vshift = (int)(outInfo->bdr_pad* 2 + axis_height + outInfo->label_height+ 0.5);
	if( win_info->ValCat_X_axis && win_info->ValCat_X && win_info->axisFlag ){
		outInfo->xname_vshift+= (abs(win_info->ValCat_X_levels)- 1)* axis_height;
	}

	outInfo->polar_width_factor= 3;
	outInfo->axis_width_factor= PS_AXIS_WBASE * ui->baseWidth;
	outInfo->zero_width_factor= PS_ZERO_WBASE * ui->baseWidth;
	outInfo->var_width_factor= PS_DATA_WBASE * ui->baseWidth;
	outInfo->mark_size_factor= PS_MARK* ui->baseWidth;

    outInfo->xg_text = psText;
    outInfo->xg_clear = psClear;
    outInfo->xg_seg = psSeg;
    outInfo->xg_rect = psRect;
	outInfo->xg_polyg= psPolyg;
    outInfo->xg_arc = psArc;
    outInfo->xg_dot = psDot;
    outInfo->xg_end = psEnd;
	outInfo->xg_silent= (int (*)()) psSilent;
	outInfo->xg_CustomFont_width= CustomFont_psWidth;
	outInfo->xg_CustomFont_height= CustomFont_psHeight;
	CustomFont_psWidth( NULL );
	CustomFont_psHeight( NULL );

    outInfo->user_state = (char *) ui;
	outInfo->user_ssize= sizeof( struct userInfo);
	ui->silent= False;

	theWin_Info= win_info;

	psThisRGB= NULL;

	psNw= MAX( 1, psNgraph_wide );
	psNh= MAX( 1, psNgraph_high );

	if( initFile ){
      /* Header */
	  int i= 0;
	  char *c;
	  extern Boolean PIPE_error;
		PS("%%!PS-Adobe-3.0");
		if( psEPS ){
			PS(" EPSF-3.0");
		}
		PS( "\n" );
		errno= 0;
		fflush( psFile );
		if( errno== EPIPE || PIPE_error ){
		  extern FILE *NullDevice;
			ui->psFile= psFile= NullDevice;
		}

		gethostname( hnam, 80);
		getdomainname( dnam, 80);
		{ struct passwd *pwd;
			if( (pwd= getpwuid( getuid())) ){
				linfo= concat( pwd->pw_gecos, " <", pwd->pw_name, "@", hnam, dnam, ">", NULL );
			}
			else{
				linfo= concat( getlogin(), "@", hnam, dnam, NULL );
			}
		}
		if( (c= win_info->graph_titles) ){
			if( !*c && win_info->graph_titles_length> 0 ){
				c++;
				title[i++]= '?';
			}
			while( i< 127 && *c && *c!= '\n' ){
				title[i]= *c++;
				i++;
			}
		}
		title[i]= '\0';
#ifdef DEBUG
		{ char *c;
			if( (c= win_info->graph_titles) ){
				if( !*c && win_info->graph_titles_length> 0 ){
					c++;
				}
				fprintf( StdErr, "psInit(): \"%s\" -> \"%s\"\n", c, title );
			}
		}
#endif
		SetWindowTitle( win_info, 0);
		XFetchName( disp, win_info->window, &Window_Name );
		fprintf( psFile, "%%%%Creator: () (XGraph) (build %s [PS module "__DATE__" "__TIME__"])\n", XGraphBuildString );
/* 		fprintf( psFile, "%%%%For: (" );	*/
		PSprint_string( psFile, "%%For: (", "%%+ ", ") ()\n", linfo, True );
		if( strlen(Window_Name) ){
		  char *c;
			  /* Cut after the process-id: there the read_buffer is stored, which
			   \ can contain invalid characters for PS.
			   */
			if( (c= strstr( Window_Name, "$=")) ){
				while( *c && !isspace(*c) ){
					c++;
				}
				*c= '\0';
			}
			  /* 20050112: Put the printfilename into the first line of the title, the rest
			   \ in a continuation. This is necessary now that ps2pdf.preview takes the %%Title
			   \ text to construct an understandable/relevant filename.
			   */
			PSprint_string( psFile, "%%Title: (", "%%+ ", ")\n", pf, True );
			PSprint_string( psFile, "%%+ (", "%%+ ", " ", Window_Name, True );
			PSprint_string( psFile, NULL, "%%+ ", ")\n", title, True );
			sprintf( ui->JobName, "%s - %s", linfo, (title[0])? title : Window_Name);
			STRINGCHECK( ui->JobName, sizeof(ui->JobName) );
		}
		else{
			PSprint_string( psFile, "%%Title: (", "%%+ ", ")\n", pf, True );
			PSprint_string( psFile, "%%+ (XGraph Plot ", "%%+ ", ")\n", title, True );
			sprintf( ui->JobName, "%s - %s", linfo, (title[0] && title[1])? title : "XGraph" );
			STRINGCHECK( ui->JobName, sizeof(ui->JobName) );
		}
		{ char *c= &PrintTime[strlen(PrintTime)-1];
			if( *c== '\n' ){
				*c= '\0';
			}
			fprintf( psFile, "%%%%CreationDate: (%s) ()\n", PrintTime);
		}
		if( psDSC ){
			if( ps_previous_dimensions ){
				ps_previous_dimensions= False;
				fprintf( ui->psFile,
					"%% %%%%Dimensions and pagecount taken from discarded dump of the same data:\n"
					"%%%%BoundingBox: %d %d %d %d\n",
					rd(ps_hbbox[0]), rd(ps_hbbox[1]),
					rd( ps_hbbox[2]), rd(ps_hbbox[3])
				);
				fprintf( ui->psFile, "%%%%HiResBoundingBox: %g %g %g %g\n",
					ps_hbbox[0], ps_hbbox[1],
					ps_hbbox[2], ps_hbbox[3]
				);
				OUT( ui->psFile, "%%%%Pages: %d\n", ps_pages );
				OUT(psFile, "%%%%PageOrientation: %s\n", (orient)? "Landscape" : "Portrait" );
			}
			else{
				fprintf( ui->psFile, "%%%%BoundingBox: (atend)\n");
				fprintf( ui->psFile, "%%%%HiResBoundingBox: (atend)\n");
				OUT( psFile, "%%%%Pages: (atend)\n" );
			}
			OUT( psFile, "%%%%PageOrder: Ascending\n" );
		}
			Write_ps_comment(psFile);
		{ char *CommandLine;
		  int j;
		  extern char **Argv, ExecTime[], *InFilesTStamps;
		  extern int Argc;
			fprintf( psFile, "%% %%%%Command: (" );
			if( (CommandLine= cgetenv( "XGRAPHCOMMAND" )) ){
				fprintf( psFile, "%s", CommandLine );
			}
			if( False ){
			  char *c= rindex( Argv[0], '/' );
				fprintf( psFile, "\n%% %%%%       : (%s ", (c)? &c[1] : Argv[0] );
				for( j= 1; j< Argc; j++ ){
					fprintf( psFile, " %s", Argv[j] );
				}
			}
			fputs( ")\n", psFile );
			fprintf( psFile, "%% %%%%Executed d.d. (%s)\n", ExecTime );
			fprintf( psFile, "%% %%%%Input files: (%s)\n", InFilesTStamps );
			if( win_info->version_list ){
				OUT( psFile, "%% %%%%Version info:\n" );
				psLineLength= 0;
				PSprint_string( psFile, "% %%+ ", "% %%+ ", "\n", win_info->version_list, True );
			}
			if( win_info->process.description ){
				OUT( psFile, "%% %%%%Processing info:\n" );
				psLineLength= 0;
				PSprint_string( psFile, "% %%+ ", "% %%+ ", "\n", win_info->process.description, True );
			}
			if( win_info->transform.description ){
				OUT( psFile, "%% %%%%Transformation info:\n" );
				psLineLength= 0;
				PSprint_string( psFile, "% %%+ ", "% %%+ ", "\n", win_info->transform.description, True );
			}
			if( (!win_info->raw_display && win_info->use_transformed) ){
				OUT( psFile, "%% %%%%Display mode: quick (precooked)\n" );
			}
			else if( (win_info->raw_display && win_info->raw_once>= 0) || (win_info->raw_once< 0 && win_info->raw_val) ){
				OUT( psFile, "%% %%%%Display mode: raw\n" );
			}
			else{
				OUT( psFile, "%% %%%%Display mode: cooked\n" );
			}
		}
		OUT( psFile, "%%%%DocumentNeededFonts:\n%%%%+ %s\n%%%%+ %s\n%%%%+ %s\n%%%%+ %s\n%%%%+ %s\n",
			wi->hard_devices[PS_DEVICE].dev_title_font, wi->hard_devices[PS_DEVICE].dev_legend_font,
			wi->hard_devices[PS_DEVICE].dev_label_font, wi->hard_devices[PS_DEVICE].dev_axis_font,
			PS_greek_font
		);
		PS( "%%%%DocumentNeededResources: procset Adobe_level2_AI5 1.2 0\n" );
		 /* The following AI command/comment should have only 1 '%', but there are things
		  \ in the setup that AI doesn't like...
		  \ The whole idea behind AI compatibility is that it *would* be nice to do grouping of
		  \ individual elements into things like: labelboxes, legendbox, composite markers, multiline
		  \ text,...
		  */
		PS( "%%%%AI5_FileFormat 3\n\n" );

			{ char *xpos[]= { "left", "centre", "right"}, *ypos[]= { "bottom", "centre", "top"};
			  char cbuf[2048];
				  extern int showpage;
				if( wi->print_orientation ){
					fprintf( psFile, "%% %%%% scale %g, posn %s,%s%s\n", wi->ps_scale, xpos[wi->ps_ypos], ypos[wi->ps_xpos], (showpage)? ",showpage" : "" );
				}
				else{
					fprintf( psFile, "%% %%%% scale %g, posn %s,%s%s\n", wi->ps_scale, xpos[wi->ps_xpos], ypos[wi->ps_ypos], (showpage)? ",showpage" : "" );
				}
				cbuf[0]= '\0';
				Collect_Arguments( wi, cbuf, 2048);
				if( *cbuf ){
					fputs( "% %%Arguments:\n", psFile ); psLineLength= 0;
					PSprint_string( psFile, "% %%+ ", "% %%+ ", "\n", cbuf, True );
					fprintf( psFile, "%% %%%%+ -maxWidth %g -maxHeight %g\n",
						wi->hard_devices[PS_DEVICE].dev_max_width, wi->hard_devices[PS_DEVICE].dev_max_height
					);
					fprintf( psFile, "%% %%%%+ %s -ps_scale %s -ps_xpos %d%s -ps_ypos %d%s -ps_eps%d\n",
						(wi->print_orientation)? "-Landscape" : "-Portrait",
						d2str( wi->ps_scale, NULL, NULL),
						wi->ps_xpos, (wi->ps_xpos== 0)? d2str( wi->ps_l_offset, " (%g) ", NULL) : "",
						wi->ps_ypos, (wi->ps_ypos== 0)? d2str( wi->ps_b_offset, " (%g) ", NULL) : "",
						psEPS
					);
					fprintf( psFile, "%% %%%%+ -PSm %g,%g\n\n", psm_base, psm_incr+ 1 );
					psLineLength= 0;
				}
			}
			{ extern int XGDump_AllWindows, XG_NUp_X, XG_NUp_Y, XG_NUp_aspect;
			  extern double XG_NUp_scale;
				if( XGDump_AllWindows && (XG_NUp_X> 1 || XG_NUp_Y> 1) && WindowListTail ){
					fprintf( psFile, "%% %%%% PS_NUp AllWindows setting: %d x %d", XG_NUp_X, XG_NUp_Y );
					if( XG_NUp_aspect ){
						fprintf( psFile, " 1:1" );
					}
					if( XG_NUp_scale!= 1 ){
						fprintf( psFile, " *%g", XG_NUp_scale );
					}
					fputc( '\n', psFile );
				}
				if( XGDump_AllWindows ){
					fprintf( psFile, "%% %%%% PS_NUp settings string: '%s'\n", XG_PS_NUp_buf );
				}
			}

		ps_l_offset= wi->ps_l_offset;
		ps_b_offset= wi->ps_b_offset;
		  /* 20040615: impose using the window-specific settings, instead of the mixture it was before! */
		ps_scale= wi->ps_scale;
		ps_xpos= wi->ps_xpos;
		ps_ypos= wi->ps_ypos;

		PS("%% %%%% XGraph PostScript output\n");
		PS("%% %%%% Rewritten by RenE J.V. Bertin 1991-[build date above]\n");
		PS("%% %%%% Universiteit Utrecht, College de France/CNRS\n");
		PS("%% %%%% Original program by\n");
		PS("%% %%%% Rick Spickelmier and David Harrison\n");
		PS("%% %%%% University of California, Berkeley\n");

		PS( "\n%%%%EndComments\n%%%%BeginProlog\n" );
		PS( "%%%%IncludeResource procset Adobe_level2_AI5 1.2 0\n");
		PS( "userdict begin/RJVB 21690 def/featurebegin{countdictstack RJVB[}bind def\n" );
		PS( "/featurecleanup{stopped{cleartomark dup RJVB eq{pop exit}if}loop\n" );
		PS( "countdictstack exch sub dup 0 gt{{end}repeat}{pop}ifelse}bind def end\n" );
		PS( "%%EndResource\n" );
		PS( "\n%%%%EndProlog\n" );
		PS( "%%%%BeginSetup\n" );

		if( PSSetupIncludes ){
			DoPSIncludes( psFile, "psInit()" );
		}

		if( psSetPage ){
		  double w= psSetPage_width, h= psSetPage_height, ww, hh;
			if( orient ){
				SWAP( w, h, double );
			}
			if( w> 0 ){
				ww= w / 2.54 * 72;
			}
			else{
				ww= PNT_WIDTH(width)* ps_scale * psSetWidth_corr_factor / 100;
			}
			if( h> 0){
				hh= h / 2.54 * 72;
			}
			else{
				hh= PNT_WIDTH(height)* ps_scale* psSetHeight_corr_factor/ 100;
			}
			fprintf( psFile, "%% A PageFeature command to define the currently used \"pagesize\"\n");
			PS( "featurebegin{\n" );
			PS( "%%BeginFeature: *PageSize Current\n" );
			  /* 20040611: include a set of correction factors that makes it possible "to cheat".
			   \ See in HO_ok_fun() for explanation.
			   */
			if( orient ){
				fprintf( psFile, "<</PageSize[%g %g]/ImagingBBox null>>setpagedevice\n",
					hh, ww
				);
			}
			else{
				fprintf( psFile, "<</PageSize[%g %g]/ImagingBBox null>>setpagedevice\n",
					ww, hh
				);
			}
			PS( "%%EndFeature\n" );
			PS( "}featurecleanup\n" );
		}

		PS( "\n%% save the toplevel state\n");
		PS( "gsave\n");
		PS( "\n");

		PS( "%% Some useful commands to know where we are\n");
		PS( "/bdf{bind def}bind def\n");
		PS( "/edf{exch def}bind def\n");
		PS( "/buf 128 string def\n" );
		if( !psEPS ){
			PS( "/jn{/statusdict where exch pop{statusdict exch /jobname exch put}if}bdf\n");
		}
		else{
			PS( "/jn{/msg edf msg = flush} def\n" );
		}
		PS( "/C{closepath}bdf\n" );
		PS( "/N{newpath}bdf\n" );
		PS( "/M{moveto}bdf\n" );
		PS( "/L{lineto}bdf\n" );
		PS( "/S{stroke}bdf\n" );
		PS( "%% From a Next printhandler:\n" );
		PS( "/__NXdef{1 index where{pop pop pop}{def}ifelse}bind def\n" );
		PS( "/__NXbdef{1 index where{pop pop pop}{bind def}ifelse}bind def\n" );
		PS( "%% idem: if ever we will allow CMYK printing...\n");
		PS( "/_rgbtocmyk\n" );
		PS( "{\n" );
		PS( "\t3{\n" );
		PS( "\t\t1 exch sub 3 1 roll\n" );
		PS( "\t} repeat\n" );
		PS( "\t3 copy 1 4 1 roll\n" );
		PS( "\t3{\n" );
		PS( "\t\t3 index 2 copy gt{\n" );
		PS( "\t\t\texch\n" );
		PS( "\t\t} if\n" );
		PS( "\t\tpop 4 1 roll\n" );
		PS( "\t} repeat\n" );
		PS( "\tpop pop pop\n" );
		PS( "\t4 1 roll\n" );
		PS( "\t3{\n" );
		PS( "\t\t3 index sub\n" );
		PS( "\t\t3 1 roll\n" );
		PS( "\t} repeat\n" );
		PS( "\t4 -1 roll\n" );
		PS( "} def\n" );
		PS( "/setcmykcolour{\n" );
		PS( "\t1.0 exch sub dup dup 6 -1 roll sub dup 0 lt{pop 0}if 5 1 roll\n" );
		PS( "\t4 -1 roll sub dup 0 lt{pop 0}if 3 1 roll exch sub dup 0 lt{pop 0}if setrgbcolor}__NXbdef\n" );
		PS( "/currentcmykcolour{currentrgbcolor 3{1.0 exch sub 3 1 roll}repeat 0}__NXbdef\n" );
		PS( "\t/printrgbcolour{ /trail edf /B edf /R edf /G edf\n" );
		PS( "\t\t(RGB: ) print R buf cvs print ( ) print G buf cvs print ( ) print B buf cvs print trail print\n" );
		PS( "\t} bdf\n" );
		PS( "\t/printcmykcolour{ /trail edf /K edf /Y edf /M edf /C edf\n" );
		PS( "\t\t(CMYK: ) print C buf cvs print ( ) print M buf cvs print ( ) print Y buf cvs print ( ) print\n" );
		PS( "\t\tK buf cvs print trail print\n" );
		PS( "\t} bdf\n" );
		PS( "\t/cmykcolour{3{1.0 exch sub 3 1 roll}repeat 0}__NXbdef\n" );
		PS( "\t/RGBtoCMYK{\n" );
		PS( "\t\t1.0 exch sub dup dup 6 -1 roll sub dup 0 lt{pop 0}if 5 1 roll\n" );
		PS( "\t\t4 -1 roll sub dup 0 lt{pop 0}if 3 1 roll exch sub dup 0 lt{pop 0}if (\\n) printrgbcolour}__NXbdef\n" );
		PS( "\t/CMYKtoRGBcheck{\n" );
		PS( "\t\t1.0 exch sub dup dup 6 -1 roll sub dup 0 lt{pop 0}if 5 1 roll\n" );
		PS( "\t\t4 -1 roll sub dup 0 lt{pop 0}if 3 1 roll exch sub dup 0 lt{pop 0}if cmykcolour (\\n) printcmykcolour}__NXbdef\n" );
		PS( "%% A verbose setrgbcolor that prints out RGB and CMYK values:\n" );
		PS( "/vsco{setrgbcolor currentrgbcolor ( == ) printrgbcolour currentcmykcolour (\\n) printcmykcolour}bdf\n" );

		fflush(psFile);

		psMessage( ui, ui->JobName );

		/* Definitions */

		PS( "%% DrawEllipse command pirated from XFig: draws in current fgColour\n");
		PS( "/$F2psDict 32 dict def $F2psDict begin $F2psDict /mtrx matrix put\n");
		PS( "/DrawEllipse { /endangle edf /startangle edf /yrad edf /xrad edf /y edf ");
		PS( "/x edf /savematrix mtrx currentmatrix def x y translate xrad yrad scale 0 0 1 startangle endangle ");
		PS( "arc savematrix setmatrix } def\n");
		PS( "end /$F2psBegin {$F2psDict begin /$F2psEnteredState save def} def /$F2psEnd {$F2psEnteredState restore end} def\n");

		if( pssoft_coloured ){
			PS( "%% Select colour or (psychometric) grayscale mode:\n" );
			OUT( psFile, "/coloured %d def\n", ps_coloured );
			OUT( psFile, "/sco{ /B edf /G edf /R edf\n" );
			OUT( psFile, "\tcoloured 0 ne{\n" );
			OUT( psFile, "\t\tR G B setrgbcolor\n" );
			OUT( psFile, "\t} {\n" );
			OUT( psFile, "\t\t%% Psychometric gray has the same apparent intensity as the colour,\n"
			             "\t\t%% namely 30%% red, 59%% green and 11%% blue:\n"
			);
			OUT( psFile, "\t\tR 0.3 mul G 0.59 mul add B 0.11 mul add setgray\n" );
			OUT( psFile, "\t} ifelse\n" );
			OUT( psFile, "} def\n" );
		}
		else{
			if( ps_coloured ){
				PS( "/sco{setrgbcolor}bdf\n" );
			}
			else{
				PS( "/sco{setgray}bdf\n" );
			}
		}
		if( pssoft_coloured || ps_coloured ){
			OUT( psFile, "/fRect{ /fB edf /fG edf /fR edf /filled edf /y4 edf /x4 edf /y3 edf /x3 edf /y2 edf /x2 edf /y1 edf /x1 edf\n" );
			OUT( psFile, "\tnewpath x1 y1 moveto x2 y2 lineto\n" );
			OUT( psFile, "\tx3 y3 lineto x4 y4 lineto closepath\n" );
			OUT( psFile, "\tfilled 0 ne{\n" );
			OUT( psFile, "\t\tfR fG fB sco gsave fill grestore fgColour\n" );
			OUT( psFile, "\t} if\n" );
			OUT( psFile, "\tstroke\n" );
			OUT( psFile, "} def\n" );
		}
		else{
			OUT( psFile, "/fRect{ /fG edf /filled edf /y4 edf /x4 edf /y3 edf /x3 edf /y2 edf /x2 edf /y1 edf /x1 edf\n" );
			OUT( psFile, "\tnewpath x1 y1 moveto x2 y2 lineto\n" );
			OUT( psFile, "\tx3 y3 lineto x4 y4 lineto closepath\n" );
			OUT( psFile, "\tfilled 0 ne{\n" );
			OUT( psFile, "\t\tfG sco gsave fill grestore fgColour\n" );
			OUT( psFile, "\t} if\n" );
			OUT( psFile, "\tstroke\n" );
			OUT( psFile, "} def\n" );
		}

		psFonts(psFile, wi);
		psMarks(psFile);

		OUT( psFile, "/black {%s sco} def\n", PSconvert_colour( psFile, black_pixel) );
		OUT( psFile, "/white {%s sco} def\n", PSconvert_colour( psFile, white_pixel) );
		OUT( psFile, "/normal {%s sco} def\n", PSconvert_colour( psFile, normPixel) );
		OUT( psFile, "/zeroColour {%s sco} def\n", PSconvert_colour( psFile, zeroPixel) );
		OUT( psFile, "/bgColour {%s sco} def\n", PSconvert_colour( psFile, bgPixel) );
		PS("black\n");

		  /* Set linecap to butt, and use rounded linejoin(t)s
		   \ and (to be sure), set the miter-limit to a nice not-too-low value.
		   */
		OUT( psFile, "0 setlinecap\n1 setlinejoin\n3 setmiterlimit\n" );

		psScale(psFile, width, height, orient);

/* 		psFonts(psFile, wi);	*/
/* 		psMarks(psFile);	*/

		PS( "\n%%%%EndSetup\n%%\n" );
		PS( "%% Main body begins here\n%%\n");

		if( psDSC ){
			OUT( psFile, "%%%%Page: %d %d\n", ps_page_nr, ps_page_nr );
			OUT( psFile, "(%%%%[ Page: %d ]%%%%) = flush\n", ps_page_nr );
			OUT(psFile, "%%%%PageOrientation: %s\n", (orient)? "Landscape" : "Portrait" );
			OUT( psFile, "%%%%PageFeatures:\n" );
			OUT( psFile, "%%+ *PageSize Current\n" );
			OUT( psFile, "\n" );
		}

		PS( "$F2psBegin\n");
		Write_ps_comment(psFile);
		fflush(psFile);
		ui->Printing= PS_PRINTING;
		legend_number= 0;
		stroked= 0;
		mXSet= mYSet= False;
		fg_called= 0;

		PSdrawCounter+= 1;
	}
	else{
		if( ps_comment[0] ){
			ps_comment_buf= concat2( ps_comment_buf, ps_comment, NULL );
			ps_comment[0]= '\0';
		}
	}
	newfile= True;
	ui->clear_all_pos= ftell(ui->psFile);
	psLineLength= 0;

	xfree( linfo );

    return 1;
}



static void psScale(psFile, width, height, orient)
FILE *psFile;			/* Output stream */
int width;			/* Output width  */
int height;			/* Output height */
int orient;			/* 1==landscape	*/
/*
 * This routine figures out how transform the basic postscript
 * transformation into one suitable for direct use by
 * the drawing primitives.  Two variables X-CENTRE-PLOT
 * and Y-CENTRE-PLOT determine whether the plot is centred
 * on the page.
 */
{

	Write_ps_comment( psFile );
    PS("%% Scaling information\n");
    PS("%%\n");
    PS("%% Change these if you would like to change the centring\n");
    PS("%% of the plot in either dimension\n");
	if( psNw> 1 || psNh> 1 ){
		OUT( psFile, "/X-CENTRE-PLOT 0 def\n" );
		OUT( psFile, "/Y-CENTRE-PLOT 0 def\n" );
	}
	else if( psEPS ){
		OUT( psFile, "/X-CENTRE-PLOT 0 def\n" );
		OUT( psFile, "/Y-CENTRE-PLOT %d def\n", (orient)? 2 : 0 );
	}
	else{
		OUT( psFile, "/X-CENTRE-PLOT %d def\n", ps_xpos );
		OUT( psFile, "/Y-CENTRE-PLOT %d def\n", ps_ypos );
	}
    PS("%%\n");

    /*
     * Determine page size
     */
    PS("%% Page size computation\n");
    PS("clippath pathbbox\n");
    PS("/page-height edf\n");
    PS("/page-width edf\n");
    PS("pop pop\n");
	PS( "%% sizes of the graph:\n" );
	OUT( psFile, "/Ngraph-wide %hu def\n", psNgraph_wide );
	OUT( psFile, "/Ngraph-high %hu def\n", psNgraph_high );
	if( orient ){
		graph_width= PNT_WIDTH( height );
		graph_height= PNT_WIDTH( width );
	}
	else{
		graph_width= PNT_WIDTH( width );
		graph_height= PNT_WIDTH( height );
	}
	OUT( psFile, "/graph-width %lg Ngraph-wide div def\n", graph_width );
	OUT( psFile, "/graph-height %lg Ngraph-high div def\n", graph_height );
	graph_width*= ps_scale/ 100.0;
	graph_height*= ps_scale/ 100.0;

	if( 1 /* psNw> 1 || psNh> 1 */ ){
	  double psX= ps_scale/ 100.0;
	  double psY= ps_scale/ 100.0;
		OUT( psFile, "/graph-scale-x %lf Ngraph-wide div def\n", psX );
		OUT( psFile, "/graph-scale-y %lf Ngraph-high div def\n", psY );
		PS( "/graph-scale graph-scale-x graph-scale-x mul graph-scale-y graph-scale-y mul add sqrt def\n" );
	}
/* 	else{	*/
/* 		OUT( psFile, "/graph-scale-x %lf def\n", ps_scale/ 100.0 );	*/
/* 		OUT( psFile, "/graph-scale-y %lf def\n", ps_scale/ 100.0 );	*/
/* 		OUT( psFile, "/graph-scale %lf def\n", ps_scale/ 100.0 );	*/
/* 	}	*/
	if( psEPS ){
		OUT( psFile, "/left-offset 0 def\n" );
		OUT( psFile, "/bottom-offset 0 def\n" );
	}
	else{
		OUT( psFile, "/left-offset %lf def\n", (ps_l_offset) );
		OUT( psFile, "/bottom-offset %lf def\n", (ps_b_offset) );
	}
	OUT( psFile, "/graph-x-idx %hu def\n", psGraph_x_idx );
	OUT( psFile, "/graph-y-idx %hu def\n", psGraph_y_idx );
	PS( "/x-idx graph-x-idx def\n");
	PS( "/y-idx Ngraph-high 1 sub graph-y-idx sub def\n");

	OUT(psFile, "%%Orientation: %s", (orient)? "Landscape" : "Portrait" );
/* 	if( psEPS ){	*/
/* 		OUT( psFile, "; obtained by swapping all X and Y co-ordinates!\n" );	*/
/* 		OUT( psFile, "/Landscape 0 def\n" );	*/
/* 	}	*/
/* 	else{	*/
		OUT( psFile, "\n/Landscape %d def\n", orient );
/* 	}	*/
    OUT(psFile, "Landscape 0 gt\n" );
    PS("{ %% landscape requested: Rotation required (X and Y trade places!)\n");
		PS("   90 rotate\n");
		PS("   0 page-width neg translate\n");
		PS("   %% Handle centring\n");
		PS("   X-CENTRE-PLOT 1 eq { %% Centre in y\n");
		PS("      page-height graph-height graph-scale-y mul sub 2 div\n");
		PS("   } {\n");
		PS("      X-CENTRE-PLOT 2 eq { %% top of page\n");
		PS("         page-height graph-height graph-scale-y mul sub\n");
		PS("      } { %% bottom of page\n");
		PS("        left-offset\n");
		PS("      } ifelse\n");
		PS("   } ifelse graph-height graph-scale-x mul x-idx mul Ngraph-high mul add\n");
		PS("   Y-CENTRE-PLOT 1 eq { %% Centre in x\n");
		PS("      page-width graph-width graph-scale-x mul sub 2 div\n");
		PS("   } {\n");
		PS("      Y-CENTRE-PLOT 2 eq { %% right of page\n");
		PS("         page-width graph-width graph-scale-x mul sub\n");
		PS("      } { %% left of page\n");
		PS("         bottom-offset\n");
		PS("      } ifelse\n");
		if( psSetWidth_corr_factor!= 1 ){
			PS("   } ifelse graph-width graph-scale-y mul y-idx mul Ngraph-wide mul add graph-width sub\n");
		}
		else{
			PS("   } ifelse graph-width graph-scale-y mul y-idx mul Ngraph-wide mul add\n");
		}
		PS("   translate\n");
    PS("} {\n%% No rotation - just handle centring\n");
		PS("   X-CENTRE-PLOT 1 eq { %% Centre in x\n");
		PS("      page-width graph-width graph-scale-x mul sub 2 div\n");
		PS("   } {\n");
		PS("       X-CENTRE-PLOT 2 eq { %% right of page\n");
		PS("         page-width graph-width graph-scale-x mul sub\n");
		PS("       } { %% left of page\n");
		PS("         left-offset\n");
		PS("       } ifelse\n");
		PS("   } ifelse graph-width graph-scale-x mul x-idx mul Ngraph-wide mul add\n");
		PS("   Y-CENTRE-PLOT 1 eq { %% Centre in y\n");
		PS("      page-height graph-height graph-scale-y mul sub 2 div\n");
		PS("   } {\n" );
		PS("      Y-CENTRE-PLOT 2 eq {%% top of page\n");
		PS("         page-height graph-height graph-scale-y mul sub\n");
		PS("      } { %% bottom of page\n");
		PS("         bottom-offset\n");
		PS("      } ifelse\n");
		PS("   } ifelse graph-height graph-scale-y mul y-idx mul Ngraph-high mul add\n");
		PS("   translate\n");
    PS("} ifelse\n");
	PS( "%% end orientation section\n");

    /*
     * Now: scaling.  We have points.  We want devs.
     */
    PSscale = POINTS_PER_INCH / VDPI;
	PS( "%% compensate a little bit\n");
	OUT( psFile, "%%0 %g translate\n", transl_compensation );
    PS("%% Set the scale\n");
    OUT(psFile, "%lg graph-scale-x mul %lg graph-scale-y mul scale\n", PSscale, PSscale);
}


static void psFontReEncode( FILE *psFile )
{
	PS("%% whether or not to reencode fonts so that the full latin-1252/iso8859-1 charset is available:\n" );
	OUT( psFile, "/reencode %d def\n", ps_font_reencode );
	PS("reencode 0 gt{\n" );
      /* Reencoding code from the StarOffice 5.1/Linux distribution. This gives nice results when printed,
	   \ but is not fully compatible with Illustrator.
	   */
#ifdef STAROFFICE_REENCODE
	PS("     /ski/ISOLatin1Encoding where{pop true}{false}ifelse def\n" );
	PS("     /reencodesmalldict 12 dict def\n" );
	PS("     /ReEncodeSmall { \n" );
	PS("          reencodesmalldict begin\n" );
	PS("               /newcodesandnames exch def\n" );
	PS("               /newfontname exch def\n" );
	PS("               /basefontname exch def\n" );
	PS("               /basefontdict basefontname findfont def\n" );
	PS("               /newfont basefontdict maxlength dict def\n" );
	PS("\n" );
	PS("               basefontdict\n" );
	PS("               {\n" );
	PS("                    exch dup /FID ne\n" );
	PS("                    { \n" );
	PS("                         dup /Encoding eq\n" );
	PS("                         { \n" );
	PS("                              ski\n" );
	PS("                              {\n" );
	PS("                                   exch pop\n" );
	PS("                                   ISOLatin1Encoding dup length array copy\n" );
	PS("                              }{\n" );
	PS("                                   exch dup length array copy\n" );
	PS("                              }\n" );
	PS("                              ifelse\n" );
	PS("                              newfont 3 1 roll put\n" );
	PS("                         }{\n" );
	PS("                              exch newfont 3 1 roll put\n" );
	PS("                         }\n" );
	PS("                         ifelse\n" );
	PS("                    }{\n" );
	PS("                         pop pop\n" );
	PS("                    }\n" );
	PS("                    ifelse\n" );
	PS("               } forall\n" );
	PS("\n" );
	PS("               newfont /FontName newfontname put\n" );
	PS("               newcodesandnames aload pop\n" );
	PS("               newcodesandnames length 2 idiv\n" );
	PS("               {\n" );
	PS("                    newfont /Encoding get 3 1 roll put\n" );
	PS("               } repeat\n" );
	PS("\n" );
	PS("               newfontname newfont definefont pop\n" );
	PS("          end\n" );
	PS("     } def\n" );
	PS("\n" );
	PS("     /changesvec [\n" );
	PS("          16#80 /euro\n" );
	PS("          16#82 /quotesinglbase\n" );
	PS("          16#83 /florin\n" );
	PS("          16#84 /quotedblbase\n" );
	PS("          16#85 /ellipsis\n" );
	PS("          16#86 /dagger\n" );
	PS("          16#87 /daggerdbl\n" );
	PS("          16#88 /circumflex\n" );
	PS("          16#89 /perthousand\n" );
	PS("          16#8a /Scaron\n" );
	PS("          16#8b /guilsinglleft\n" );
	PS("          16#8c /OE\n" );
	PS("          16#8e /zcaron\n" );
	PS("          16#91 /quoteleft\n" );
	PS("          16#92 /quoteright\n" );
	PS("          16#93 /quotedblleft\n" );
	PS("          16#94 /quotedblright\n" );
	PS("          16#95 /bullet\n" );
	PS("          16#96 /endash\n" );
	PS("          16#97 /emdash\n" );
	PS("          16#98 /tilde\n" );
	PS("          16#99 /trademark\n" );
	PS("          16#9a /scaron\n" );
	PS("          16#9b /guilsinglright\n" );
	PS("          16#9c /oe\n" );
	PS("          16#9e /zcaron\n" );
	PS("          16#9f /Ydieresis\n" );
	PS("     ] def\n" );
	PS("\n" );
	PS("     %% example:\n" );
	PS("     %% /Helvetica-Narrow-Bold /Helvetica-Narrow-Bold-L1 changesvec ReEncodeSmall\n" );
#else
	PS("     %%<oldfont> <newname> <newencoding> reencode_font\n" );
	PS("     /reencodedict 5 dict def\n" );
	PS("     /reencode_font { reencodedict begin\n" );
	PS("       /newencoding exch def\n" );
	PS("       /newname exch def\n" );
	PS("       /basefont exch def\n" );
	PS("       /basefontdict basefont findfont def\n" );
	PS("       /newfont basefontdict maxlength dict def\n" );
	PS("       basefontdict {\n" );
	PS("          exch dup dup /FID ne exch /Encoding ne and\n" );
	PS("            { exch newfont 3 1 roll put }\n" );
	PS("            { pop pop }\n" );
	PS("            ifelse\n" );
	PS("       } forall\n" );
	PS("       newfont /Encoding newencoding put\n" );
	PS("       newname newfont definefont pop\n" );
	PS("     end } def\n" );
	PS("     %% Latin1252/iso8859-1 encoding vector, with stubs (naXXX) at the places (XXX) that are empty.\n" );
	PS("     %% The '-' character (between the comma and the period; ascii 45) must be /hyphen\n" );
	PS("     %% and not /minus to ensure that Illustrator 8 will show something at that location! I don't know\n" );
	PS("     %% if this is a bug in Illustrator (printing goes OK with /minus).\n" );
	PS("     %% This could also be done by redefining ISOLatin1Encoding itself, which *should* in fact have /hyphen at 45 (whereas Latin1252 should have /hyphen)!\n" );
	PS("     /Latin1252Encoding [\n" );
	PS("          /.notdef /na001 /na002 /na003 /na004 /breve /dotaccent /na007 /ring /hungarumlaut /ogonek /caron\n");
	PS("          /dotlessi /na0014 /na0015 /na0016 /na0017 /na0018 /na0019 /na0020 /na0021 /na0022 /na0023 /na0024\n");
	PS("          /na0025 /fraction /fi /fl /Lslash /lslash /Zcaron /zcaron\n");
	PS("          /space /exclam /quotedbl /numbersign /dollar /percent /ampersand /quotesingle /parenleft /parenright\n");
	PS("          /asterisk /plus /comma /hyphen /period /slash\n");
	PS("          /zero /one /two /three /four /five /six /seven /eight /nine\n");
	PS("          /colon /semicolon /less /equal /greater /question /at\n");
	PS("          /A /B /C /D /E /F /G /H /I /J /K /L /M /N /O /P /Q /R /S /T /U /V /W /X /Y /Z\n");
	PS("          /bracketleft /backslash /bracketright /asciicircum /underscore /grave\n");
	PS("          /a /b /c /d /e /f /g /h /i /j /k /l /m /n /o /p /q /r /s /t /u /v /w /x /y /z\n");
	PS("          /braceleft /bar /braceright /asciitilde /na127 /Euro /na129\n");
	PS("          /quotesinglbase /florin /quotedblbase /ellipsis /dagger /daggerdbl /circumflex /perthousand /Scaron\n");
	PS("          /guilsinglleft /OE /na141 /na142 /na143 /na144 /quoteleft /quoteright /quotedblleft /quotedblright\n");
	PS("          /bullet /endash /emdash /tilde /trademark /scaron /guilsinglright /oe /na157 /na158 /Ydieresis\n");
	PS("          /nbspace /exclamdown /cent /sterling /currency /yen /brokenbar /section /dieresis /copyright /ordfeminine /guillemotleft\n");
	PS("          /logicalnot /sfthyphen /registered /macron /degree /plusminus /twosuperior /threesuperior /acute /mu /paragraph\n");
	PS("          /periodcentered /cedilla /onesuperior /ordmasculine /guillemotright /onequarter /onehalf /threequarters /questiondown\n");
	PS("          /Agrave /Aacute /Acircumflex /Atilde /Adieresis /Aring /AE /Ccedilla /Egrave /Eacute /Ecircumflex\n");
	PS("          /Edieresis /Igrave /Iacute /Icircumflex /Idieresis /Eth /Ntilde /Ograve /Oacute /Ocircumflex /Otilde\n");
	PS("          /Odieresis /multiply /Oslash /Ugrave /Uacute /Ucircumflex /Udieresis /Yacute /Thorn /germandbls\n");
	PS("          /agrave /aacute /acircumflex /atilde /adieresis /aring /ae /ccedilla /egrave /eacute /ecircumflex\n");
	PS("          /edieresis /igrave /iacute /icircumflex /idieresis /eth /ntilde /ograve /oacute /ocircumflex /otilde\n");
	PS("          /odieresis /divide /oslash /ugrave /uacute /ucircumflex /udieresis /yacute /thorn /ydieresis\n");
	PS("     ] readonly def\n" );
	PS("    \n" );
	PS("     %% example:\n" );
	PS("     %% /Helvetica-Narrow-Bold /Helvetica-Narrow-Bold-L1 Latin1252Encoding reencode_font\n" );
#endif
	PS("     %% /Helvetica-Narrow-Bold-Latin1252 /Helvetica-Narrow-Bold-L1 def\n" );
	PS("     %% The <fontname>-L1 font can then be used as any other standard font. In Acrobat, it will show up as the base\n" );
	PS("     %% font with a custom encoding. The font <fontname>-Latin1252 can be used as any other font, but without the leading\n" );
	PS("     %% '/' character (it is a variable after all...)\n" );
	PS("     %% NB!! Symbol fonts should not be reencoded. I don't (yet) know how to prevent that other than preventing it for /Symbol...\n" );
	PS("} {\n" );
	PS("     %% Don't reencode, but define variables that point to the desired fonts, such that the following code can be unconditional.\n" );
	PS("     %% /Helvetica-Narrow-Bold-Latin1252 /Helvetica-Narrow-Bold def\n" );
	PS("} ifelse\n" );
}

void reencode_font( FILE *psFile, char *font, int reencode )
{
	if( reencode && strcmp( font, "Symbol") && strcmp( font, "ZapfDingbats") ){
#ifdef STAROFFICE_REENCODE
		OUT( psFile, "     /%s /%s-L1 changesvec ReEncodeSmall\n", font, font );
#else
		OUT( psFile, "     /%s /%s-L1 Latin1252Encoding reencode_font\n", font, font );
#endif
		OUT( psFile, "     /%s-Latin1252 /%s-L1 def\n", font, font );
	}
	else{
		OUT( psFile, "     /%s-Latin1252 /%s def\n", font, font );
	}
}

static void psFonts( FILE *psFile, LocalWin *wi)
/*
 * Downloads code for drawing title and axis labels
 */
{
	Write_ps_comment( psFile );
    PS("%% Font Handling Functions\n");
    PS("%%\n");
	psFontReEncode( psFile );
	if( wi ){
		PS("reencode 0 gt{\n" );
		reencode_font( psFile, wi->hard_devices[PS_DEVICE].dev_axis_font, True );
		reencode_font( psFile, wi->hard_devices[PS_DEVICE].dev_legend_font, True );
		reencode_font( psFile, wi->hard_devices[PS_DEVICE].dev_label_font, True );
		reencode_font( psFile, wi->hard_devices[PS_DEVICE].dev_title_font, True );
		OUT( psFile, "     /%s-Latin1252 /%s def\n", PS_greek_font, PS_greek_font );
		PS("} {\n" );
		OUT( psFile, "     /%s-Latin1252 /%s def\n",
			wi->hard_devices[PS_DEVICE].dev_axis_font, wi->hard_devices[PS_DEVICE].dev_axis_font );
		OUT( psFile, "     /%s-Latin1252 /%s def\n",
			wi->hard_devices[PS_DEVICE].dev_legend_font, wi->hard_devices[PS_DEVICE].dev_legend_font );
		OUT( psFile, "     /%s-Latin1252 /%s def\n",
			wi->hard_devices[PS_DEVICE].dev_label_font, wi->hard_devices[PS_DEVICE].dev_label_font );
		OUT( psFile, "     /%s-Latin1252 /%s def\n", 
			wi->hard_devices[PS_DEVICE].dev_title_font, wi->hard_devices[PS_DEVICE].dev_title_font );
		OUT( psFile, "     /%s-Latin1252 /%s def\n", PS_greek_font, PS_greek_font );
		PS("} ifelse\n" );
	}

    PS("%%\n");
    PS("%% Function giving y-offset to centre of font\n");
    PS("%% Assumes font is set and uses numbers to gauge centre\n");
    PS("%%\n");
	PS("/cp {currentpoint} def\n");
	PS("/ps-old-font-offsets 0 def\n" );
    PS("/c-f	%% stack: curX curY string fontsize fontname => ---\n");
    PS("{\n");
    PS("   findfont \n");
    PS("   exch scalefont \n");
    PS("   setfont\n");
	PS("   /string edf\n" );
	PS("   /currentY edf\n");
	PS("   /currentX edf\n");
    PS("   newpath\n");
	PS("   %% 20010528: now passing the to-be-printed string as parameter; before, a string containing a single 0 was given (no descender)\n");
	PS("   %% the ps-old-font-offsets variable alters this.\n" );
    PS("   0 0 moveto ps-old-font-offsets 0 ne { (0) }{ string } ifelse true charpath flattenpath pathbbox\n");
    PS("   /top edf pop\n");
    PS("   /bottom edf pop\n");
    PS("   bottom top bottom top add 2 div\n");
    PS("   /centre-font-val edf \n");
    PS("   /upper-font-val edf \n");
    PS("   /lower-font-val edf\n");
	PS("   currentX currentY moveto\n");
    PS("} def\n");
	 /* This generates a font rotated over (angle) degrees. When used for y-name
	  \ rotated over 90 degrees, shift X over (upper-font-val add centre-font-val add) and Y over
	  \ (upper-font-val add) to get (approximately) correct alignment.
	  */
    PS("/c-rf	%% stack: curX curY string angle fontsize fontname => ---\n");
    PS("{\n");
    PS("   findfont \n");
    PS("   exch scalefont \n");
    PS("   setfont\n");
	PS("   /R edf\n");
	PS("   /string edf\n");
	PS("   /currentY edf\n");
	PS("   /currentX edf\n");
    PS("   newpath\n");
	PS("   %% 20010528: now passing the to-be-printed string as parameter; before, a string containing a single 0 was given (no descender)\n");
    PS("   0 0 moveto ps-old-font-offsets 0 ne { (0) }{ string } ifelse true charpath flattenpath pathbbox\n");
    PS("   /top edf pop\n");
    PS("   /bottom edf pop\n");
    PS("   bottom top bottom top add 2 div\n");
    PS("   /centre-font-val edf \n");
    PS("   /upper-font-val edf \n");
    PS("   /lower-font-val edf\n");
	PS("   %%%% RJVB: I have not yet understood around what point the rotation takes place. The following\n"
	   "   %%%% empirically determined translation values put the printed string with its upper-left corner\n"
	   "   %%%% at the requested co-ordinates (for upper-left justification, and 90deg rotation)\n" );
	PS("   /vertical-adjust-x {upper-font-val centre-font-val add} bdf\n" );
	PS("   /vertical-adjust-y {upper-font-val lower-font-val add} bdf\n" );
	PS("   %% 20010528: we want a measure of the string's height. The rotation must thus be done after obtaining those values, otherwise we'd get the width...\n");
	PS("   currentfont [R cos R sin R sin -1 mul R cos 0 0] makefont setfont\n" );
	PS("   currentX currentY moveto\n");
    PS("} def\n");
    PS("%%\n");
    PS("%% Justfication offset routines\n");
    PS("%%\n");
    PS("/centre-x-just	%% stack: (string) x y => (string) newx y\n");
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
    PS("/centre-y-just	%% stack: (string) x y => (string) x newy\n");
    PS("{\n");
    PS("   centre-font-val sub\n");
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
    PS("/j-s	%% stack: (string) x y just => ---\n");
    PS("{\n");
    PS("   dup 0 eq { pop centre-x-just centre-y-just 		} if\n");
    PS("   dup 1 eq { pop left-x-just centre-y-just		} if\n");
    PS("   dup 2 eq { pop left-x-just upper-y-just	 	} if\n");
    PS("   dup 3 eq { pop centre-x-just upper-y-just 		} if\n");
    PS("   dup 4 eq { pop right-x-just upper-y-just	 	} if\n");
    PS("   dup 5 eq { pop right-x-just centre-y-just 		} if\n");
    PS("   dup 6 eq { pop right-x-just lower-y-just	 	} if\n");
    PS("   dup 7 eq { pop centre-x-just lower-y-just  		} if\n");
    PS("   dup 8 eq { pop left-x-just lower-y-just	 	} if\n");
    PS("   moveto\n");
    PS("} def\n");
	PS("%% Show a string in standout. Must have a gsave/grestore around it!\n");
	if( pssoft_coloured || ps_coloured ){
		PS("/standout   %% stack: x y font size string teR,G,B soR,G,B\n"
			"{/soB edf /soG edf /soR edf /teB edf /teG edf /teR edf /string edf /size edf /fn edf /y edf /x edf\n"
		);
	}
	else{
		PS("/standout   %% stack: x y font size string tegray sogray\n"
			"{/sogray edf /tegray edf /string edf /size edf /fn edf /y edf /x edf\n"
		);
	}
	PS("   gsave\n");
	PS("      cp string size fn c-f normal\n");
	PS("      /standout-yshift-a centre-font-val 2 div def\n");
/* 	PS("      cp string size /Symbol c-f normal\n");	*/
	OUT(psFile, "      cp string size %s-Latin1252 c-f normal\n", PS_greek_font );
	PS("      /standout-yshift-b centre-font-val 2 div def\n");
	PS("      /standout-yshift standout-yshift-b standout-yshift-a gt {\n");
	PS("         standout-yshift-b\n");
	PS("      } {\n");
	PS("         standout-yshift-a\n");
	PS("      } ifelse def\n");
	PS("   grestore\n");
	PS("   /ex x string stringwidth pop add def\n");
	PS("   %% bottom of rectangle is estimated as y - centre-font-val/2\n");
	PS("   /yy y standout-yshift sub def\n");
	PS("   /ey yy size add def\n");
	PS("   gsave\n");
	PS("   newpath x yy moveto x ey lineto ex ey lineto ex yy lineto closepath\n");
	if( pssoft_coloured || ps_coloured ){
		PS("      soR soG soB sco eofill soR soG soB sco S\n");
	}
	else{
		PS("      sogray sco eofill sogray sco S\n");
	}
	PS("   grestore\n");
	if( pssoft_coloured || ps_coloured ){
		PS("   string teR teG teB sco show\n");
	}
	else{
		PS("   string tegray sco show\n");
	}
	PS("} def\n");
    PS("%%\n");
}


#ifndef PSMARKERSFILE
	char psMarkersFile[256]= "/usr/local/lib/Xgraph.psMarkers";
#else
	char psMarkersFile[256]= PSMARKERSFILE;
#endif

static void psColour( struct userInfo *ui, int colour, Pixel pixval )
{
	if( colour>= 0 ){
		pixval= AllAttrs[colour].pixelValue;
	}
	if( !fg_called || colour!= fg_ccol || pixval!= fg_cpix || (psThisRGB && psThisRGB!= fg_rgb) ){
/* 		if( colour< 0 ){	*/
			OUT( ui-> psFile, "/fgColour {%s sco} def\n", PSconvert_colour( ui->psFile, pixval) );
/* 		}	*/
/* 		else{	*/
/* 			OUT( ui-> psFile, "/fgColour {%s sco} def\n", PSconvert_colour( ui->psFile, AllAttrs[colour].pixelValue) );	*/
/* 		}	*/
		fg_ccol= colour;
		fg_cpix= pixval;
		fg_called= 1;
		fg_rgb= psThisRGB;
	}
	OUT( ui->psFile, "fgColour " );
	psLineLength= 9;
}

static void psMarks(psFile)
FILE *psFile;
/*
 * Writes out marker definitions
 */
{  FILE *markFile= NULL;
   char psMarkersHeader[]= "%% XGraph Marker definitions (%d)\n";
   char *mFName= getenv( "XG_PS_MARKSFILE" );
   char lname[1024];

	Write_ps_comment( psFile );
	errno= 0;
	if( mFName ){
		markFile= fopen( mFName, "r" );
		if( !markFile && mFName ){
			fprintf( StdErr, "psMarks(): $XG_PS_MARKSFILE=\"%s\": %s\n", mFName, serror() );
		}
	}
	else{
/* 		tildeExpand( lname, "~/.Preferences/.xgraph/Xgraph.psMarkers" );	*/
		snprintf( lname, sizeof(lname), "%s/Xgraph.psMarkers", PrefsDir );
		markFile= fopen( lname, "r" );
		mFName= lname;
	}
	if( !markFile ){
		if( errno!= ENOENT && mFName ){
			fprintf( StdErr, "psMarks(): $XG_PS_MARKSFILE=\"%s\": %s\n", mFName, serror() );
		}
		markFile= fopen( psMarkersFile, "r");
		mFName= psMarkersFile;
	}
	if( markFile ){
	  int i= 0;
	  char buf[256];
		if( fgets( buf, 255, markFile ) ){
			if( !sscanf( buf, psMarkersHeader, &i ) || i<= 0 ){
				fprintf( StdErr, "psMarks(): invalid header (%s= %d) in %s (%s)\n",
					buf, i, mFName, serror()
				);
				fflush( StdErr);
				goto internal_markers;
			}
			else{
				psMarkers= i;
				i= 0;
			}
		}
		else{
			fprintf( StdErr, "psMarks(): can't read from %s (%s)\n",
				mFName, serror()
			);
			fflush( StdErr);
			goto internal_markers;
		}
		OUT( psFile, "%% %s [%d markers]:\n", mFName, psMarkers );
		fputs( buf, psFile );
		while( fgets( buf, 255, markFile ) ){
			i++;
			fputs( buf, psFile );
		}
		fclose( markFile );
		if( debugFlag ){
			fprintf( StdErr, "psMarks(): copied %d markers (%d lines) from %s\n",
				psMarkers, i, mFName
			);
			fflush( StdErr );
		}
	}
	else{
internal_markers:;
		if( debugFlag ){
			fprintf( StdErr, "psMarks(): can't read from %s (%s)\n",
				mFName, serror()
			);
			fflush( StdErr);
		}
		OUT( psFile, psMarkersHeader, psMarkers );
		PS( "%% Internal markers\n");
/* 		PS("black\n");	*/
		PS("%% a blank, filled rectangle");
		PS("\n/M0 {/sizeY edf /sizeX edf /y edf /x edf\n");
		PS("newpath x sizeX sub y sizeY sub moveto\n");
		PS("sizeX sizeX add 0 rlineto 0 sizeY sizeY add rlineto\n");
		PS("0 sizeX sizeX add sub 0 rlineto closepath bgColour gsave fill grestore fgColour S} def\n");
		
		PS("\n%% a fgColour, filled rectangle");
		PS("\n/M1 {/sizeY edf /sizeX edf /y edf /x edf\n");
		PS("newpath x sizeX sub y sizeY sub moveto\n");
		PS("sizeX sizeX add 0 rlineto 0 sizeY sizeY add rlineto\n");
		PS("0 sizeX sizeX add sub 0 rlineto closepath fill} def\n");
		
		PS("\n%% a blank, filled circle");
		PS("\n/M2 {/sizeY edf /sizeX edf /y edf /x edf\n");
/* 		PS("newpath x y moveto x y sizeX sizeY 0 360 DrawEllipse bgColour gsave fill grestore fgColour S} def\n");	*/
		PS("newpath x y sizeX sizeY 0 360 DrawEllipse bgColour gsave fill grestore fgColour S} def\n");

		PS("\n%% a fgColour, filled circle");
		PS("\n/M3 {/sizeY edf /sizeX edf /y edf /x edf\n");
		PS("newpath x y moveto x y sizeX sizeY 0 360 DrawEllipse fill} def\n");

		PS("\n%% a blank, filled diamond");
		PS("\n/M4 {/sizeY edf /sizeX edf /y edf /x edf\n");
		PS("newpath x sizeX sub y moveto sizeX sizeY rlineto\n");
		PS("sizeX sizeY -1 mul rlineto sizeX -1 mul sizeY -1 mul rlineto\n");
		PS("closepath bgColour gsave fill grestore fgColour S} def\n");
		
		PS("\n%% a fgColour, filled diamond");
		PS("\n/M5 {/sizeY edf /sizeX edf /y edf /x edf\n");
/* 		PS("newpath x sizeX sub y moveto x y sizeY add lineto\n");	*/
/* 		PS("x sizeX add y lineto x y sizeY sub lineto\n");	*/
		PS("newpath x sizeX sub y moveto sizeX sizeY rlineto\n");
		PS("sizeX sizeY -1 mul rlineto sizeX -1 mul sizeY -1 mul rlineto\n");
		PS("closepath fill} def\n");

		PS("\n%% a blank, filled upwards triangle");
		PS("\n/M6 {/sizeY edf /sizeX edf /y edf /x edf\n");
		PS("%%/osize size 7.0710678119E-01 mul def\n");
		PS("/osizeX sizeX def /osizeY sizeY def\n");
/* 		PS("newpath x y sizeY add moveto x osizeX add y osizeY sub lineto\n");	*/
/* 		PS("x osizeX sub y osizeY sub lineto\n");	*/
		PS("newpath x y sizeY add moveto osizeX sizeY osizeY add -1 mul rlineto\n");
		PS("sizeX osizeX add -1 mul 0 rlineto\n");
		PS("closepath bgColour gsave fill grestore fgColour S} def\n");

		PS("\n%% a fgColour, filled upwards triangle");
		PS("\n/M7 {/sizeY edf /sizeX edf /y edf /x edf\n");
		PS("%%/osize size 7.0710678119E-01 mul def\n");
		PS("/osizeX sizeX def /osizeY sizeY def\n");
/* 		PS("newpath x y sizeY add moveto x osizeX add y osizeY sub lineto\n");	*/
/* 		PS("x osizeX sub y osizeY sub lineto\n");	*/
		PS("newpath x y sizeY add moveto osizeX sizeY osizeY add -1 mul rlineto\n");
		PS("sizeX osizeX add -1 mul 0 rlineto\n");
		PS("closepath fill} def\n");

		PS("\n%% a blank, filled downwards triangle");
		PS("\n/M8 {/sizeY edf /sizeX edf /y edf /x edf\n");
		PS("%%/osize size 7.0710678119E-01 mul def\n");
		PS("/osizeX sizeX def /osizeY sizeY def\n");
/* 		PS("newpath x y sizeY sub moveto x osizeX add y osizeY add lineto\n");	*/
/* 		PS("x osizeX sub y osizeY add lineto\n");	*/
		PS("newpath x y sizeY sub moveto osizeX sizeY osizeY add rlineto\n");
		PS("sizeX osizeX add -1 mul 0 rlineto\n");
		PS("closepath bgColour gsave fill grestore fgColour S} def\n");

		PS("\n%% a fgColour, filled downwards triangle");
		PS("\n/M9 {/sizeY edf /sizeX edf /y edf /x edf\n");
		PS("%%/osize size 7.0710678119E-01 mul def\n");
		PS("/osizeX sizeX def /osizeY sizeY def\n");
/* 		PS("newpath x y sizeY sub moveto x osizeX add y osizeY add lineto\n");	*/
/* 		PS("x osizeX sub y osizeY add lineto\n");	*/
		PS("newpath x y sizeY sub moveto osizeX sizeY osizeY add rlineto\n");
		PS("sizeX osizeX add -1 mul 0 rlineto\n");
		PS("closepath fill} def\n");

		PS("\n%% a blank, filled diabolo");
		PS("\n/M10 {/sizeY edf /sizeX edf /y edf /x edf\n");
/* 		PS("newpath x y moveto x sizeX sub y sizeY sub lineto\n");	*/
/* 		PS("x sizeX add y sizeY sub lineto closepath bgColour gsave fill grestore fgColour S\n");	*/
/* 		PS("newpath x y moveto x sizeX add y sizeY add lineto\n");	*/
/* 		PS("x sizeX sub y sizeY add lineto\n");	*/
		PS("newpath x y moveto sizeX -1 mul sizeY -1 mul rlineto\n");
		PS("sizeX 2 mul 0 rlineto closepath bgColour gsave fill grestore fgColour S\n");
		PS("newpath x y moveto sizeX sizeY rlineto\n");
		PS("sizeX -2 mul 0 rlineto\n");
		PS("closepath bgColour gsave fill grestore fgColour S} def\n");

		PS("\n%% a fgColour, filled diabolo");
		PS("\n/M11 {/sizeY edf /sizeX edf /y edf /x edf\n");
/* 		PS("newpath x y moveto x sizeX sub y sizeY sub lineto\n");	*/
/* 		PS("x sizeX add y sizeY sub lineto closepath fill\n");	*/
/* 		PS("newpath x y moveto x sizeX add y sizeY add lineto\n");	*/
/* 		PS("x sizeX sub y sizeY add lineto\n");	*/
		PS("newpath x y moveto sizeX -1 mul sizeY -1 mul rlineto\n");
		PS("sizeX 2 mul 0 rlineto closepath fill\n");
		PS("newpath x y moveto sizeX sizeY rlineto\n");
		PS("sizeX -2 mul 0 rlineto\n");
		PS("closepath fill} def\n");

		PS("\n%% a diagonal cross");
		PS("\n/M12 {/sizeY edf /sizeX edf /y edf /x edf\n");
		PS("newpath x sizeX sub y sizeY sub moveto x sizeX add y sizeY add lineto\n");
		PS("x sizeX sub y sizeY add moveto x sizeX add y sizeY sub lineto S} def\n");

		PS("\n%% a cross");
		PS("\n/M13 {/sizeY edf /sizeX edf /y edf /x edf\n");
		PS("newpath x sizeX sub y moveto x sizeX add y lineto\n");
		PS("x y sizeY add moveto x y sizeY sub lineto S} def\n");

		PS("\n%% a rectangle with a diagonal cross");
		PS("\n/M14 {/sizeY edf /sizeX edf /y edf /x edf\n");
		PS("x y sizeX sizeY M0\n");
		PS("newpath x y sizeX sizeY M12 } def\n");

#ifdef OLD_M15
		PS("\n/M15b {/sizeY edf /sizeX edf /y edf /x edf\n");
		PS("/sizex sizeX 1.2 div def\n");
		PS("/sizey sizeY 1.2 mul def\n");
		PS("newpath x sizex sub y moveto x y sizey add lineto\n");
		PS("x sizex add y lineto x y sizey sub lineto\n");
		PS("closepath bgColour gsave fill grestore fgColour S} def\n");
		
		PS("\n%% a rectangle/diamond (\"star\")");
		PS("\n/M15 {/sizeY edf /sizeX edf /y edf /x edf\n");
		PS("x y sizeX 1.3 div sizeY 1.3 div M0\n");
		PS("newpath x y sizeX sizeY M15b } def\n");
#else
		PS("\n%% a rectangle/diamond (\"star\")");
		PS("\n/M15 {/sizeY edf /sizeX edf /y edf /x edf\n");
		PS("/sizex sizeX def\n/sizey sizeY def\n" );
		PS("x y sizeX 1.3 div sizeY 1.3 div M0\n");
		PS("newpath x y sizex sizey M4 } def\n");
#endif
		
		PS("\n%% a circle with a cross");
		PS("\n/M16 {/sizeY edf /sizeX edf /y edf /x edf\n");
		PS("x y sizeX sizeY M2\n");
		PS("newpath x y sizeX sizeY M12 } def\n");

		psMarkers= internal_psMarkers= 17;
	}
	psLineLength= 0;
}


static char *psCookText( char *text )
{ char *Text= text, *plain_text;
  int brackets= 0;
	while( (Text= index( Text, '(')) ){
		brackets++;
		Text++;
	}
	Text= text;
	while( (Text= index( Text, ')')) ){
		brackets++;
		Text++;
	}
	Text= text;
	plain_text= (char*) calloc( 1,  strlen(Text) + brackets + 1 );
	if( plain_text ){
	  char *a= plain_text, *b= Text;
		while( *b){
			if( *b!= 0x7f ){
				switch( *b ){
					case '\\':
/* 						if( b[1]== '\\' )	*/
						if( !xtb_toggle_greek(b, Text) )
						{
/* 							*a++= *b++;	*/
							*a++= '\\';
							*a++= *b;
						}
						break;
					case ')':
					case '(':
						*a++= '\\';
						  /* no break!	*/
					default:	
						*a++= *b;
						break;
				}
			}
			b++;
		}
		*a= '\0';
	}
	else{
		plain_text= Text;
	}
	return( plain_text );
}

/* extern char *index();	*/

int ps_old_font_offsets= False;

static void psText( char *state, int x, int y, char *text, int just, int style, CustomFont *cfont)
/*
 * Draws text at the given location with the given justification
 * and style.
 */
{
    struct userInfo *ui = (struct userInfo *) state;
	int pass= 0, easy, memok= True;
	char *currentpoint= "0 0";
	double size;
	char *lfont, *font1, *Text, *greek_text, *plain_text;
	Boolean standout;
	int socolour= 0, fh;

	if( ui->silent || !text ){
		  /* 990615: we guarantee that we reset xg_text_highlight. So we must
		   \ also do that even when we produce no output at all
		   */
		xg_text_highlight= False;
		psThisRGB= NULL;
		return;
	}
	Write_ps_comment( ui->psFile );
	if( ps_old_font_offsets ){
		OUT( ui->psFile, "/ps-old-font-offsets %d def\n", ps_old_font_offsets );
	}
	if( cfont ){
		font1= cfont->PSFont;
		size= cfont->PSPointSize;
		fh= rd( size* INCHES_PER_POINT * VDPI);
		if( cfont->PSdrawcnt!= PSdrawCounter ){
			cfont->PSdrawcnt= PSdrawCounter;
			OUT( ui->psFile, "reencode 0 gt {\n" );
			reencode_font( ui->psFile, font1, cfont->PSreencode );
			OUT( ui->psFile, "} {\n" );
			OUT( ui->psFile, "     /%s-Latin1252 /%s def\n", font1, font1 );
			OUT( ui->psFile, "} ifelse\n" );
		}
	}
	else switch (style) {
		case T_AXIS:
			font1= ui->axis_family;
			size= ui->axis_size;
			fh= theWin_Info->dev_info.axis_height;
			break;
		case T_LEGEND:
			size= ui->legend_size;
			font1= ui->legend_family;
			fh= theWin_Info->dev_info.legend_height;
			break;
		case T_LABEL:
			size= ui->label_size;
			font1= ui->label_family;
			fh= theWin_Info->dev_info.label_height;
			break;
		case T_TITLE:
			size= ui->title_size;
			font1= ui->title_family;
			fh= theWin_Info->dev_info.title_height;
			break;
	}
	lfont= font1;
	easy= !strcmp( font1, PS_greek_font );

	if( xg_text_highlight ){
		standout= True;
		socolour= xg_text_highlight_colour;
	}
	else{
		standout= (text[0]== 0x01);
		socolour= 0;
	}
	Text= text;
	if( (plain_text= psCookText( Text ))== Text ){
		memok= False;
	}
	easy= 0;
	psColour( ui, -1, (use_textPixel)? textPixel : normPixel );
	while( Text && *Text ){
	  double tw, th;
		if( (greek_text= xtb_has_greek( &Text[1])) && !easy ){
			*greek_text= '\0';
		}
/* 		if( Text[0]== '\\' && Text[1]!= '\\' && !easy )	*/
		if( !easy && xtb_toggle_greek(Text, text) )
		{
			lfont= ( lfont== font1 )? PS_greek_font  : font1;
			Text++;
		}

		if( _use_gsTextWidth ){
			tw= gsTextWidth( theWin_Info, Text, style, cfont);
		}
		else{
			  /* stupid estimation:	*/
			tw= strlen(Text)* rd(ui->current_size* INCHES_PER_POINT * VDPI);
		}

		ui->current_family= lfont;
		ui->current_size= size;
		if( just== T_VERTICAL ){
			if( debugFlag ){
				OUT( ui->psFile, "%% \"%s\" at %gpt, vertical\n", ui->current_family, ui->current_size );
			}
			OUT(ui->psFile, "%s (%s) 90 %lg %s-Latin1252 c-rf normal\n",
				currentpoint, plain_text,
				ui->current_size * INCHES_PER_POINT * VDPI, ui->current_family
			);
			th= tw;
			tw= fh;
		}
		else{
			if( debugFlag ){
				OUT( ui->psFile, "%% \"%s\" at %gpt\n", ui->current_family, ui->current_size );
			}
			OUT(ui->psFile, "%s (%s) %lg %s-Latin1252 c-f normal\n",
				currentpoint, plain_text,
				ui->current_size * INCHES_PER_POINT * VDPI, ui->current_family
			);
			th= fh;
		}
		ui->currentTextStyle = style;
		if( pass ){
			_psText( ui->psFile, Text, standout, font1, ui->current_size* INCHES_PER_POINT * VDPI, socolour, NULL, False, True, True );
		}
		else{
			  /* first time pass whole text to j-s (just-string) to
			   \ obtain justified coordinates; then print the portion
			   \ of the string to be printed
			   */
			if( !easy ){
				if( greek_text ){
					*greek_text= '\\';
				}
				if( just== T_VERTICAL ){
					OUT(ui->psFile, "(%s) %d vertical-adjust-x add %d vertical-adjust-y add %d j-s pop\n",
						plain_text, IX(x,y), IY(x,y), T_UPPERLEFT
					);
				}
				else{
					OUT(ui->psFile, "(%s) %d %d %d j-s pop\n", plain_text, IX(x,y), IY(x,y), just);
				}
				if( greek_text ){
					*greek_text= '\0';
				}
				if( standout ){
					OUT(ui->psFile, "gsave\n");
				}
				_psText( ui->psFile, Text, standout, font1, ui->current_size* INCHES_PER_POINT * VDPI, socolour, NULL, False, True, True );
			}
			else{
				if( just== T_VERTICAL ){
					OUT(ui->psFile, "(%s) %d vertical-adjust-x add %d vertical-adjust-y add %d j-s show\n",
						plain_text, IX(x,y), IY(x,y), T_UPPERLEFT
					);
				}
				else{
					OUT(ui->psFile, "(%s) %d %d %d j-s show\n", plain_text, IX(x,y), IY(x,y), just);
				}
/* 				OUT(ui->psFile, "(%s) %d %d %d j-s show\n", plain_text, IX(x,y), IY(y), just);	*/
			}
			{ int X, Y;
				if( just== T_VERTICAL ){
					X= y+ pass* tw;
					Y= x;
				}
				else{
					X= x;
					Y= y+ pass* th;
				}
				switch( just ){
					case T_CENTER:
						psbounds( X- tw/2, Y- th/2 );
						psbounds( X+ tw/2, Y+ th/2 );
						break;
					case T_LEFT:
						psbounds( X, Y- th/2 );
						psbounds( X+ tw, Y+ th/2 );
						break;
					case T_VERTICAL:
					case T_UPPERLEFT:
						psbounds( X, Y );
						psbounds( X+ tw, Y+ th );
						break;
					case T_TOP:
						psbounds( X- tw/2, Y );
						psbounds( X+ tw/2, Y+ th );
						break;
					case T_UPPERRIGHT:
						psbounds( X- tw, Y );
						psbounds( X, Y+ th );
						break;
					case T_RIGHT:
						psbounds( X- tw, Y- th/2 );
						psbounds( X, Y+ th/2 );
						break;
					case T_LOWERRIGHT:
						psbounds( X- tw, Y- th );
						psbounds( X, Y );
						break;
					case T_BOTTOM:
						psbounds( X- tw/2, Y- th );
						psbounds( X+ tw/2, Y );
						break;
					case T_LOWERLEFT:
						psbounds( X, Y- th );
						psbounds( X+ tw, Y );
						break;
				}
			}
		}
		pass+= 1;
		stroked= 0;
		currentpoint= "cp";
		if( greek_text && !easy ){
			*greek_text= '\\';
			Text= greek_text;
		}
		else{
			Text= NULL;
		}
    }
	xg_text_highlight= False;
	if( standout ){
		OUT( ui->psFile, "grestore\n");
	}
	if( ps_old_font_offsets ){
		ps_old_font_offsets= False;
		OUT( ui->psFile, "/ps-old-font-offsets %d def\n", ps_old_font_offsets );
	}
	if( memok ){
		xfree( plain_text );
	}
	psLineLength= 0;
	psThisRGB= NULL;
	return;
}

typedef struct gsTWLists{
	char *text;
	int style;
	char *font;
	double ptsize;
	long hash;
	double width;
} gsTWLists;

gsTWLists *gsTWList= NULL;
static int gsTWListLen= 0;
static int gsTW= 0, gsTWMax= 0;

void gsResetTextWidths( LocalWin *wi, int destroy )
{
	if( destroy && gsTWList ){
	  int i;
		for( i= 0; i< gsTWListLen; i++ ){
			xfree( gsTWList[i].text );
			xfree( gsTWList[i].font );
		}
		xfree( gsTWList );
		gsTWListLen= 0;
		gsTWMax= 0;
	}
	gsTW= 0;
	if( wi ){
		if( wi->textrel.used_gsTextWidth>= 0 ){
			wi->textrel.prev_used_gsTextWidth= wi->textrel.used_gsTextWidth;
		}
		wi->textrel.used_gsTextWidth= -1;
	}
}

static int gsFontCompare( LocalWin *wi, int nr, int style, CustomFont *cfont )
{ int r= 0;
  double size;
  char *font1;
	if( gsTWList[nr].style== style && gsTWList[nr].font ){
		if( cfont ){
			font1= cfont->PSFont;
			size= cfont->PSPointSize;
		}
		else switch (style) {
			case T_AXIS:
				font1= wi->hard_devices[PS_DEVICE].dev_axis_font;
				size= wi->hard_devices[PS_DEVICE].dev_axis_size;
				break;
			case T_LEGEND:
				font1= wi->hard_devices[PS_DEVICE].dev_legend_font;
				size= wi->hard_devices[PS_DEVICE].dev_legend_size;
				break;
			case T_LABEL:
				font1= wi->hard_devices[PS_DEVICE].dev_label_font;
				size= wi->hard_devices[PS_DEVICE].dev_label_size;
				break;
			case T_TITLE:
				font1= wi->hard_devices[PS_DEVICE].dev_title_font;
				size= wi->hard_devices[PS_DEVICE].dev_title_size;
				break;
		}
		if( gsTWList[nr].ptsize== size && strcmp(gsTWList[nr].font, font1)== 0 ){
			r= True;
		}
	}
	return(r);
}

static int gsFindWidth( LocalWin *wi, char *text, int style, CustomFont *cfont )
{ int i, hash= ascanf_hash(text, NULL);
	for( i= 0; i< gsTWMax; i++ ){
		if( gsTWList[i].hash== hash && gsFontCompare( wi, i, style, cfont ) &&
			gsTWList[i].text && strcmp( gsTWList[i].text, text)== 0
		){
			return(i);
		}
	}
	return(-1);
}

int gsTextWidth_Add( LocalWin *wi, char *text, int style, char *font1, double size, CustomFont *cfont, double width )
{
	if( gsTW>= gsTWListLen ){
	  int i, n= gsTWListLen* 2;
		if( !n ){
			n= 50;
		}
		if( (gsTWList= (gsTWLists*) XGrealloc( gsTWList, n* sizeof(gsTWLists) )) ){
			for( i= gsTWListLen; i< n; i++ ){
				gsTWList[i].text= NULL;
				gsTWList[i].font= NULL;
			}
			gsTWListLen= n;
		}
		else{
			gsTWListLen= 0;
		}
	}
	if( gsTWList ){
		xfree( gsTWList[gsTW].text );
		gsTWList[gsTW].text= strdup(text);
		gsTWList[gsTW].hash= ascanf_hash(text, NULL);
		gsTWList[gsTW].style= style;
		xfree(gsTWList[gsTW].font);
		if( cfont ){
			gsTWList[gsTW].font= strdup(cfont->PSFont);
			gsTWList[gsTW].ptsize= cfont->PSPointSize;
		}
		else{
			gsTWList[gsTW].font= strdup(font1);
			gsTWList[gsTW].ptsize= size;
		}
		gsTWList[gsTW].width= width;
		if( debugFlag ){
			fprintf( StdErr, "gsTextWidth(\"%s\")=%g (cached %d)\n", text, width, gsTW );
		}
		snprintf( ps_comment, sizeof(ps_comment), "gsTextWidth(\"%s\")=%g (cached %d)\n", text, width, gsTW );
		gsTW+= 1;
		gsTWMax+= 1;
		return(gsTWMax);
	}
	else{
		if( debugFlag ){
			fprintf( StdErr, "gsTextWidth(\"%s\")=%g not cached (%s)\n", text, width, serror() );
		}
#ifdef linux
		snprintf( ps_comment, sizeof(ps_comment), "gsTextWidth(\"%s\")=%g not cached (%s)\n", text, width, serror() );
#endif
		return(-1);
	}
}

double gsTextWidth( LocalWin *wi, char *text, int style, CustomFont *cfont)
{ int ngsTW= gsTW, level= 0, easy, memok= True, ok, increment= True, reencode= True;
  double size, width= -1;
  char *lfont, *font1, *Text, *greek_text, *plain_text;
  FILE *psFile= NULL;
  char comm[512];

	if( !text || !*text ){
		return(0);
	}
	if( gsTW< gsTWListLen && gsFontCompare( wi, gsTW, style, cfont) &&
		gsTWList[gsTW].text && strcmp( gsTWList[gsTW].text, text)== 0
	){
		width= gsTWList[gsTW].width;
		if( debugFlag ){
			fprintf( StdErr, "gsTextWidth(\"%s\")=%g (from cache %d)\n", text, width, gsTW );
#ifdef linux
			snprintf( ps_comment, sizeof(ps_comment), "gsTextWidth(\"%s\")=%g (from cache %d)\n", text, width, gsTW );
#endif
		}
		if( width< 0 && !wi->textrel.gs_batch ){
			increment= False;
			goto gs_getnow;
		}
		gsTW+= 1;
	}
	else if( (ngsTW= gsFindWidth( wi, text, style, cfont ))>= 0 ){
		width= gsTWList[ngsTW].width;
		if( debugFlag ){
			fprintf( StdErr, "gsTextWidth(\"%s\")=%g (from cache %d)\n", text, width, ngsTW );
#ifdef linux
			snprintf( ps_comment, sizeof(ps_comment), "gsTextWidth(\"%s\")=%g (from cache %d)\n", text, width, ngsTW );
#endif
		}
		if( width< 0 && !wi->textrel.gs_batch ){
			gsTW= ngsTW;
			increment= False;
			goto gs_getnow;
		}
		gsTW= MAX(gsTW, ngsTW)+ 1;
	}
	else{
		gsTW= gsTWMax;
gs_getnow:;
		if( !wi->textrel.gs_batch || wi->textrel.gs_batch_items== 0 ){
			sprintf( wi->textrel.gs_fn, "/tmp/XGPSsize%d-%d.ps", (int) wi, getpid() );
			psFile= fopen( wi->textrel.gs_fn, "wb");
		}
		else{
			psFile= fopen( wi->textrel.gs_fn, "a");
		}
		if( !psFile ){
			return(-1);
		}
		if( cfont ){
			font1= cfont->PSFont;
			size= cfont->PSPointSize;
			reencode= cfont->PSreencode;
		}
		else switch (style) {
			case T_AXIS:
				font1= wi->hard_devices[PS_DEVICE].dev_axis_font;
				size= wi->hard_devices[PS_DEVICE].dev_axis_size;
				break;
			case T_LEGEND:
				font1= wi->hard_devices[PS_DEVICE].dev_legend_font;
				size= wi->hard_devices[PS_DEVICE].dev_legend_size;
				break;
			case T_LABEL:
				font1= wi->hard_devices[PS_DEVICE].dev_label_font;
				size= wi->hard_devices[PS_DEVICE].dev_label_size;
				break;
			case T_TITLE:
				font1= wi->hard_devices[PS_DEVICE].dev_title_font;
				size= wi->hard_devices[PS_DEVICE].dev_title_size;
				break;
		}
		lfont= font1;
		easy= !strcmp( font1, PS_greek_font );

		Text= text;
		if( (plain_text= psCookText( Text ))== Text ){
			memok= False;
		}
		easy= 0;
		if( !wi->textrel.gs_batch || wi->textrel.gs_batch_items== 0 ){
			OUT( psFile, "%%!PS\n" );
			OUT( psFile, "/graph-scale %g def\n", wi->ps_scale/ 100.0 );
			OUT( psFile, "/buf 128 string def\n" );
			OUT( psFile, "%g graph-scale mul %g graph-scale mul scale\n", POINTS_PER_INCH/VDPI, POINTS_PER_INCH/VDPI );
			if( PSSetupIncludes ){
				DoPSIncludes( psFile, "gsTextWidth()" );
			}
			psFontReEncode( psFile );
			PS( "reencode 0 gt {\n" );
			OUT( psFile, "     /%s-Latin1252 /%s def\n", PS_greek_font, PS_greek_font );
			PS( "} {\n" );
			OUT( psFile, "     /%s-Latin1252 /%s def\n", PS_greek_font, PS_greek_font );
			PS( "} ifelse\n" );
		}
#ifdef DEBUG
		Write_ps_comment( psFile );
		strncat( ps_comment, text, 1023 );
		Write_ps_comment( psFile );
#endif
		if( !wi->textrel.gs_batch ){
			OUT( psFile, "(w=) print" );
		}
		while( Text && *Text ){
			if( (greek_text= xtb_has_greek( &Text[1])) && !easy ){
				*greek_text= '\0';
			}
/* 			if( Text[0]== '\\' && Text[1]!= '\\' && !easy )	*/
			if( !easy && xtb_toggle_greek(Text, text) )
			{
				lfont= ( lfont== font1 )? PS_greek_font  : font1;
				Text++;
			}
			  /* 20020317: this is still not very elegant: the font is probably reencode (many) more times than necessary. */
			if( strcmp( lfont, PS_greek_font ) && level== 0 ){
				PS( "\nreencode 0 gt {\n" );
				reencode_font( psFile, lfont, reencode );
				PS( "} {\n" );
				OUT( psFile, "     /%s-Latin1252 /%s def\n", font1, font1 );
				PS( "} ifelse\n" );
			}
			OUT( psFile, "\n\t%s-Latin1252 findfont %lg scalefont setfont\n", lfont, size * INCHES_PER_POINT * VDPI );
			_psText( psFile, Text, False, font1, size* INCHES_PER_POINT * VDPI, -1, NULL, False, True, True );
			  /* 20010528: without the pop this gives the height...	*/
			if( level ){
				OUT( psFile, " stringwidth pop add" );
			}
			else{
				OUT( psFile, " stringwidth pop" );
			}
			if( greek_text && !easy ){
				*greek_text= '\\';
				Text= greek_text;
			}
			else{
				Text= NULL;
			}
			level+= 1;
		}
		if( !wi->textrel.gs_batch ){
			OUT( psFile, " buf cvs print (\\n) print\n" );
			OUT( psFile, "%%%%EOF\n" );
			fclose( psFile );
			sprintf( comm, "gs -dBATCH -dNODISPLAY -quiet %s", wi->textrel.gs_fn );
			if( (psFile= popen(comm, "r")) ){
				if( (fscanf( psFile, "w=%lf", &width ))== 1 ){
					ok= True;
				}
				else{
					ok= False;
				}
				pclose( psFile );
			}
			else{
				ok= False;
			}
			if( ok ){
				if( increment ){
					gsTextWidth_Add( wi, text, style, font1, size, cfont, width );
				}
				else{
				  /* This is an "update" of a previously not determined width.	*/
					gsTWList[gsTW].width= width;
				}
			}
			else{
				width= -1;
			}
			if( memok ){
				xfree( plain_text );
			}
			unlink( wi->textrel.gs_fn );
			wi->textrel.gs_fn[0]= '\0';
		}
		else{
			OUT( psFile, " buf cvs print (,) print (%d) print (,) print\n", gsTW );
			fclose( psFile );
			  /* Store all relevant info that is known at this point	*/
			gsTextWidth_Add( wi, text, style, font1, size, cfont, -1 );
			wi->textrel.gs_batch_items+= 1;
		}
	}
	return(width);
}

int gsTextWidthBatch( LocalWin *wi )
{ int N= 0, n= 0, read= 0;
  FILE *psFile;
  char comm[512];
	if( use_gsTextWidth && wi && wi->textrel.gs_batch && (N= wi->textrel.gs_batch_items)> 0 ){
	  ALLOCA( result, double, N*2, res_size );
	  Sinc input;
		input.sinc.string= NULL;
		Sinc_string_behaviour( &input, NULL, 0,0, SString_Dynamic );
		Sflush( &input );
		psFile= fopen( wi->textrel.gs_fn, "a");
		if( result && psFile ){
			OUT( psFile, "(\\n) print\n" );
			OUT( psFile, "%%%%EOF\n" );
			fclose( psFile );
			sprintf( comm, "gs -dBATCH -dNODISPLAY -quiet %s", wi->textrel.gs_fn );
			if( (psFile= popen(comm, "r")) ){
			  char *d, buf[512];
				while( !feof(psFile) && !ferror(psFile) && (d= fgets( buf, sizeof(buf)/sizeof(char), psFile)) ){
					Sputs( buf, &input );
				}
				Sflush( &input );
				pclose( psFile );
			}
			unlink( wi->textrel.gs_fn );
			wi->textrel.gs_fn[0]= '\0';
			if( input.sinc.string ){
				n= N*2;
				if( fascanf2( &n, input.sinc.string, result, ',')> 0 ){
				  int i, j;
					for( i= 0; i< n; i+= 2 ){
						if( (j= result[i+1])<= gsTWMax ){
							gsTWList[j].width= result[i];
							read+= 1;
						}
					}
				}
				xfree( input.sinc.string );
			}
			else{
				fprintf( StdErr, "gsTextWidthBatch(): error reading from pipe `%s`: %s\n", comm, serror() );
			}
			wi->textrel.gs_batch_items= 0;
		}
		else{
			fprintf( StdErr, "gsTextWidthBatch(): error: %s\n", serror() );
		}
	}
	return( read );
}

int ps_transparent= 0;

void PS_set_ClearHere( char *state )
{ struct userInfo *ui = (struct userInfo *) state;
	ui->clear_all_pos= ftell( ui->psFile );
}

static void psClear(char *state, int x, int y, int w, int h, int use_colour, int colour)
{ struct userInfo *ui = (struct userInfo *) state;
  Boolean wipe;
	if( ui->silent ){
		psThisRGB= NULL;
		return;
	}
	if( x || y || w!= ui->dev_info->area_w || h!= ui->dev_info->area_h ){
		wipe= False;
	}
	else{
		wipe= True;
	}
	if( newfile ){
		  /* clear_all_pos is set at the end of psInit. Don't output anything that
		   \ should not get wiped after that!
		   */
/* 		ui->clear_all_pos= ftell( ui->psFile );	*/
		newfile= False;
	}
	else if( wipe ){
	  char fsemsg[256]= "", ftemsg[256]= "";
		if( ui->clear_all_pos >= 0 && fseek( ui->psFile, ui->clear_all_pos, SEEK_SET ) ){
			sprintf( fsemsg, "Error seeking on PS dump: %s", serror() );
		}
		Write_ps_comment( ui->psFile );
		fflush( ui->psFile );
		if( (ui->clear_all_pos = ftell(ui->psFile)) > 0 && ftruncate( fileno(ui->psFile), ui->clear_all_pos ) ){
			sprintf( ftemsg, "Error truncating PS dump to pos %ld: %s", ui->clear_all_pos, serror() );
		}
		if( *fsemsg || *ftemsg ){
			fprintf( StdErr, "Warning: %s; %s\n", fsemsg, ftemsg );
			OUT( ui->psFile, "%%%% Warning: %s; %s\n", fsemsg, ftemsg );
		}
		ui->truncated+= 1;
		OUT( ui->psFile, "%% File truncated %dx at position %d\n", ui->truncated, ui->clear_all_pos );
		psLineLength= 0;
		fg_called= 0;

		mXSet= mYSet= False;
	}
	Write_ps_comment( ui->psFile );
	if( !ps_transparent || !wipe ){
		if( use_colour ){
			OUT( ui->psFile, "gsave\n\t%s sco 0 setlinewidth N\n", PSconvert_colour( ui->psFile, AllAttrs[colour].pixelValue) );
		}
		else{
			OUT( ui->psFile, "gsave\n\tbgColour 0 setlinewidth N\n");
		}
		OUT( ui->psFile, "\t%d %d M %d %d L\n\t%d %d L %d %d L\n",
			IX(x,y), IY(x,y),
			IX(x+ w,y), IY(x+w,y),
			IX(x+ w,y+h), IY(x+w,y+ h),
			IX(x,y+h), IY(x,y+ h)
		);
		OUT( ui->psFile, "\tclosepath fill S\n");
		OUT( ui->psFile, "grestore\n");
		stroked= 1;
		psLineLength= 0;
	}
	psThisRGB= NULL;
}

int use_ps_LStyle= 0, ps_LStyle= 0;
double psdash_power= 0.75;
Boolean psSeg_disconnect_jumps= True;

/*ARGSUSED*/
static void psSeg(char *state, int ns, XSegment *seglist, double width, int style, int lappr, int colour, Pixel pixval, DataSet *set)
/*
 * Draws a number of line segments.  Grid lines are drawn using
 * light lines.  Variable lines (L_VAR) are drawn wider.
 */
{ struct userInfo *ui = (struct userInfo *) state;
  double newwidth;
  int i, Style, grestore= 0, moveto= True;
  static unsigned long s_list= 0L;
  static char called= 0;
  static XSegment *first_list;
  extern int debugFlag;

	if( ui->silent ){
		psThisRGB= NULL;
		return;
	}
	colour%= MAXATTR;
	if( !called ){
		first_list= seglist;
		called= 1;
	}
	if( seglist== first_list ){
		s_list= 0;
	}
	else{
		s_list+= 1;
	}
	Write_ps_comment( ui->psFile );
	Style= (use_ps_LStyle)? ps_LStyle : style;
/*     if ((Style != ui->currentLStyle) || (width != ui->currentWidth)) 	*/
    {
		  /* Outcomment width*= newwidth lines to return to wildtype
		   * xgraph behaviour. This phenotype draws a line with width 2
		   * twice as broad as the default line (w=1)
		   */
		switch (Style) {
			case L_POLAR:
				newwidth = 3;
				width= 3;
				psColour( ui, colour, pixval );
				grestore= 1;
				break;
			case L_AXIS:
				newwidth = PS_AXIS_WBASE * ui->baseWidth;
				width*= newwidth;
				if( colour< 0 ){
					PSU(" [] 0 setdash ");
					psColour( ui, colour, pixval );
				}
				else{
					PSU(" [] 0 setdash normal ");
				}
				break;
			case L_ZERO:
				newwidth = PS_ZERO_WBASE * ui->baseWidth;
				width*= newwidth;
				PSU(" gsave [] 0 setdash zeroColour ");
				grestore= 1;
				break;
			case L_VAR:
				if( !set || set->lineWidth >= 0 ){
					newwidth = PS_DATA_WBASE * ui->baseWidth;
					width*= newwidth;
				}
				psColour( ui, colour, pixval );
				break;
		}
/* 		ui->currentWidth = MAX(newwidth, width);	*/
		ui->currentWidth = width;
		ui->currentLStyle = Style;
		if( debugFlag){
			OUT(ui->psFile, "%%baseWidth=%d pswidth=%g linewidth=%g\n",
				ui->baseWidth, newwidth, width
			);
		}
		OUT(ui->psFile, " %g setlinewidth\n", ui->currentWidth);
    }
#ifdef OLD_LAPPR
 /* This is the more or less original code for generating linepatterns
  \ in PostScript. The new code below tries to generate patterns which
  \ are as close to their screen-counterparts as possible, using the same
  \ "pixel-scale-factor" as for the width of the lines.
  */
    if( /* (lappr != ui->currentDashStyle) && */ (style == L_VAR)) {
		if (lappr == 0) {
			PSU(" [] 0 setdash ");
		} else {
		  extern double pow();
		  double __lappr, _lappr= ((double) lappr) * BASE_DASH * VDPI;
			if( _lappr< 200){
				OUT(ui->psFile, " [%lg] 0 setdash ", _lappr);
			}
			else{
				_lappr= pow((double)lappr, psdash_power ) * BASE_DASH * VDPI;
				__lappr= pow((double)lappr, psdash_power * psdash_power ) * BASE_DASH * VDPI;
				OUT(ui->psFile, " [%lg %lg %lg %lg] 0 setdash ",
					_lappr, __lappr, __lappr, __lappr
				);
			}
		}
		ui->currentDashStyle = lappr;
    }
#else
	lappr%= MAXATTR;
    if( (style == L_VAR) ){
		if (lappr == 0) {
			PSU(" [] 0 setdash ");
		} else {
		  int i;
		  char *sep= "";
			if( debugFlag && ui->currentDashStyle!= lappr ){
				OUT( ui->psFile, "%% dash #%d = \"", lappr );
				for( i= 0; i< AllAttrs[lappr].lineStyleLen; i++ ){
					OUT( ui->psFile, "%d", AllAttrs[lappr].lineStyle[i] );
				}
				OUT( ui->psFile, "\"\n");
			}
			PSU( " [");
			for( i= 0; i< AllAttrs[lappr].lineStyleLen; i++ ){
				OUT( ui->psFile, "%s%d", sep, AllAttrs[lappr].lineStyle[i]* ui->baseWidth* PS_DATA_WBASE );
				sep= " ";
			}
			PSU( "] 0 setdash ");
		}
		ui->currentDashStyle = lappr;
    }
#endif
    PSU(" N\n");
	if( debugFlag){
		OUT( ui->psFile, "%% seglist %lu\n", s_list );
	}
#if 0
	  /* 20050212: corrections for improved handling of interrupted traces with plot_interval>1:
	   \ use the new moveto flag to track whether an M is required instead of an L
	   \ and include seglist[0] into the main loop.
	   */
    OUT(ui->psFile, "  %d %d M ", IX(seglist[0].x1, seglist[0].y1), IY(seglist[0].x1, seglist[0].y1));
	if( debugFlag){
		OUT(ui->psFile, "  %d %d L %% seg %d\n", IX(seglist[0].x2, seglist[0].y2), IY(seglist[0].x2, seglist[0].y2), 0 );
	}
	else{
		OUT(ui->psFile, "  %d %d L\n", IX(seglist[0].x2, seglist[0].y2), IY(seglist[0].x2, seglist[0].y2));
	}
	moveto= False;
#endif
    for( i = 0;  i < ns;  i++) {
		if( moveto || !(seglist[i].x1 == seglist[i-1].x2 && seglist[i].y1 == seglist[i-1].y2) ) {
			if( psSeg_disconnect_jumps ){
			  /* We interrupt a non-continuous line	*/
				PSU( "   S %% non-continuous line\n");
				stroked= 1;
				moveto= True;
			}
			{
			  /* non-continuous lines are shown also as fully connected. Used when AllSets[idx].plot_interval>0
			   \ (and hence the seglist is not continuous)
			   */
				OUT(ui->psFile, "  %d %d %s ", IX(seglist[i].x1, seglist[i].y1), IY(seglist[i].x1, seglist[i].y1),
					(moveto)? "M" : "L"
				);
				moveto= False;
			}
		}
		if( seglist[i].x2== seglist[i].x1 && seglist[i].y2== seglist[i].y1 ){
#if 0
			  /* 20050212: zero length segment is how we interrupt a trace for e.g. a NaN value:
			   \ PostScript must stroke and output an M on the next segment (XDrawSegments does that
			   \ automagically).
			   */
			if( debugFlag){
				OUT(ui->psFile, "  %d %d L %% zero length seg %d\n",
					IX(seglist[i].x2, seglist[i].y2), IY(seglist[i].x2, seglist[i].y2), i);
			}
#else
			PSU( "   S %% zero length segment\n");
			stroked= 1;
			moveto= True;
#endif
		}
		else{
			if( debugFlag){
				OUT(ui->psFile, "  %d %d %s %% seg %d\n", IX(seglist[i].x2, seglist[i].y2), IY(seglist[i].x2, seglist[i].y2),
					(moveto)? "M" : "L", i
				);
			}
			else{
				OUT(ui->psFile, "  %d %d %s\n", IX(seglist[i].x2, seglist[i].y2), IY(seglist[i].x2, seglist[i].y2),
					(moveto)? "M" : "L"
				);
			}
			moveto= False;
		}
    }
    PSU(" S\n");
	stroked= 1;
/* 	called++;	*/
	if( grestore ){
		PSU( " grestore\n");
	}
	psLineLength= 0;
	psThisRGB= NULL;
}

static int SetShapePS( psUserInfo *ui, int style, int colour, int pixval, Pixel bg, int fill, double *lwidth, int lappr )
{ int Style, grestore= 0;
  double newwidth;
  extern int debugFlag;
	Style= (use_ps_LStyle)? ps_LStyle : style;
    {
		  /* Outcomment width*= newwidth lines to return to wildtype
		   * xgraph behaviour. This phenotype draws a line with width 2
		   * twice as broad as the default line (w=1)
		   */
		switch (Style) {
			case L_POLAR:
				newwidth = 3;
				*lwidth= 3;
				psColour( ui, colour, pixval );
				grestore= 1;
				break;
			case L_AXIS:
				newwidth = PS_AXIS_WBASE * ui->baseWidth;
				*lwidth*= newwidth;
				PSU(" [] 0 setdash normal ");
				break;
			case L_ZERO:
				newwidth = PS_ZERO_WBASE * ui->baseWidth;
				*lwidth*= newwidth;
				PSU(" gsave [] 0 setdash zeroColour ");
				grestore= 1;
				break;
			case L_VAR:
				newwidth = PS_DATA_WBASE * ui->baseWidth;
				*lwidth*= newwidth;
				psColour( ui, colour, pixval );
				break;
		}
		ui->currentWidth = *lwidth;
		ui->currentLStyle = Style;
		if( debugFlag){
			OUT(ui->psFile, "%%baseWidth=%d pswidth=%g linewidth=%g\n",
				ui->baseWidth, newwidth, *lwidth
			);
		}
		OUT(ui->psFile, " %g setlinewidth\n", ui->currentWidth);
		psLineLength= 0;
    }
	lappr%= MAXATTR;
    if( (style == L_VAR) ){
		if( lappr == 0 ){
			psLineLength+= PSU(" [] 0 setdash ");
		}
		else{
		  int i;
		  char *sep= "";
			if( debugFlag && ui->currentDashStyle!= lappr ){
				OUT( ui->psFile, "%% dash #%d = \"", lappr );
				for( i= 0; i< AllAttrs[lappr].lineStyleLen; i++ ){
					OUT( ui->psFile, "%d", AllAttrs[lappr].lineStyle[i] );
				}
				OUT( ui->psFile, "\"\n");
			}
			psLineLength+= PSU( " [");
			for( i= 0; i< AllAttrs[lappr].lineStyleLen; i++ ){
				psLineLength+= OUT( ui->psFile, "%s%d", sep, AllAttrs[lappr].lineStyle[i]* ui->baseWidth* PS_DATA_WBASE );
				sep= " ";
			}
			psLineLength+= PSU( "] 0 setdash ");
		}
		ui->currentDashStyle = lappr;
    }
	return( grestore );
}

static void psRect(char *state, XRectangle *specs, double lwidth, int style, int lappr, int colour, Pixel pixval, 
	int fill, int fill_colour, Pixel fill_pixval, DataSet *set
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
{ struct userInfo *ui = (struct userInfo *) state;
	static char called= 0;
	int grestore;
	extern int debugFlag;
	Pixel bg;

	if( ui->silent ){
		psThisRGB= NULL;
		return;
	}
	colour%= MAXATTR;
	fill_colour= fill_colour % MAXATTR;
	if( fill_colour< 0 ){
		bg= fill_pixval;
	}
	else{
		bg= AllAttrs[fill_colour].pixelValue;
	}
	if( !called ){
		called= 1;
	}
	Write_ps_comment( ui->psFile );
	grestore= SetShapePS( ui, style, colour, pixval, bg, fill, &lwidth, lappr );
	if( debugFlag){
		OUT( ui->psFile, "%% Rectangle %d,%d + %u,%u\n", specs->x, specs->y, specs->width, specs->height );
	}
	OUT( ui->psFile, "%d %d %d %d %d %d %d %d  %d %s fRect\n",
		IX(specs->x, specs->y), IY(specs->x, specs->y),
		IX(specs->x+ specs->width, specs->y), IY(specs->x+ specs->width, specs->y),
		IX(specs->x+ specs->width, specs->y+ specs->height), IY(specs->x+ specs->width, specs->y+ specs->height),
		IX(specs->x, specs->y+ specs->height), IY(specs->x, specs->y+ specs->height),
		fill, PSconvert_colour( ui->psFile, bg)
	);
	stroked= 1;
	called++;
	if( grestore ){
		PSU( " grestore\n");
	}
	psLineLength= 0;
	psThisRGB= NULL;
}

static void psPolyg(char *state, XPoint *specs, int N, double lwidth, int style, int lappr, int colour, Pixel pixval, 
	int fill, int fill_colour, Pixel fill_pixval, DataSet *set
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
{ struct userInfo *ui = (struct userInfo *) state;
  static char called= 0;
  int grestore, i;
  Pixel bg;
  extern int debugFlag;

	if( ui->silent || N<= 0 ){
		psThisRGB= NULL;
		return;
	}
	colour%= MAXATTR;
	fill_colour= fill_colour % MAXATTR;
	if( fill_colour< 0 ){
		bg= fill_pixval;
	}
	else{
		bg= AllAttrs[fill_colour].pixelValue;
	}
	if( !called ){
		called= 1;
	}
	Write_ps_comment( ui->psFile );
	grestore= SetShapePS( ui, style, colour, pixval, bg, fill, &lwidth, lappr );
	if( debugFlag){
		OUT( ui->psFile, "%% Polygon of %d points, first at %d,%d\n", N, specs->x, specs->y );
	}
	OUT( ui->psFile, "N %d %d M", IX(specs[0].x, specs[0].y), IY(specs[0].x, specs[0].y) );
	for( i= 1; i< N; i++ ){
		if( ((i-1) % 3)== 0 && i!= N-1 ){
			fputs( "\n\t", ui->psFile );
		}
		OUT( ui->psFile, " %d %d L", IX(specs[i].x, specs[i].y), IY(specs[i].x, specs[i].y) );
	}
	OUT( ui->psFile, " C" );
	if( fill ){
		fputc( '\n', ui->psFile );
		OUT( ui->psFile, "%s sco gsave fill grestore fgColour", PSconvert_colour( ui->psFile, bg ) );
	}
	OUT( ui->psFile, " S\n" );
	stroked= 1;
	called++;
	if( grestore ){
		PSU( " grestore\n");
	}
	psLineLength= 0;
	psThisRGB= NULL;
}


static void psArc(char *state, int x, int y, int rx, int ry, double la, double ha, double width,
	int style, int lappr, int colour, int pixval)
{
    struct userInfo *ui = (struct userInfo *) state;
    double newwidth;
	int Style, grestore= 0;
	extern int debugFlag;
	extern double radix, radix_offset;

	if( ui->silent ){
		psThisRGB= NULL;
		return;
	}
	colour%= MAXATTR;
	Write_ps_comment( ui->psFile );
	Style= (use_ps_LStyle)? ps_LStyle : style;
/*     if ((Style != ui->currentLStyle) || (width != ui->currentWidth)) 	*/
    {
		  /* Outcomment width*= newwidth lines to return to wildtype
		   * xgraph behaviour. This phenotype draws a line with width 2
		   * twice as broad as the default line (w=1)
		   */
		switch (Style) {
			case L_POLAR:
				newwidth = 3;
				width= 3;
				psColour( ui, colour, pixval );
				grestore= 1;
				break;
			case L_AXIS:
				newwidth = PS_AXIS_WBASE * ui->baseWidth;
				width*= newwidth;
				if( colour< 0 ){
					PSU(" [] 0 setdash ");
					psColour( ui, colour, pixval );
				}
				else{
					PSU(" [] 0 setdash normal ");
				}
				break;
			case L_ZERO:
				newwidth = PS_ZERO_WBASE * ui->baseWidth;
				width*= newwidth;
				PSU(" gsave [] 0 setdash zeroColour ");
				grestore= 1;
				break;
			case L_VAR:
				newwidth = PS_DATA_WBASE * ui->baseWidth;
				width*= newwidth;
				psColour( ui, colour, pixval );
				break;
		}
/* 		ui->currentWidth = MAX(newwidth, width);	*/
		ui->currentWidth = width;
		ui->currentLStyle = Style;
		if( debugFlag){
			OUT(ui->psFile, "%%baseWidth=%d pswidth=%g linewidth=%g\n",
				ui->baseWidth, newwidth, width
			);
		}
		OUT(ui->psFile, " %g setlinewidth\n", ui->currentWidth);
    }
#ifdef OLD_LAPPR
    if( /* (lappr != ui->currentDashStyle) && */ (style == L_VAR || style== L_POLAR) ){
		if (lappr == 0) {
			PSU(" [] 0 setdash ");
		} else {
		  extern double pow();
		  double __lappr, _lappr= ((double) lappr) * BASE_DASH * VDPI;
			if( _lappr< 200){
				OUT(ui->psFile, " [%lg] 0 setdash ", _lappr);
			}
			else{
				_lappr= pow((double)lappr, psdash_power ) * BASE_DASH * VDPI;
				__lappr= pow((double)lappr, psdash_power * psdash_power ) * BASE_DASH * VDPI;
				OUT(ui->psFile, " [%lg %lg %lg %lg] 0 setdash ",
					_lappr, __lappr, __lappr, __lappr
				);
			}
		}
		ui->currentDashStyle = lappr;
    }
#else
	lappr%= MAXATTR;
    if( (style == L_VAR) || style== L_POLAR ){
		if (lappr == 0) {
			PSU(" [] 0 setdash ");
		} else {
		  int i;
		  char *sep= "";
			if( debugFlag && ui->currentDashStyle!= lappr ){
				OUT( ui->psFile, "%% dash #%d = \"", lappr );
				for( i= 0; i< AllAttrs[lappr].lineStyleLen; i++ ){
					OUT( ui->psFile, "%d", AllAttrs[lappr].lineStyle[i] );
				}
				OUT( ui->psFile, "\"\n");
			}
			PSU( " [");
			for( i= 0; i< AllAttrs[lappr].lineStyleLen; i++ ){
				OUT( ui->psFile, "%s%d", sep, AllAttrs[lappr].lineStyle[i]* ui->baseWidth* PS_DATA_WBASE );
				sep= " ";
			}
			PSU( "] 0 setdash ");
		}
		ui->currentDashStyle = lappr;
    }
#endif
    PSU(" N\n");
	OUT( ui->psFile, "%d %d %d %d %g %g DrawEllipse S\n", IX(x, y), IY(x, y), rx, ry,
		((la+ radix_offset)/ radix)* 360, ((ha+ radix_offset)/ radix)* 360
	);
	if( grestore ){
		PSU( " grestore\n");
	}
	stroked= 1;
	psLineLength= 0;
	psThisRGB= NULL;
}

extern int SCREENXDIM(), SCREENYDIM();
extern int setNumber;

double ps_MarkSize_X( struct userInfo *ui, double setnr )
{ int i= (int) setnr;
  double s;

	if( i>= 0 && i< setNumber && !NaNorInf(AllSets[i].markSize) ){
		if( AllSets[i].markSize< 0 ){
			s= SCREENXDIM( (theWin_Info)? theWin_Info : ActiveWin, -AllSets[i].markSize );
		}
		else{
			s= ( PS_MARK* ui->baseWidth* AllSets[i].markSize );
		}
	}
	else{
		s= ( PS_MARK* ui->baseWidth* ( (int)(setnr/psMarkers)* psm_incr+ psm_base) );
	}
	return(s);
}

double ps_MarkSize_Y( struct userInfo *ui, double setnr )
{ int i= (int) setnr;
  extern int setNumber;
  double s;

	if( i>= 0 && i< setNumber && !NaNorInf(AllSets[i].markSize) ){
		if( AllSets[i].markSize< 0 ){
			s= SCREENYDIM( (theWin_Info)? theWin_Info : ActiveWin, -AllSets[i].markSize );
		}
		else{
			s= ( PS_MARK* ui->baseWidth* AllSets[i].markSize );
		}
	}
	else{
		s= ( PS_MARK* ui->baseWidth* ( (int)(setnr/psMarkers)* psm_incr+ psm_base) );
	}
	return(s);
}

/*ARGSUSED*/
static void psDot(state, x, y, style, type, colour, pixval, setnr, set)
char *state;    	/* state information */
int x,y;    		/* coord of dot */
int style;  		/* type of dot */
int type;   		/* dot style variation */
int colour;  		/* colour of dot */
Pixel pixval;
int setnr;
DataSet *set;
/*
 * Prints out a dot at the given location
 */
{
    struct userInfo *ui = (struct userInfo *) state;

	if( ui->silent ){
		psThisRGB= NULL;
		return;
	}
	colour%= MAXATTR;
	Write_ps_comment( ui->psFile );
    if (ui->currentDashStyle != PS_NO_DSTYLE) {
		OUT(ui->psFile, "[] 0 setdash ");
		ui->currentDashStyle = PS_NO_DSTYLE;
    }
    if (ui->currentWidth != PS_ZERO_WBASE * ui->baseWidth) {
		ui->currentWidth = PS_ZERO_WBASE * ui->baseWidth;
		OUT(ui->psFile, "%g setlinewidth ", ui->currentWidth);
    }
    
	psColour( ui, colour, pixval );
    switch (style) {
		case P_PIXEL:
			OUT(ui->psFile, "N %d %d M %d %d %d 0 360 arc fill\n",
				IX(x, y), IY(x, y), IX(x, y), IY(x, y), PS_PIXEL * ui->baseWidth);
			break;
		case P_DOT:
			OUT(ui->psFile, "N %d %d M %d %d %g 0 360 arc fill\n",
				IX(x, y), IY(x, y), IX(x, y), IY(x, y), PS_DOT * psm_base * ui->baseWidth);
			break;
		case P_MARK:
			OUT(ui->psFile, "%d %d %g %g M%d\n",
				IX(x, y), IY(x, y), ps_MarkSize_X( ui, setnr ), ps_MarkSize_Y( ui, setnr ), type % psMarkers );
			break;
    }
	psLineLength= 0;
	psThisRGB= NULL;
    return;
}


static int finished= 0;

void psMessage( void *state, char *message)
{  extern FILE *StdErr;
   struct userInfo *ui= (struct userInfo*) state;
   extern int debugFlag;
   char *m= message;
	XStoreName( disp, HO_Dialog.win, message );
	xtb_XSync( disp, False );
	if( ui->silent ){
		return;
	}
	Write_ps_comment( ui->psFile );
	OUT( ui->psFile, "(");
	_psText( ui->psFile, message, False, "", 0, 0, NULL, False, True, False);
	if( finished ){
		OUT( ui->psFile, " - finished");
	}
	OUT( ui->psFile, ")jn\n");
	fflush( ui->psFile );
	if( debugFlag){
		fprintf( StdErr, "psMessage(%s)\n", m);
		fflush( StdErr);
	}
	psLineLength= 0;
}

int PS_PrintComment= 0;
int ps_show_margins= 0;

extern double window_width, window_height;
extern double data_width, data_height;

static void psEnd(LocalWin *info)
{ extern char *comment_buf, *ShowLegends(), *XGstrdup(), *XGstrdup2();
  struct userInfo *ui = (struct userInfo *) info->dev_info.user_state;
  char *lbuf= NULL, *cbuf= NULL;
  extern int showpage;

	window_width= (maxX- minX)/ PS_scale;
	window_height= (maxY- minY)/ PS_scale;
	data_width= (theWin_Info->win_geo._XOppX- theWin_Info->win_geo._XOrgX)/ PS_scale;
	data_height= (theWin_Info->win_geo._XOppY- theWin_Info->win_geo._XOrgY)/ PS_scale;

	Write_ps_comment( ui->psFile );
	finished= 1;
	psMessage( ui, ui->JobName );
	if( PS_PrintComment && !psEPS ){
		if( theWin_Info ){
		  int i;
		  char hdr[128];
			cbuf= concat2( cbuf, comment_buf, (lbuf= ShowLegends( theWin_Info, False, -1)),
				theWin_Info->version_list, "\n", NULL
			);
			for( i= 0; i< setNumber; i++ ){
				if( AllSets[i].set_info && draw_set( theWin_Info, i) ){
					sprintf( hdr, "Info for set #%d:\n", i );
					cbuf= concat2( cbuf, hdr, AllSets[i].set_info, "\n", NULL );
				}
			}
		}
		else{
			cbuf= XGstrdup( comment_buf);
		}
	}
	fflush( ui->psFile );
	PSU( "$F2psEnd\n");
	PSU("grestore\n");
	OUT( ui->psFile, "%%%% show margin crosses at 20 units from the page-borders:\n" );
	OUT( ui->psFile, "%d 0 gt {\n", ps_show_margins );
	PSU( "\t10 20 M 30 20 L S\n" );
	PSU( "\t20 10 M 20 30 L S\n" );
	PSU( "\tpage-width -30 add page-height -20 add M page-width -10 add page-height -20 add L S\n" );
	PSU( "\tpage-width -20 add page-height -30 add M page-width -20 add page-height -10 add L S\n" );
	PSU( "} if\n" );
	if( !psEPS && (showpage || cbuf) ){
		PSU("showpage\n");
		ps_page_nr+= 1;
		fg_called= 0;
	}
	if( !psEPS && cbuf ){
	 char *substr= cbuf, *nextstr= index(substr, '\n');
	 int just, x, y= bdr_pad;
	 char *trailerf= NULL, *trailer= NULL;

		if( theWin_Info->raw_display ){
			trailerf= concat2( trailerf, ui->JobName, "; ",
				theWin_Info->YUnits, " vs. ", theWin_Info->XUnits, "; Page %d       ", NULL 
			);
			trailer= XGstrdup( trailerf );
		}
		else{
			trailerf= concat2( trailerf, ui->JobName, "; ",
				theWin_Info->tr_YUnits, " vs. ", theWin_Info->tr_XUnits, "; Page %d       ", NULL 
			);
			trailer= XGstrdup( trailerf );
		}
		if( psDSC ){
			OUT( ui->psFile, "%%%%Page: %d %d\n", ps_page_nr, ps_page_nr );
			OUT( ui->psFile, "(%%%%[ Page: %d ]%%%%) = flush\n", ps_page_nr );
			OUT( ui->psFile, "%%%%PageOrientation: Portrait\n" );
		}
		OUT( ui->psFile, "%lf %lf scale\n%d 0 translate\n", PSscale, PSscale, 2* bdr_pad );
		OUT( ui->psFile, "%% A4 sizes with this scaling: width=%g height=%g\n", 9916.67, 14033.3);
		ui->height_devs= 13033;
		do{
			if( nextstr && *nextstr== '\n' ){
				*nextstr= '\0';
			}
			if( (nextstr && xtb_is_leftalign(&nextstr[1])) || xtb_is_leftalign(&substr[0]) ){
			  /* This line, or the next begins with a whitespace, and is therefore
			   \ drawn left-justified.
			   */
				x= bdr_pad+ legend_width* 2;
				just= 2;
			}
			else{
				x= bdr_pad+ legend_width* 2+ (9916- 2* bdr_pad)/ 2;
				just= 3;
			}
			psText( info->dev_info.user_state, x, y, substr, just, T_LEGEND, NULL );
			y+= legend_height;
			if( nextstr ){
				*nextstr= '\n';
				substr= &nextstr[1];
				nextstr= index( substr, '\n');
			}
			if( nextstr && y>= ui->height_devs- bdr_pad* 5- legend_height ){
				y+= legend_height;
				sprintf( trailer, trailerf, ps_page_nr );
				psText( info->dev_info.user_state, bdr_pad+ legend_width* 2+ (9916- 2* bdr_pad)/2, y, trailer, 3, T_LEGEND, NULL  );
				OUT( ui->psFile, "showpage\n");
				y= bdr_pad;
				ps_page_nr+= 1;
				fg_called= 0;
				if( psDSC ){
					OUT( ui->psFile, "%%%%Page: %d %d\n", ps_page_nr, ps_page_nr );
					OUT( ui->psFile, "(%%%%[ Page: %d ]%%%%) = flush\n", ps_page_nr );
					OUT( ui->psFile, "%%%%PageOrientation: Portrait\n" );
				}
				OUT( ui->psFile, "%lf %lf scale\n%d 0 translate\n", PSscale, PSscale, 2* bdr_pad );
			}
		} while( nextstr );
		y= ui->height_devs- bdr_pad* 5;
		sprintf( trailer, trailerf, ps_page_nr );
		psText( info->dev_info.user_state, bdr_pad+ legend_width* 2+ (9916- 2* bdr_pad)/2, y, trailer, 3, T_LEGEND, NULL  );
		PSU( "showpage\n");
		psLineLength= 0;
		ps_page_nr+= 1;
		fg_called= 0;
		xfree( cbuf);
		xfree( lbuf );
		xfree( trailerf );
		xfree( trailer );
	}

	Write_ps_comment( ui->psFile );

	OUT( ui->psFile, "\n%%%%Trailer\n" );
	if( !psEPS && (theWin_Info->ps_xpos || theWin_Info->ps_ypos) ){
		fprintf( ui->psFile, "%%%%%% following boundaries are approximative at best and valid only for portrait, 100%%...\n" );
	}
	if( theWin_Info->print_orientation ){
	  double PSs= PSscale* ps_scale/ 100.0;
	  double mix= rd(minY*PSs+ ps_b_offset);
	  double dbmaxX, dbmaxY;
		if( psEPS ){
			mix= MAX( 45, mix );
			OUT( ui->psFile, "%%%%%%bbox minX is forced to min. 45; maxX is increased by 28\n" );
		}
		dbmaxX= maxY* PSs+ ps_b_offset+ ((psEPS)? 28 : 0);
		dbmaxY= maxX* PSs+ ps_l_offset;
		ps_hbbox[0]= mix, ps_hbbox[1]= minX* PSs+ ps_l_offset;
		ps_hbbox[2]= MAX( dbmaxX, graph_width), ps_hbbox[3]= MAX( dbmaxY, graph_height);
		if( psDSC ){
			fprintf( ui->psFile, "%%%%%%Correct bbox:\n%%%%BoundingBox: %d %d %d %d\n",
				rd(ps_hbbox[0]), rd(ps_hbbox[1]),
				rd( ps_hbbox[2]), rd(ps_hbbox[3])
			);
			fprintf( ui->psFile, "%%%%HiResBoundingBox: %g %g %g %g\n",
				ps_hbbox[0], ps_hbbox[1],
				ps_hbbox[2], ps_hbbox[3]
			);
		}
	}
	else{
	  double PSs= PSscale* ps_scale/ 100.0;
		ps_hbbox[0]= minX* PSs+ ps_l_offset, ps_hbbox[1]= minY* PSs+ ps_b_offset;
		ps_hbbox[2]= maxX* PSs+ ps_l_offset, ps_hbbox[3]= maxY* PSs+ ps_b_offset;
		if( psDSC ){
			fprintf( ui->psFile, "%%%%%%Correct bbox:\n%%%%BoundingBox: %d %d %d %d\n",
				rd(ps_hbbox[0]), rd(ps_hbbox[1]),
				rd(ps_hbbox[2]), rd(ps_hbbox[3])
			);
			fprintf( ui->psFile, "%%%%HiResBoundingBox: %g %g %g %g\n",
				ps_hbbox[0], ps_hbbox[1],
				ps_hbbox[2], ps_hbbox[3]
			);
		}
	}
	{ double YRange= (double)(info->win_geo._XOppY - info->win_geo._XOrgY);
	  double aspect= fabs( (info->win_geo.bounds._hiX - info->win_geo.bounds._loX) /
						(info->win_geo.bounds._hiY - info->win_geo.bounds._loY)
					);
		snprintf( ps_comment, sizeof(ps_comment)/sizeof(char),
			"Image dimensions WxH: total %gx%g; datawin %gx%g; aspect %s:%s\n",
			window_width/ 10000.0, window_height/ 10000.0, data_width/ 10000.0, data_height/ 10000.0,
			d2str( aspect, "%.2g", NULL),
			(YRange)? d2str( (double)(info->win_geo._XOppX - info->win_geo._XOrgX) / YRange, "%.2g", NULL) : "?"
		);
		fprintf( ui->psFile, "%% %s", ps_comment );
	}
	if( psDSC ){
		OUT( ui->psFile, "%%%%Pages: %d\n", (ps_pages= MAX(1,ps_page_nr- 1)) );
	}

	  /* Don't output the %%EOF as it will disable multi-page documents in gsview. */
	if( psEPS ){
		OUT( ui->psFile, "%%%%EOF\n" );
	}

	fflush( ui->psFile );

	showpage= 1;
	psLineLength= 0;
	ui->Printing= PS_FINISHED;
	xfree( ui );
	info->dev_info.user_state= NULL;
	theWin_Info= NULL;
	finished= 0;
	psThisRGB= NULL;
}

