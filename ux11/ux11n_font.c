/*
 * UX11 Utiltity Functions
 */

#include <stdio.h>
/* #include <strings.h>	*/
#include "ux11.h"
#include "ux11_internal.h"
#include <X11/Xatom.h>

#define MAX_FONTS	256

extern int debugFlag, debugLevel;

long ux11_find_font_start= 0;

/* Some routines to cache XFont data.	*/
typedef struct FontInfoList{
	char *name_pattern;
	int font_count;
	char **name_list;
	XFontStruct *font_list;
} FontInfoList;

static FontInfoList FontList[16];
static int LastFontList= 0;

static void FreeFontList()
{  int i;
	for( i= 0; i< LastFontList; i++ ){
		XFreeFontInfo( FontList[i].name_list, FontList[i].font_list, FontList[i].font_count );
	}
}

static char **FindFontListEntry( char *pat, int *font_count, XFontStruct **Font_Data )
{  int i;
	for( i= 0; i< LastFontList; i++ ){
		if( strcmp( FontList[i].name_pattern, pat)== 0 ){
			*font_count= FontList[i].font_count;
			*Font_Data= FontList[i].font_list;
			return( FontList[i].name_list );
		}
	}
	*font_count= 0;
	*Font_Data= NULL;
	return( NULL );
}

static AddFontListEntry( char *pat, int font_count, char **name_list, XFontStruct *Font_Data )
{ static char called= 0;
	if( LastFontList< sizeof(FontList)/sizeof(FontInfoList) ){
		FontList[LastFontList].name_pattern= pat;
		FontList[LastFontList].font_count= font_count;
		FontList[LastFontList].name_list= name_list;
		FontList[LastFontList].font_list= Font_Data;
		LastFontList+= 1;
	}
	if( !called ){
		atexit( FreeFontList );
		called= 1;
	}
}

long ux11_find_font(disp, scrn_num, pat, good_func, data_p, rtn_font, rtn_name)
Display *disp;			/* What display to examine */
int scrn_num;			/* Screen number           */
char *pat;			/* Font pattern            */
long (*good_func)();		/* Desirablity function    */
VOID_P data_p;			/* Data to function        */
XFontStruct **rtn_font;		/* Returned font           */
char **rtn_name;		/* Font name (returned)    */
/*
 * Locates an appropriate font.  Uses `good_func' to evaluate
 * the list of fonts from the server.  `good_func' has the
 * following form:
 *   long good_func(disp, scrn_num, font, data)
 *   Display *disp;
 *   int scrn_num;
 *   XFontStruct *font;
 *   VOID_P data;
 * This should return the desirability of the font (larger values
 * mean better visuals).  The `data' parameter is passed to
 * the function unchanged.  Returns a non-zero status if successful.
 */
{
    int font_count, i, new=0;
    long max_eval= ux11_find_font_start, eval;
    char **font_list, *chosen_name= NULL;
    XFontStruct *Font_Data, *font_data, *chosen_data= NULL;
	extern FILE *StdErr;
	char *mesg= "";

	if( debugFlag && debugLevel== -3 ){
		fprintf( StdErr, "\tux11_find_font(%s)...",
			pat
		);
		fflush( StdErr);
	}
	if( !(font_list= FindFontListEntry( pat, &font_count, &Font_Data)) ){
	  /* Haven't seen this one yet...	*/
		if( (font_list = XListFontsWithInfo(disp, pat, MAX_FONTS, &font_count, &Font_Data )) ){
			mesg= "new ";
			AddFontListEntry( pat, font_count, font_list, Font_Data);
			new= 1;
		}
	}
	if( debugFlag && debugLevel== -3 ){
		fprintf( StdErr, "... %d %sfonts:\n", font_count, mesg );
		for( i= 0; i< font_count && new; i++ ){
			fprintf( StdErr, "\t\t\t'%s'\n", font_list[i] );
		}
		fflush( StdErr);
	}
    if (font_count <= 0)
		return 0;

    for (i = 0;  i < font_count;  i++) {
		if( (font_data = &Font_Data[i]) ){
			if( (eval = (*good_func)(disp, scrn_num, font_data, data_p)) > max_eval){
				max_eval = eval;
				chosen_name = font_list[i];
				chosen_data = font_data;
				if( debugFlag && debugLevel== -3 ){
					fprintf( StdErr, "\t\tfont '%s', %ld mu, score= %ld\n",
						chosen_name, ux11_font_microm( disp, scrn_num, chosen_data), eval
					);
					fflush( StdErr);
				}
			}
		}
    }
    if( chosen_name && chosen_data ){
	    *rtn_name = strcpy(malloc((unsigned) (strlen(chosen_name)+1)), chosen_name);
	    *rtn_font= XLoadQueryFont( disp, chosen_name);
	}
    ux11_find_font_start= 0;
    return( max_eval );
}



typedef struct font_size_defn {
    long micrometers;		/* Size in micrometers (10e-6 meters) */
} font_size;

#define MAX_VALUE	23400L
#define ABS(val)	((val) < 0 ? (-(val)) : (val))

long ux11_font_microm( disp, scrn_num, font)
Display *disp;
int scrn_num;
XFontStruct *font;
{
    double um_per_pixel = ((double) (DisplayHeightMM(disp, scrn_num) * 1000)) /
      ((double) DisplayHeight(disp, scrn_num));
    return(
	(int) ((((double)
		      (font->max_bounds.ascent + font->max_bounds.descent))
		     * um_per_pixel) + 0.5)
	);
}

static long ux11_size_eval(disp, scrn_num, font, data)
Display *disp;			/* Display          */
int scrn_num;			/* Screen number    */
XFontStruct *font;		/* Font to examine  */
VOID_P data;			/* Data to function */
/*
 * This routine examines the font `font' and returns an evaluation
 * of it based on size.  The size is passed in as `data'.  If it
 * is close between fonts,  whether one is proportionally spaced
 * counts for a few more points.
 */
{
    font_size *desired_size = (font_size *) data;
    long height= ux11_font_microm( disp, scrn_num, font);

    return  (MAX_VALUE - ABS(height - desired_size->micrometers)) +
      (font->per_char ? 300 : 0);
}


int ux11_size_font(disp, scrn_num, size, rtn_font, rtn_name, bold)
Display *disp;			/* What display to examine */
int scrn_num;			/* Screen number           */
long size;			/* Font size (micrometers) */
XFontStruct **rtn_font;		/* Returned font           */
char **rtn_name;		/* Returned name           */
int bold;			/* should they be bold fonts or not? (RJB)		*/
/*
 * Finds the closest font supported on the indicated screen of
 * the supplied display whose size is `size' measured in micrometers.
 * The font is returned in `rtn_font' and its name is returned
 * in `rtn_name'.  The routine returns a non-zero status if
 * successful.
 */
{
	font_size data;
	int i;
	static int hit= 0;
	long eval;
	static long max_eval= 0;
	char *Pat[16];
	int pats;
	static XFontStruct *tmpfont;
	static char *tmpname;
	extern FILE *StdErr;
	static Display *last_disp= NULL;
	static int last_scrn_num= -1;
	static int last_bold= 0;
	static long last_size= 0;
	static char last_rtn_name[64]= "\0";
	int same_name;


	if( *rtn_name && strstr( *rtn_name, "symbol") ){
		pats= 6;
		Pat[0]= "-adobe-symbol*-r-*";
		Pat[1]= "*symbol*-r-*";
		Pat[2]= "-*school*-r-*";
		Pat[3]= "-*charter*-r-*";
		Pat[4]= "-adobe-times*-r-*";
		Pat[5]= "-adobe*-r-*";
	}
	else if( bold> 0){
		pats= 5;
		Pat[0]= "*palatino-*bold-r-*";
		Pat[1]= "*schoolbook*bold*-r-*";
		Pat[2]= "*charter*bold*-r-*";
		Pat[3]= "*agfa*bold*";
		Pat[4]= "*bold*-iso8859-1";
	}
	else if( bold< 0){
		pats= 6;
		Pat[0]= "*palatino-*medium*-r-*";
		Pat[1]= "*schoolbook*medium*-r-*";
		Pat[2]= "*lucida*medium-r-*";
		Pat[3]= "*charter*medium*-r-*";
		Pat[4]= "*swiss*medium*-r-*";
		Pat[5]= "*normal*-r-*iso-8859-1";
	}
	else{
		pats= 8;
		Pat[0]= "*helvetica*-r-*";
		Pat[1]= "*palatino-*-r-*";
		Pat[2]= (size< 4200)? "*lucida*sans*-r-*" : "*lucida*-r-*";
		Pat[3]= (size< 4200)? "*lucida*-r-*" : "*schoolbook*-r-*";
		Pat[4]= "*prestige*medium*-r-*";
		Pat[5]= "-adobe*-r-*";
		Pat[6]= "*medium*-r-*";
		Pat[7]= "*normal*-r-*iso8859-1";
	}
	if( (!*rtn_name || !**rtn_name) && !*last_rtn_name ){
		same_name= 0;
	}
	else if( *rtn_name ){
		same_name= strncmp( last_rtn_name, *rtn_name, 63)== 0;
		strncpy( last_rtn_name, *rtn_name, 63);
	}
	else if( strlen(last_rtn_name) ){
		same_name= 0;
		last_rtn_name[0]= '\0';
	}
	else{
		same_name= 1;
	}
	if( !last_disp ||
		!( last_disp== disp && last_scrn_num== scrn_num && last_bold== bold && last_size== size &&
			same_name
		)
	){
		data.micrometers = size;
		ux11_find_font_start= 0;
		max_eval= 0;
		for( hit= i= 0; i< pats; i++){
			eval= ux11_find_font(disp, scrn_num, Pat[i], ux11_size_eval, (VOID_P) &data, &tmpfont, &tmpname);
			if( eval> max_eval ){
				*rtn_font= tmpfont;
				*rtn_name= tmpname;
				hit= i;
				max_eval= eval;
				if( debugFlag){
					fprintf( StdErr, "\tfont '%s' (pattern '%s' ; size %ld mu) score=%ld\n",
						*rtn_name, Pat[hit], ux11_font_microm(disp, scrn_num, *rtn_font ), max_eval
					);
					fflush( StdErr);
				}
/* 				ux11_find_font_start= max_eval;	*/
			}
		}
		if( max_eval ){
			last_disp= disp;
			last_scrn_num= scrn_num;
			last_bold= bold;
			last_size= size;
			tmpfont= *rtn_font;
			tmpname= *rtn_name;
		}
	}
	else{
	  /* Everything is the same, so return remembered values.	*/
		*rtn_name= tmpname;
		*rtn_font= tmpfont;
	}
	if( debugFlag){
		fprintf( StdErr, "ux11_size_font(): font '%s' (pattern #%d; size %ld mu, %ld requested)\n",
			*rtn_name, hit, ux11_font_microm(disp, scrn_num, *rtn_font ), last_size
		);
		fflush( StdErr);
	}
	return( max_eval> 0 );
}
