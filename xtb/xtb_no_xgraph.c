/*
vim:ts=4:sw=4:
 */
#include <stdio.h>
#include <signal.h>
#include <X11/X.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/cursorfont.h>
#include <X11/keysym.h>


#ifndef XGRAPH
#	define NO_ENUM_BOOLEAN
#	include "local/Macros.h"
IDENTIFY("XGraph Mini Toolkit for X11; routines for standalone use in other applications");
#endif
#include <local/xtb.h>
#include "ux11/ux11.h"

int debugFlag, debugLevel;
FILE *StdErr= NULL;

int RemoteConnection=0;

#include <errno.h>
#include <sys/stat.h>
/* stat( filename, &stats);	*/

extern Display *t_disp;
extern int t_screen;

XGFontStruct dialogFont, dialog_greekFont, title_greekFont, titleFont, cursorFont;
char *xtb_ProgramName= "";
extern char *getenv(const char*), *strdup(const char*), *rindex(const char*, int);

int ButtonContrast= 65535/3;

char *FontName( XGFontStruct *f)
{
	if( f== &dialogFont){
		return( "dialogFont");
	}
	else if( f== &dialog_greekFont){
		return( "dialog_greekFont");
	}
	else if( f== &title_greekFont){
		return( "title_greekFont");
	}
	else if( f== &titleFont){
		return( "titleFont");
	}
	else if( f== &cursorFont){
		return( "cursorFont");
	}
	else{
		return( "???");
	}
}

#define xfree(x)	if(x) free(x)

void CheckPrefsDir( char *name )
{ static char called= 0;
  FILE *fp;
	if( !called && name && *name ){
	  extern char *concat();
	  char *pd= concat( getenv("HOME"), "/.Preferences/.", name, NULL );
		if( pd ){
		  struct stat stats;
			if( stat( pd, &stats) ){
				fprintf( StdErr, "Creating new preferences directory \"%s\"\n", pd );
			}
			else{
				xfree( pd );
				return;
			}
			xfree( pd );
		}
		if( (fp= popen( "sh", "w")) ){
			fprintf( fp, "cd $HOME\n" );

			fprintf( fp, "PREFDIR=\".Preferences\"\n" );

			fprintf( fp, "if [ -d ${PREFDIR}/.%s ] ;then\n", name );
			fprintf( fp, "     exit 0\n" );
			fprintf( fp, "fi\n" );

			fprintf( fp, "set -x\n" );

			fprintf( fp, "if [ ! -d ${PREFDIR} ] ;then\n" );
			fprintf( fp, "     mkdir ${PREFDIR}\n" );
			fprintf( fp, "fi\n" );

			fprintf( fp, "if [ ! -d ${PREFDIR}/.%s ] ;then\n", name );
			fprintf( fp, "     if [ -d .%s ] ;then\n", name );
			fprintf( fp, "          tar -cf - .%s | ( cd ${PREFDIR} ; tar -xf - )\n", name );
			fprintf( fp, "          if [ $? = 0 ] ;then\n" );
			fprintf( fp, "               rm -r .%s\n", name );
			fprintf( fp, "          fi\n" );
			fprintf( fp, "     else\n" );
			fprintf( fp, "          mkdir ${PREFDIR}/.%s\n", name );
			fprintf( fp, "     fi\n" );
			fprintf( fp, "fi\n" );

			fprintf( fp, "if [ ! -d ${PREFDIR}/.xtb_default_fonts ] ;then\n" );
			fprintf( fp, "     if [ -d .xtb_default_fonts ] ;then\n" );
			fprintf( fp, "          tar -cf - .xtb_default_fonts | ( cd ${PREFDIR} ; tar -xf - )\n" );
			fprintf( fp, "          if [ $? = 0 ] ;then\n" );
			fprintf( fp, "               rm -r .xtb_default_fonts\n" );
			fprintf( fp, "          fi\n" );
			fprintf( fp, "     else\n" );
			fprintf( fp, "          mkdir ${PREFDIR}/.xtb_default_fonts\n" );
			fprintf( fp, "     fi\n" );
			fprintf( fp, "fi\n" );

			pclose( fp );

			called= 1;
		}
		else{
			fprintf( StdErr, "Warning: can't open connection to the shell sh to verify/establish a preferences directory! (%s)\n",
				serror()
			);
		}
	}
}

int RememberedFont( Display *disp, XGFontStruct *font, char **rfont_name)
{  char name[1024], *home= getenv( "HOME");
   static char font_name[256];
   XFontStruct *tempFont;
   FILE *fp;
   struct stat stats;

	CheckPrefsDir( xtb_ProgramName );

	*rfont_name= NULL;
	if( debugFlag ){
		fprintf( StdErr, "Reading remembered \"%s\": ",
			FontName( font)
		);
	}
	if( !home){
		if( debugFlag)
			fprintf( StdErr, "Can't find home directory (set HOME variable)\n");
		return( 0);
	}
	sprintf( name, "%s/.Preferences/.%s/%s/%s", home, xtb_ProgramName, DisplayString(disp), FontName( font) );
	if( stat( name, &stats) ){
		perror( name);
		sprintf( name, "%s/.Preferences/%s/%s/%s", home, ".xtb_default_fonts", DisplayString(disp), FontName( font) );
		if( stat( name, &stats) ){
			perror( name);
			return(0);
		}
	}
	if( (fp= fopen( name, "r")) ){
		fgets( font_name, 255, fp);
		if( font_name[strlen(font_name)-1]== '\n' )
			font_name[strlen(font_name)-1]= '\0';
		font->font= NULL;
		if( (tempFont= XLoadQueryFont( disp, font_name)) ){
			font->font= tempFont;
			strncpy( font->name, font_name, 127);
			if( debugFlag){
				fprintf( StdErr, "%s : %s (%s)\n", FontName( font), font_name, name);
			}
			fclose( fp);
			*rfont_name= font_name;
			return( 1);
		}
		else{
			if( debugFlag)
				fprintf( StdErr, "Can't load font '%s' for %s\n", font_name, FontName(font) );
			fclose( fp);
			return(0);
		}
	}
	else
		perror( name);
	return(0);
}

int RememberFont( Display *disp, XGFontStruct *font, char *font_name)
{  char name[1024], *home= getenv( "HOME");
   FILE *fp;

	CheckPrefsDir( xtb_ProgramName );

	if( debugFlag ){
		fprintf( StdErr, "Remembering '%s' for \"%s\": ",
			font_name, FontName( font)
		);
	}
	if( !home){
		if( debugFlag)
			fprintf( StdErr, "Can't find home directory (set HOME variable)\n");
		return( 0);
	}
	sprintf( name, "%s/.Preferences/.%s", home, xtb_ProgramName );
	if( mkdir( name, 0744) ){
		if( errno!= EEXIST){
			if( debugFlag){
				perror( name);
			}
			sprintf( name, "%s/.Preferences/%s", home, ".xtb_default_fonts" );
			if( mkdir( name, 0744) ){
				if( errno!= EEXIST){
					if( debugFlag){
						perror( name);
					}
					return(0);
				}
			}
		}
	}
	strcat( name, "/"); strcat( name, DisplayString(disp) );
	if( mkdir( name, 0744) ){
		if( errno!= EEXIST){
			if( debugFlag)
				perror( name);
			return(0);
		}
	}
	strcat( name, "/"); strcat( name, FontName(font) );
	if( (fp= fopen( name, "w")) ){
		fprintf( fp, "%s\n", font_name);
		fclose( fp);
		if( debugFlag)
			fprintf( StdErr, "Remembered %s in '%s'\n", FontName(font), name);
		return( 1);
	}
	else
		perror( name);
	return(0);
}

char *def_str;
XFontStruct *def_font;
Pixel def_pixel;
int def_int;
double def_dbl;

int stricmp( const char *a,  const char *b)
#ifndef NO_STRCASECMP
{
	return( strcasecmp( a, b) );
}
#else
/*
 * This routine compares two strings disregarding case.
 */
{ int value;

    if ((a == (char *) 0) || (b == (char *) 0)) {
		return a - b;
    }

    for( /* nothing */; ((*a | *b) &&
	  !(value = ((isupper(*a) ? *a - 'A' + 'a' : *a) -
		     (isupper(*b) ? *b - 'A' + 'a' : *b))));
		a++, b++
	){
      /* Empty Body */;
	}

    return value;
}
#endif

char *XG_GetDefault( Display *disp, char *Prog_Name, char *name )
{ char *r;
  extern char *getenv();
	if( (r= getenv(name)) ){
		if( debugFlag ){
			fprintf( StdErr, "default %s*%s: using env.var %s=%s\n",
				Prog_Name, name, name, r
			);
			fflush( StdErr );
		}
	}
	else if( (r= XGetDefault( disp, Prog_Name, name )) ){
		if( debugFlag && debugLevel ){
			fprintf( StdErr, "default %s*%s:%s\n",
				Prog_Name, name, r
			);
			fflush( StdErr );
		}
	}
	else if( debugFlag && debugLevel ){
		fprintf( StdErr, "default %s*%s not found in env or X-resources.\n",
			Prog_Name, name
		);
		fflush( StdErr );
	}
	return( r );
}

int rd_flag(char *name)
/* Result in def_int */
{
    if( (def_str = XG_GetDefault(t_disp, xtb_ProgramName, name)) ){
		def_int = (stricmp(def_str, "true") == 0) || (stricmp(def_str, "on") == 0) || (stricmp(def_str, "1") == 0);
		return 1;
    } else {
		return 0;
    }
}

int rd_font(char *name, char **font_name)
/* Result in def_font */
{
    if( (def_str = XG_GetDefault(t_disp, xtb_ProgramName, name)) ){
		if( (def_font = XLoadQueryFont(t_disp, def_str)) ){
			*font_name= def_str;
			return 1;
		} else {
			fprintf(StdErr, "warning: could not load font for %s\n", name);
			return 0;
		}
    } else {
		return 0;
    }
}

int rd_str(char *name)
/* Result in def_str */
{
    if( (def_str = XG_GetDefault(t_disp, xtb_ProgramName, name)) ){
		return 1;
    } else {
		return 0;
    }
}

char *GetFont( XGFontStruct *Font, char *resource_name, char *default_font, long size, int bold, int use_remembered )
{  char *font_name= default_font, *rfont_name= NULL;
   static char called= 0;
	if( !Font->font){
		if( rd_font(resource_name, &font_name )) {
			Font->font = def_font;
			strncpy( Font->name, font_name, 127);
			RememberFont( t_disp, Font, font_name);
		}
		else if( !( use_remembered && RememberedFont( t_disp, Font, &rfont_name) ) ){
			if( font_name){
				if( !(Font->font= XLoadQueryFont( t_disp, font_name )) ){
					fprintf(StdErr, "Can't find %s font %s - trying RememberedFont or suitable default (sized %ld mu)\n",
						FontName(Font), font_name, size
					);
					fflush( StdErr );
					if( !use_remembered || !RememberedFont( t_disp, Font, &rfont_name) ){
						goto load_default;
					}
				}
				else{
					strncpy( Font->name, font_name, 127);
					RememberFont( t_disp, Font, font_name);
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
				if (!ux11_size_font(t_disp, t_screen, size, &Font->font, &font_name, bold)) {
					fprintf(StdErr, "Can't find an appropriate default %s\n", FontName( Font) );
					fflush( StdErr );
					exit(10);
				}
				else{
					strncpy( Font->name, font_name, 127);
					RememberFont( t_disp, Font, font_name);
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
		return( rfont_name? rfont_name : font_name );
    }
	return( (*(Font->name))? Font->name : NULL );
}

UnGetFont( XGFontStruct *Font, char *resource_name)
{
	if( Font && Font->font ){
		XFreeFont( t_disp, Font->font );
	}
}

int GetPointSize( char *fname )
{ char *ff, *lf;
  int lf_ptsz;
	ff= lf= fname;
	while( lf && *lf && !( *lf== '-' && lf[1]== '-' ) ){
		lf++;
	}
	if( lf && sscanf( lf, "--%d-", &lf_ptsz)== 1 ){
		return( lf_ptsz );
	}
	else{
		return(0);
	}
}

XFontStruct *Find_greekFont( char *name, char *greek )
{ int pt;
  XFontStruct *tempFont= NULL;
	if( greek ){
		pt= GetPointSize( name );
		  /* the line below finds a Type1 (DPS) greek font (Symbol) under IRIX 6.3 . I don't
		   \ know how portable this is...
		   */
/* 		sprintf( greek, "-adobe-symbol-medium-r-normal--%d-0-0-0-p-0-adobe-fontspecific", pt);	*/
		sprintf( greek, "-*-symbol-medium-r-normal--%d-0-0-0-p-0-*-*", pt);
		if( !(tempFont= XLoadQueryFont( t_disp, greek )) ){
			sprintf( greek, "*-symbol*--%d-*", pt);
			if( !(tempFont= XLoadQueryFont( t_disp, greek )) ){
				sprintf( greek, "*-symbol*--%d-*", pt+ 1 );
				if( !(tempFont= XLoadQueryFont( t_disp, greek )) ){
					sprintf( greek, "*-symbol*--%d-*", pt- 1 );
					tempFont= XLoadQueryFont( t_disp, greek );
				}
			}
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
	}
	return(r);
}

int TrueGray= 0, reverseFlag= 0;
extern int xtb_depth, xtb_vismap_type;
extern Colormap xtb_cmap;

/* The following routine reverses a colour. Black and white are reversed (whatever
 \ their colour actually is!!). Other pixel-values assume a monotonously increasing
 \ mapping from pixel value (index) to colour.
 */
int ReversePixel(Pixel *pixValue)
{
    if (*pixValue == xtb_white_pix )
      *pixValue = xtb_black_pix;
    else if (*pixValue == xtb_black_pix)
      *pixValue = xtb_white_pix;
	else{
		CLIP_EXPR( *pixValue, (1 << xtb_depth)-1 - *pixValue, 0, (1 << xtb_depth)-1 );
	}
}

char *ParsedColourName= NULL;
#define StoreCName(name)	xfree(name);name=strdup(ParsedColourName)

double IntensityRGBValues[4]= { 0,0,0, 0 };

XColor GetThisColor, GotThisColor;
int GetColor( char *Name, Pixel *pix )
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
  int ret= 0;

	if( !Name || !pix ){
		return(0);
	}

	emsg[0]= '\0';

	GotThisColor.pixel= -1;

	if( Name== GETCOLOR_USETHIS ){
		def= GetThisColor;
		sprintf( cname, "rgb:%x/%x/%x", def.red, def.green, def.blue );
		strcpy( name, cname);
		ok= 1;
	}
	else{
	  char *xg_rgbi;
		if( ! *Name ){
			return(0);
		}
		strncpy( name, Name, 255 );
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
					rgb[n]/= 255;
				}
				sprintf( cname, "rgbi:%g/%g/%g", rgb[0], rgb[1], rgb[2] );
				strcpy( name, cname);
			}
		}
		if( (xg_rgbi= strstr( name, " (rgbi:")) ){
		  /* A colour saved by xgraph, having an intensity-spec. associated with it
		   \ "name (rgbi:ir/ig/ib)"
		   */
			*xg_rgbi++ = '\0';
		}
		if( XLookupColor( t_disp, xtb_cmap, name, &def, &GotThisColor ) ){
			ok= 1;
		}
		else{
		  char *c;
			if( xg_rgbi && (c= index( xg_rgbi, ')')) ){
				*c= '\0';
				sprintf( emsg, "GetColor(): cannot parse colour specification: \"%s\", trying \"%s\"\n",
					name, &xg_rgbi[1]
				);
				strcpy( name, &xg_rgbi[1] );
				if( XLookupColor( t_disp, xtb_cmap, name, &def, &GotThisColor ) ){
					ok= 1;
				}
			}
		}
	}
	if( ok ){
		if( TrueGray ){
		  int g= (int)(xtb_PsychoMetric_Gray( &def )+ 0.5);
			def.red= g;
			def.green= g;
			def.blue= g;
			GotThisColor.pixel= -1;
		}
		if( reverseFlag ){
			def.red= 65535- def.red;
			def.green= 65535- def.green;
			def.blue= 65535- def.blue;
			GotThisColor.pixel= -1;
		}
		ret= 0;
		if( xtb_vismap_type== 2 ){
			if( XStoreColor(  t_disp, xtb_cmap, &def ) ){
				ret= 1;
				if( GotThisColor.pixel== -1 ){
					GotThisColor= def;
				}
			}
		}
		else{
			if( XAllocColor( t_disp, xtb_cmap, &def)) {
				ret= 1;
				if( GotThisColor.pixel== -1 ){
					GotThisColor= def;
/* 
					GotThisColor.pixel= def.pixel;
					XQueryColor(  t_disp, xtb_cmap, &GotThisColor );
 */
				}
			}
		}
		if( ret ){
			*pix= def.pixel;
			GetThisColor= def;
			sprintf( cname, "%s (rgbi:%g/%g/%g)", name, def.red/ 65535.0, def.green/65535.0, def.blue/65535.0 );
			ParsedColourName= cname;
				IntensityRGBValues[0]= GotThisColor.red/ 65535.0;
				IntensityRGBValues[1]= GotThisColor.green/ 65535.0;
				IntensityRGBValues[2]= GotThisColor.blue/ 65535.0;
				IntensityRGBValues[3]= GotThisColor.pixel;
		}
		else{
			sprintf( emsg, "GetColor(): could not store/allocate color: `%s'\n", name);
		}
    } else {
		sprintf( emsg, "%sGetColor(): cannot parse color specification: `%s'\n", emsg, name);
    }
	if( emsg[0] ){
		fprintf( StdErr, "xtb::%s", emsg );
	}
	return ret;
}

Boolean xtb_inited= False;

void init_xtb_standalone(char *name, Display *disp, int scrn, XVisualInfo *vi, Colormap colourmap, int depth, unsigned long fg, unsigned long bg )
{  char *c= rindex( name, '/');
   extern Visual *xtb_vis;
   static Colormap xcmap= 0;
	if( !StdErr ){
		StdErr= stderr;
	}
	if( c ){
		xtb_ProgramName= strdup(&c[1]);
	}
	else{
		xtb_ProgramName= strdup(name);
	}

	  /* Switch on synchronised mode when problems occur: they will be reported when they occur,
	   \ and not when the X11 cache is flushed!
	   */
/* 	XSynchronize( disp, True );	*/

	t_disp= disp;
	t_screen= scrn;

	xtb_depth= vi->depth;
	xtb_vis= vi->visual;
	  /* Should make a separate colourmap as in ux11_std_vismap?
	   \ NB: ux11_std_vismap() no longer needs to be called in xtb.c!
	   \ 20020126: solved a problem when opening from graftool/grafvarsV with a "simulated"
	   \ colourdepth different from the server's default depth. With the code as it is now,
	   \ the VLC window opens correctly when $PLOTD<16 on a screen with a single 16planes
	   \ visual. Should still be tested on a multi-visual server (SGI)! But there is less
	   \ reason it won't function because all (or most) calls now do not longer use the
	   \ default visual combined with the desired depth...
	   */
	xtb_cmap= colourmap;
/* 		if( vi->visual== DefaultVisual(t_disp, DefaultScreen(t_disp)) ){	*/
/* 			xtb_cmap = DefaultColormap(t_disp, DefaultScreen(t_disp));	*/
/* 		}	*/
/* 		else{	*/
/* 			if( xcmap ){	*/
/* 				XFreeColormap( t_disp, xcmap );	*/
/* 			}	*/
/* 			xtb_cmap= xcmap = XCreateColormap(t_disp,	*/
/* 							RootWindow(t_disp, t_screen),	*/
/* 							vi->visual, AllocNone	*/
/* 			);	*/
/* 			XInstallColormap( t_disp, xtb_cmap );	*/
/* 		}	*/

	titleFont.font= NULL;
	dialogFont.font= NULL;
	title_greekFont.font= NULL;
	dialog_greekFont.font= NULL;
	cursorFont.font= NULL;

	if( disp ){
		GetFont( &titleFont, "TitleFont", NULL, 4500, 1, 1);
		{ char *gf;
			 if( Update_greekFonts( 'TITL') ){
				  RememberFont( disp, &title_greekFont, title_greekFont.name );
			 }
			 else if( !RememberedFont( disp, &title_greekFont, &gf) ){
				 fprintf( stdout, "xgraph: can't get title_greek font matching titleFont (%s)\n",
					 titleFont.name
				 );
			 }
		}
	}
	if( disp ){
		GetFont( &dialogFont, "DialogFont", NULL, 4175, -1, 1);
		{ char *gf;
			 if( Update_greekFonts( 'DIAL') ){
				  RememberFont( disp, &dialog_greekFont, dialog_greekFont.name );
			 }
			 else if( !RememberedFont( disp, &dialog_greekFont, &gf) ){
				 fprintf( stdout, "xgraph: can't get dialog_greek font matching dialogFont (%s)\n",
					 dialogFont.name
				 );
			 }
		}
	}
	GetFont( &cursorFont, "CursorFont", "cursor", 0, 0, 1);
	xtb_init( t_disp, scrn, fg, bg, dialogFont.font, dialog_greekFont.font, True );
	  /* Flushing will cause the errors to appear now ?! */
	XSync( t_disp, False );
	xtb_inited= True;

/* 	XSynchronize( disp, True );	*/
}

