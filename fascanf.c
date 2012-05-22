#include "config.h"
IDENTIFY("Data reading routines: fascanf2(), ReadString() and Sinc routines");

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <ctype.h>

#include "cpu.h"
#include "xgALLOCA.h"

#include "Macros.h"

#include "Sinc.h"

extern char *index();

extern int debugFlag, debugLevel, scriptVerbose;
extern FILE *StdErr;

extern char *d2str( double, const char*, const char*);

/* read multiple floating point values	*/
int fascanf2( int *n, char *s, double *a, int separator)
{ extern char ascanf_separator;
  int sep= ascanf_separator;
	ascanf_separator= (separator> 0 && separator< 256)? separator : ' ';
	fascanf( n, s, a, NULL, NULL, NULL, NULL );
	ascanf_separator= sep;
	if( debugFlag && debugLevel ){
	  int i;
		fprintf( StdErr, "fascanf2(%d,\"%s\")= %s", *n, s, d2str( a[0], "%g", NULL) );
		for( i= 1; i< *n; i++ ){
			fprintf( StdErr, ", %s", d2str( a[i], "%g", NULL) );
		}
		fputc( '\n', StdErr );
	}
	return( *n );
}

#include "va_dcl.h"

#include "dymod.h"

extern char *concat( char *first, VA_DCL );

/* #include <readline/readline.h>	*/
/* #include <readline/history.h>	*/

int Use_ReadLine= False;
void *lib_termcap= NULL, *lib_readline= NULL;
FNPTR( gnu_readline, char*, (char *prompt))= NULL;
FNPTR( gnu_add_history, void, (char *line))= NULL;
FNPTR( gnu_read_history, int, (char *fn))= NULL;
FNPTR( gnu_write_history, int, (char *fn))= NULL;
/* FNPTR( gnu_rl_completion_entry_function, char**, (char *text, int state))= NULL;	*/
void **gnu_rl_completion_entry_function=NULL;
int *gnu_rl_completion_append_character;
#if 0
This can be used to handle X11 events periodically??? Avoid recursion of gnu_readline calls when handling events like that!

/* The address of a function to call periodically while Readline is
   awaiting character input, or NULL, for no event handling. */
extern rl_hook_func_t *rl_event_hook;
#endif
void **gnu_rl_event_hook=NULL;
unsigned int grl_HandlingEvents= 0;
extern int Num_Windows, StartUp;

void grl_HandleEvents()
{ unsigned int gHE= grl_HandlingEvents;
  extern void *ActiveWin, *LastDrawnWin;
  extern char *LocalWinRepr( void*, char * );
  static char active = 0;

	if( !active ){
		active = 1;
		if( Num_Windows && !StartUp ){
			grl_HandlingEvents+= 1;
			Handle_An_Events( -1, 1, "GNU ReadLine idle hook", 0, 0 );
			grl_HandlingEvents= gHE;
			if( !ActiveWin && LastDrawnWin ){
				ActiveWin = LastDrawnWin;
				if( debugFlag ){
					fprintf( StdErr, "## Active window now set to \"%s\" (last drawn window)\n",
						LocalWinRepr(ActiveWin, NULL)
					);
				}
			}
		}
		// 20110131 : call the current hook if it's not us...
		if( (gnu_rl_event_hook && *gnu_rl_event_hook && *gnu_rl_event_hook != grl_HandleEvents) ){
		  void (*greh)() = *gnu_rl_event_hook;
			(*greh)();
		}
		active = 0;
	}
}

#define STRINGCHECK(s,max)	StringCheck(s,max,__FILE__,__LINE__)

int ReadString_UnFinished= 0, ReadString_Warn= 1;
char Unfinished_Line[64];

int _StartJoin( char *buf, char *first, int dlen, int instring )
{
	return(0);
}

int _EndJoin( char *buf, char *first, int dlen, int instring )
{
	return( 0 );
}

static int newFile= 0;

// RJVB 20081205: wrap fgets() with some code that will translate \r\n EOLs with a "regular" '\n':
static char *_fgets( char *buffer, int len, FILE *fp )
{ char *c= buffer;
	if( buffer && (c= fgets( buffer, len, fp )) ){
	  int l= strlen(buffer);
		if( l > 1 && buffer[l-1] == '\n' && buffer[l-2] == '\r' ){
			buffer[l-2]= '\n';
			buffer[l-1]= '\0';
			if( newFile && (scriptVerbose || debugFlag) ){
				fprintf( StdErr, "## Reading file with DOS-style CRLF line-endings\n" );
			}
		}
	}
	return( c );
}

/* The read string function (replacement for fgets()) used by ReadData(). It allows to read over
 \ multiple lines; a \n before a newline is taken as a "join" operator. Also, it supports a joining
 \ mode controllable by the user, via the 2 callback routines passed as arguments. These callbacks
 \ only get called when appropriate, i.e. the one to end joining only when joining mode is active.
 */

static char *_ReadString( char *buffer, int len, char *Prompt, FILE *fp, int *line, 
	  /* Start joining mode when this method returns true. When it returns 2, really only exit this mode
	   \ when EndJoining says so.
	   */
	int (*StartJoining)(char *buf, char *first_nspace, int len, int instring),
	int (*EndJoining)(char *buf, char *first_nspace, int len, int instring),
	int url
)
{ char *c, *d, *_buffer= buffer, *first_nl= NULL;
   int join= 0;
   extern int DumpFile;
   Boolean till_empty= False, instring= False;
   static FILE *prev_fp= NULL;
   char *rlbuf= NULL, *prompt= Prompt;

	if( !buffer ){
		return( NULL );
	}
	else if( grl_HandlingEvents ){
		fprintf( StdErr, "An interactive console is already active/open.\n" );
		return( NULL );
	}

	if( !StartJoining ){
		StartJoining= _StartJoin;
	}
	if( !EndJoining ){
		EndJoining= _EndJoin;
	}

	if( fp!= prev_fp ){
		ReadString_UnFinished= False;
		prev_fp= fp;
	}

	errno= 0;
	buffer[0]= '\0';
	if( *line<= 0 ){
		newFile= 1;
	}
	else{
		newFile= 0;
	}
	*line-= 1;
	do{
		  /* Get input, using fgets(). It is stored in <buffer>, at the location
		   \ pointed to by _buffer. That location keeps track of joining lines.
		   */
		if( url ){
			if( (rlbuf= (*gnu_readline)( (till_empty)? "\\\\ ": prompt )) ){
				strncpy( _buffer, rlbuf, len- strlen(buffer)- 1 );
				  /* gnu readline removes the final newline: put it back */
				strcat( _buffer, "\n" );
				d= _buffer;
			}
			else{
				d= NULL;
			}
		}
		else{
			d= _fgets( _buffer, len- strlen(buffer), fp);
		}
		  /* 20031013 */
		if( till_empty && !d ){
			d= buffer;
		}
		if( !d ){
			if( errno== EINTR){
			  /* if we receive a signal, don't fall back
			   * to caller
			   */
				d= buffer;
				fprintf( StdErr, "_ReadString(): %s (%s)\n",
					serror(), buffer
				);
				fflush( StdErr );
				*d= '\0';
				errno= 0;
			}
			else{
				if( !fp || (feof(fp) && !ferror(fp)) ){
					  /* 20000131: this should remove the need to finish with an empty
					   \ line at EOF when reading-upto-the-next-empty-line mode.
					   */
					if( till_empty ){
						if( buffer[ strlen(buffer)-1 ]!= '\n' ){
							ReadString_UnFinished= True;
							strncpy( Unfinished_Line, buffer, 63 );
							Unfinished_Line[63]= '\0';
						}
						else{
							ReadString_UnFinished= False;
						}
						goto _ReadString_return;
					}
					d= NULL ; goto _ReadString_return;
				}
				else{
					if( errno ){
						fprintf( StdErr, "_ReadString() error at line %d: %s\n", *line, serror() );
					}
					if( *buffer && buffer[ strlen(buffer)-1 ]!= '\n' ){
						ReadString_UnFinished= True;
						strncpy( Unfinished_Line, buffer, 63 );
						Unfinished_Line[63]= '\0';
					}
					else{
						ReadString_UnFinished= False;
					}
					d= NULL ; goto _ReadString_return;
				}
			}
		}
		else{
		  int dlen= (d)? strlen(d) : 0;
		  char *first= d;
		  int start_join= False, end_join= True;
  /* 20031013: ignore EOF when in joining mode. This allows to switch off joining mode without any
   \ additional code to specifically handle the case. It should not cause problems (attempts to read
   \ beyond EOF should be ignored by the OS.
   */
#define cfeof(fp)	(!till_empty && feof(fp))
			if( (!d && (!fp || !cfeof(fp))) || (fp && ferror( fp ) && !cfeof(fp)) ){
				fprintf( StdErr, "_ReadString() error at line %d: %s\n", *line, serror() );
				if( fp ){
					clearerr(fp);
				}
				  /* 990715: we should return at this point..!	*/
				if( buffer[ strlen(buffer)-1 ]!= '\n' ){
					ReadString_UnFinished= True;
					strncpy( Unfinished_Line, buffer, 63 );
					Unfinished_Line[63]= '\0';
				}
				else{
					ReadString_UnFinished= False;
				}
				d= NULL ; goto _ReadString_return;
			}
#if defined(unix) || defined(linux)
			if( dlen> 1 && d[dlen-1]== '\n' && d[dlen-2]== '\r' ){
				d[dlen-2]= '\n';
				d[dlen-1]= '\0';
				dlen-= 1;
			}
#endif
			  /* Find the first character in the freshly read string that is not whitespace:	*/
			while( first && *first && isspace((unsigned char)*first) ){
				first++;
			}
			c= first;
			  /* See if this string contains a string section itself, between matched braces.
			   \ Such a section can extend over multiple input lines, and we should not allow special
			   \ processing (start/end joining, comments) within, or users may become upset...
			   */
			while( *c ){
				if( *c== '"' && ( c== first || c[-1]!= '\\' || (c-first>1 && c[-1]=='\\' && c[-2]=='\\')) ){
					instring= !instring;
				}
				c++;
			}
			c= NULL;
			  /* See if we should start/stop joining:	*/
			if( !till_empty ){
				start_join= (*StartJoining)( d, first, dlen, instring );
				end_join= False;
			}
			else{
				start_join= False;
				end_join= (fp && feof(fp)) || (*EndJoining)( d, first, dlen, instring );
			}
			  /* Check if we have to continue reading, either because there is a '\\' and a 'n' before
			   \ the EOL, or because we are already in "till_empty" mode. Check in that order so that
			   \ we can be in both modes at once..
			   */
			if( ((dlen> 1 && d[dlen-1]== '\n') || till_empty== 2) &&
/* 				(till_empty || (dlen>=3 &&    *(&d[dlen-2]) == 'n'&& *(c=&d[dlen-3]) == '\\') )	*/
				  /* 20020505: the check for \\n above should be replaced by start_join, since (*StartJoin) should decide! */
/* 				(till_empty || (dlen>=3 && ( (*(&d[dlen-2]) == 'n'&& *(c=&d[dlen-3]) == '\\')  )) )	*/
				(till_empty || (dlen>=3 && ( (*(&d[dlen-2]) == 'n'&& *(c=&d[dlen-3]) == '\\') || start_join )) )
			){
				  /* c may get set within the if() statement, or it may not: check that here. If not set,
				   \ OR NOT '\\', we have a "normal" line. If we're not in joining (till_empty) mode, such a line is
				   \ not processed in any particular fashion, except for comment checking.
				   */
				if( !c || *c!= '\\' ){
					c= &d[dlen-1];
					  /* 20020506: don't touch anything but a newline; if the last char is not one,
					   \ point c to the string's end!
					   */
					while( *c && *c!= '\n' ){
						c++;
					}
				}
				if( *first== '#' && !instring && !(start_join || end_join) ){
				  /* outcommented line	*/
					if( !DumpFile ){
						d[0]= '\0';
						_buffer= c= d;
					}
				}
				else{
				  /* join lines	*/
					if( DumpFile && !till_empty ){
						*c++= '\\';
						*c++= 'n';
					}
					if( !first_nl ){
						first_nl= c;
					}
					  /* If we join a line in till_empty mode, we'll replace a newline by a newline. Otherwise,
					   \ we replace a \ n sequence by a newline...
					   */
					*c++= '\n';
					  /* 20020506: just for nicer printing of feedback (if requested) : */
					*c= '\0';
					  /* Continue storing input here:	*/
					_buffer= c;
					if( start_join ){
#ifndef DEBUG
						if( debugFlag )
#endif
						{
							fprintf( StdErr, "_ReadString(): line %d=\"%s\": switched to 'read-'till-empty-line' mode\n",
								*line, buffer
							);
						}
						till_empty= start_join /* True */;
					}
					else if( end_join ){
						till_empty= False;
						c= NULL;
						if( debugFlag ){
							fprintf( StdErr, "_ReadString(): line %d=\"%s\": end of 'read-'till-empty-line' mode and EOI\n",
								*line, buffer
							);
						}
					}
				}
				join= 1;
			}
			  /* 990715: allow commentlines without trailing newline in a "joining" block	*/
			else if( join && *first== '#' && !instring && !(start_join || end_join) ){
				if( !DumpFile ){
					d[0]= '\0';
					_buffer= c= d;
				}
				else{
					_buffer= c= &d[dlen-1];
				}
			}
			else if( dlen> 1 && d[dlen]!= '\n' && *(&d[dlen-1]) == 'n'&& *(c=&d[dlen-2]) == '\\' ){
				fprintf( StdErr, "_ReadString(): line %d=\"%s\": warning: \\n not followed by a newline\n",
					*line, buffer
				);
			}
			else{
				c= NULL;
			}
		}
		*line+= 1;
		xfree( rlbuf );
	} while( d && c && len- strlen(buffer)> 0 );

	  /* 20060926: protect against buffer[-1] access when buffer is empty...!
	   \ This bug caused an out-of-bounds write to occur somewhere, that messed up something in my
	   \ window-manager, which in turn caused it to request a ridiculous window-size when full-zooming
	   \ the first window opened after triggering of this bug. Which in its turn caused the X11 server
	   \ under Mac OS X 10.4.7 to crash...
	   */
	if( *buffer && buffer[ strlen(buffer)-1 ]!= '\n' ){
		ReadString_UnFinished= True;
		strncpy( Unfinished_Line, buffer, 63 );
		Unfinished_Line[63]= '\0';
	}
	else{
		if( ReadString_UnFinished ){
		  char *nl= index( buffer, '\n' );
			if( !first_nl ){
			  /* a single line was read	*/
				first_nl= &buffer[ strlen(buffer)-1 ];
			}
			if( nl && (nl< first_nl || (nl== first_nl && nl== buffer)) && ReadString_Warn ){
			  /* There is a newline before the first newline that should be there. That may
			   \ mean that the previous line read (during the previous call) was not terminated
			   \ for some reason (*BUFLEN* too small?).
			   */
				fprintf( StdErr,
					"_ReadString(): WARNING: line %d: unexpected newline encountered -- "
					"may belong to a line that did not fit into the previous buffer (increase *BUFLEN* to >= %d) !\n"
					" Start of previous line: \"%s\"\n"
					" Current line: \"%s\"\n",
						*line,
						LMAXBUFSIZE+ (nl- buffer+ 1),
						Unfinished_Line, buffer
				);
			}
		}
		ReadString_UnFinished= False;
	}

_ReadString_return:;
	if( d ){
		if( debugFlag ){
			if( debugLevel==2 ){
				fprintf( StdErr, "_ReadString(len=%d,\"%s\") -> \"%s\"\n",
					len, (Prompt)? Prompt : "(NULL)", buffer
				);
			}
			STRINGCHECK( buffer, len );
		}
		return( buffer );
	}
	else{
		return( NULL );
	}
}

char *ReadString( char *buffer, int len, FILE *fp, int *line, 
	int (*StartJoining)(char *buf, char *first_nspace, int len, int instring),
	int (*EndJoining)(char *buf, char *first_nspace, int len, int instring)
)
{
	return( _ReadString( buffer, len, NULL, fp, line, StartJoining, EndJoining, 0 ) );
}

#ifdef __MACH__
#	define LIBREADLINE	"libreadline.dylib"
#	define LIBTERMCAP	"libtermcap.dylib"
#elif __CYGWIN__
#	define LIBREADLINE	"/usr/bin/cygreadline7.dll"
// probably non-existent, but we don't use it anyway:
#	define LIBTERMCAP	"libtermcap.dll"
#else
#	define LIBREADLINE	"libreadline.so"
#	define LIBTERMCAP	"libtermcap.so"
#endif

char *ReadLine( char *buffer, int len, char *prompt, int *line, 
	int (*StartJoining)(char *buf, char *first_nspace, int len, int instring),
	int (*EndJoining)(char *buf, char *first_nspace, int len, int instring)
)
{
	if( !lib_readline ){
	  char *err= NULL;
	  char *ltname= getenv("XG_LIBTERMCAP");
	  char *lrname= getenv("XG_LIBREADLINE");
	  int dbF= debugFlag, sV= scriptVerbose;
#ifndef DEBUG
		  // 20081219: no need to bother the user with warnings that system libs aren't found in the PrefsDir!!
		debugFlag= scriptVerbose= 0;
#endif
		if( !ltname ){
			ltname = strdup(LIBTERMCAP);
		}
		else{
			ltname = strdup(ltname);
		}
		if( !lrname ){
			lrname = strdup(LIBREADLINE);
		}
		else{
			lrname = strdup(lrname);
		}
		lib_termcap= XG_dlopen( &ltname, RTLD_NOW|RTLD_GLOBAL, &err);
		lib_readline= XG_dlopen( &lrname, RTLD_NOW|RTLD_GLOBAL, &err);
		debugFlag= dbF;
		scriptVerbose= sV;
		if( lib_readline ){
		  char (*cf)(char*, int);
		  extern char *PrefsDir;
			gnu_readline= dlsym( lib_readline, "readline");
			gnu_add_history= dlsym( lib_readline, "add_history" );
			if( (gnu_read_history= dlsym( lib_readline, "read_history" )) ){
			  char *fn= concat( PrefsDir, "/history", NULL );
				if( fn ){
					(*gnu_read_history)( fn );
					xfree( fn );
				}
			}
			gnu_write_history= dlsym( lib_readline, "write_history" );
			if( (gnu_rl_completion_entry_function= dlsym( lib_readline, "rl_completion_entry_function" )) ){
			  extern char *grl_MatchVarNames( char *string, int state );
				*gnu_rl_completion_entry_function= grl_MatchVarNames;
			}
			if( (gnu_rl_completion_append_character= dlsym( lib_readline, "rl_completion_append_character" )) ){
				*gnu_rl_completion_append_character= '\0';
			}
			gnu_rl_event_hook= dlsym( lib_readline, "rl_event_hook" );
		}
		if( !lib_readline || !gnu_readline || err ){
			fprintf( StdErr, "ReadLine: can't load/initialise %s (%s): GNU readline not available.\n",
				LIBREADLINE,
				(err)? err : "unknown error"
			);
			gnu_readline= NULL;
			gnu_add_history= NULL;
			gnu_rl_event_hook= NULL;
			if( lib_readline ){
				dlclose( lib_readline );
				lib_readline= NULL;
			}
			if( lib_termcap ){
				dlclose( lib_termcap );
				lib_termcap= NULL;
			}
			Use_ReadLine= False;
		}
		xfree(ltname);
		xfree(lrname);
	}

	  /* 20090326: for some reason, installing the libreadline idle handler (that does "background" X11 event handling)
	   \ interferes with proper command history walking when no windows are open yet, even if grl_HandleEvents()
	   \ itself doesn't do squat during the startup phase. The solution appears to be to install grl_HandleEvents()
	   \ only *after* the startup phase.
	   */
	if( gnu_rl_event_hook && !*gnu_rl_event_hook && Num_Windows && !StartUp ){
		*gnu_rl_event_hook= grl_HandleEvents;
	}
	if( gnu_readline ){
	  char *r= _ReadString( buffer, len, prompt, NULL, line, StartJoining, EndJoining, 1 );
		if( r && *r && gnu_add_history ){
		  char *c= &r[ strlen(r)-1 ], C= *c;
			if( *c== '\n' ){
				*c= '\0';
			}
			(*gnu_add_history)( r );
			*c= C;
		}
		return(r);
	}
	else{
		fputs( prompt, StdErr ); fflush( StdErr );
		return( _ReadString( buffer, len, NULL, stdin, line, StartJoining, EndJoining, 0 ) );
	}
}

Sinc *Sinc_string_behaviour( Sinc *sinc, char *string, long cnt, long base, SincString behaviour )
{
	if( !sinc ||
		(!string && behaviour!= SString_Dynamic && !(behaviour== SString_Global && SincString_Behaviour== SString_Dynamic))
	){
		return( NULL);
	}
	  /* 20020313 */
	if( cnt> 0 && !string ){
		string= calloc( cnt, sizeof(char) );
		string[0]= '\0';
	}
	if( (sinc->sinc.string= string) ){
		sinc->_tlen= strlen(string);
		if( cnt<= 0 ){
			cnt= strlen(string);
		}
	}
	else{
		sinc->_tlen= 0;
	}
	sinc->type= SINC_STRING;
	sinc->alloc_len= sinc->_cnt= ( cnt > 0L)? cnt : 0L;
	sinc->_base=  (base > 0L)? base : 0L;
	sinc->behaviour= behaviour;
	return( sinc);
}

Sinc *Sinc_string( Sinc *sinc, char *string, long cnt, long base )
{
	return( Sinc_string_behaviour( sinc, string, cnt, base, SString_Fixed ) );
}

Sinc *Sinc_file( Sinc *sinc, FILE *file, long cnt, long base )
{
	if( !sinc || !file )
		return( NULL);
	sinc->sinc.file= file;
	sinc->type= SINC_FILE;
	sinc->_cnt= sinc->_base=  0L;
	sinc->_tlen= 0;
	return( sinc);
}

Sinc *Sinc_base( Sinc *sinc, long base )
{
	sinc->_base= base;
	return( sinc);
}

int Sflush( Sinc *sinc )
{
	if( !sinc || !sinc->sinc.string ){
		  /* other error candidates: ENOENT, ESPIPE	*/
		errno= ENODATA;
		return( EOF);
	}
	switch( sinc->type){
		case SINC_STRING:
			sinc->sinc.string[ sinc->_cnt- 1]= '\0';
			sinc->_tlen= strlen(sinc->sinc.string);
			return( 0);
			break;
		case SINC_FILE:
			sinc->_tlen+= sinc->_cnt;
			sinc->_cnt= 0;
			return( fflush( sinc->sinc.file) );
			break;
		default:
			errno= EINVAL;
			return( EOF);
	}
}

int Srewind( Sinc *sinc )
{
	if( !sinc || !sinc->sinc.string ){
		  /* other error candidates: ENOENT, ESPIPE	*/
		errno= ENODATA;
		return( EOF);
	}
	switch( sinc->type){
		case SINC_STRING:
			sinc->_cnt= 0;
			sinc->sinc.string[ sinc->_cnt]= '\0';
			sinc->_tlen= strlen(sinc->sinc.string);
			return( 0);
			break;
		case SINC_FILE:
			sinc->_tlen= sinc->_cnt= 0;
			rewind( sinc->sinc.file );
			return( 0 );
			break;
		default:
			errno= EINVAL;
			return( EOF);
	}
}

int SincString_Behaviour= SString_Fixed;

int SincAllowExpansion( Sinc *sinc )
{
	if( !sinc ){
		return(0);
	}
	if( sinc->behaviour== SString_Dynamic || (sinc->behaviour== SString_Global && SincString_Behaviour!= SString_Fixed) ){
		return(1);
	}
	else{
		return(0);
	}
}

int Sputs( char *text, Sinc *sinc )
{  int i= 0;
	if( !text)
		return( 0);
	if( !sinc ){
		errno= ENOENT;
		return( EOF);
	}
	switch( sinc->type){
		case SINC_STRING:{
		  int sia= SincAllowExpansion(sinc);
			if( (!sinc->sinc.string && !sia) ){
				errno= ENOENT;
				return( EOF);
			}
			if( sinc->_base< 0){
				sinc->_base= 0;
			}
			{ int tlen= strlen(text)+1;
				if( (!sinc->sinc.string || tlen+ sinc->_base> sinc->alloc_len) && sia ){
					tlen+= sinc->_base;
					errno= 0;
					if( sinc->sinc.string && sinc->alloc_len> 1 ){
						sinc->sinc.string= (char*) realloc( sinc->sinc.string, tlen* sizeof(char));
					}
					else{
						if( sinc->sinc.string ){
							free( sinc->sinc.string );
						}
						sinc->sinc.string= (char*) calloc( tlen, sizeof(char));
						sinc->_base= 0;
					}
					sinc->alloc_len= tlen;
				}
				else{
					tlen= sinc->_cnt;
				}
				if( !sinc->sinc.string ){
					if( !errno ){
						errno= ENOMEM;
					}
					return(EOF);
				}
				sinc->_cnt= tlen;
			}
			if( sinc->_cnt> 0 ){
			  /* suppose the length indication makes sense: use it.
			   * make sure the string is always terminated
			   */
				sinc->sinc.string[sinc->_cnt-1]= '\0';
				if( sinc->_base < sinc->_cnt- 1 || (sinc->_base== sinc->_cnt-1 && sinc->alloc_len> 1 && sia) ){
				    /* 20020313 */
				  int cnt= (sia)? sinc->alloc_len : sinc->_cnt;
					for( i= 0; sinc->_base < cnt-1 && *text ; sinc->_base++, i++ )
						sinc->sinc.string[sinc->_base]= *text++ ;
					sinc->sinc.string[sinc->_base]= '\0';
					if( sia ){
						sinc->_cnt= strlen( sinc->sinc.string )+ 1;
					}
					return(i);
				}
				else{
					if( sia ){
						sinc->_base= strlen( sinc->sinc.string );
					}
					else{
						sinc->_base= sinc->_cnt;
					}
				}
			}
			else{
			  /* okay, just hope for the best...	*/
				for( i= 0; *text ; sinc->_base++, i++ )
					sinc->sinc.string[sinc->_base]= *text++ ;
				sinc->sinc.string[sinc->_base]= '\0';
				if( sia ){
					sinc->_cnt= strlen( sinc->sinc.string )+ 1;
				}
				return(i);
			}
			return( EOF);
			break;
		}
		case SINC_FILE:{
		  int n= fputs( text, sinc->sinc.file );
			if( n!= EOF ){
				sinc->_cnt+= n;
			}
			return( n );
			break;
		}
		default:
			errno= ENOENT;
			return( EOF);
	}
}

Sinc *SSputs( char *text, Sinc *sinc )
{
	if( Sputs( text, sinc)!= EOF )
		return( sinc);
	else
		return( NULL);
}

int Sputc( int c, Sinc *sinc )
{
	if( !sinc ){
		errno= ENODATA;
		return( EOF);
	}
	switch( sinc->type){
		case SINC_STRING:{
		  int sia= SincAllowExpansion(sinc);
			if( (!sinc->sinc.string && !sia) ){
				errno= ENOENT;
				return( EOF);
			}
			if( sinc->_base< 0){
				sinc->_base= 0;
			}
			{ int tlen= 2;
				if( (!sinc->sinc.string || tlen+ sinc->_base> sinc->alloc_len) && sia ){
					tlen+= sinc->_base;
					errno= 0;
					if( sinc->sinc.string && sinc->_cnt> 1 ){
						sinc->sinc.string= realloc( sinc->sinc.string, tlen* sizeof(char));
					}
					else{
						if( sinc->sinc.string ){
							free( sinc->sinc.string );
						}
						sinc->sinc.string= calloc( tlen, sizeof(char));
						sinc->_base= 0;
					}
					sinc->alloc_len= tlen;
				}
				else{
					tlen= sinc->_cnt;
				}
				if( !sinc->sinc.string ){
					if( !errno ){
						errno= ENOMEM;
					}
					return(EOF);
				}
				sinc->_cnt= tlen;
			}
			if( sinc->_cnt> 0 ){
			/* suppose the length indication makes sense: use it.
			 * make sure the string is always terminated	*/
				sinc->sinc.string[sinc->_cnt-1]= '\0';
				if( sinc->_base < sinc->_cnt- 1 || (sinc->_base== sinc->_cnt-1 && sinc->alloc_len> 1 && sia) ){
					sinc->sinc.string[ sinc->_base++ ]= c ;
					sinc->sinc.string[sinc->_base]= '\0';
					if( sia ){
						sinc->_cnt= strlen( sinc->sinc.string )+ 1;
					}
					return(c);
				}
				else{
					if( sia ){
						sinc->_base= strlen( sinc->sinc.string );
					}
					else{
						sinc->_base= sinc->_cnt;
					}
				}
			}
			else{
			  /* okay, just hope for the best...	*/
				sinc->sinc.string[ sinc->_base++ ]= c ;
				sinc->sinc.string[sinc->_base]= '\0';
				if( sia ){
					sinc->_cnt= strlen( sinc->sinc.string )+ 1;
				}
				return( c);
			}
			return( EOF );
			break;
		}
		case SINC_FILE:{
		  int n= fputc( c, sinc->sinc.file );
			if( n!= EOF ){
				sinc->_cnt+= 1;
			}
			return( n );
			break;
		}
		default:
			errno= ENOENT;
			return( EOF);
	}
}

Sinc *SSputc( int c, Sinc *sinc )
{
	if( Sputc( c, sinc)!= EOF)
		return( sinc);
	else	
		return( NULL);
}


/* env.variable extensions, from cx.c : 	*/
char _EnvDir[256]= "./.env", *EnvDir= _EnvDir;
static char CX_env[256];

/* undefine those macros to get at the system functions:	*/
#undef getenv
#undef setenv

char *cgetenv( char *name)
{
	return( (name)? getenv(name) : NULL);
}

/* check if a variable of name <pref><n> exists
 * in the environment or on disk.
 */
char *__GetEnv( char *n, char *pref, int add_prefix )
{	char *env= CX_env;
	char *set, *getenv(), name[256], fname[512];
	char *home= getenv("HOME");
	FILE *EnvFp;
	int nn= EOF;
	char *where;

	errno= 0;
	if( !home)
		home= ".";
	if( !add_prefix ){
	  /* first time: try without prefix	*/
		strncpy( name, n, 255);
	}
	else if( strlen(pref) + strlen(n) < 255 ){
		sprintf( name, "%s%s", pref, n);
	}
	else{
		if( getenv("GETENV_DEBUG") )
			fprintf( StdErr, "xgraph::__GetEnv(\"%s\",\"%s\",%d) = NULL\n", n, pref, add_prefix );
		return( NULL );
	}
	if( !(set= getenv(name)) ){
	  int en;
	  /* name not set in the environment; try to find it on disk	*/
		if( EnvDir && strlen( EnvDir)){
			sprintf( fname, "%s/%s", EnvDir, name);
		}
		else{
			sprintf( fname, "%s/.env/%s", home, name);
		}
		en= errno;
		if( ( EnvFp= fopen( fname, "r"))){
			nn= fread( env, 1, sizeof(CX_env), EnvFp);
			fclose( EnvFp);
			if( !set && nn>= 0 && nn!= EOF ){
				env[nn]= '\0';
				if( env[0]== '\\' ){
					set= ( &env[1]);
				}
				else{
					set= ( env);
				}
			}
			where= "DSK";
		}
		else{
			errno= en;
		}
	}
	else{
		where= "ENV";
	}
	if( set ){
	 char *next_set;
	 /* variable was found: see if a prefixed one exists to
	  \ override it.
	  */
		if( getenv("GETENV_DEBUG") ){
			fprintf( StdErr, "xgraph::__GetEnv(\"%s\";\"%s\",\"%s\") = \"%s\" (%s)\n",
				name, n, pref, set, where
			);
		}
		if( nn>= 0 && nn!= EOF){
			if( env[0]=='$' ){
				if( /* strcmp( n, &env[1]) && */ strcmp( name, &env[1]) ){
				  /* disk env-variable refers to another	*/
					  /* recursiveness	*/
					strncpy( fname, &env[1], 255 );
					set= ( __GetEnv( fname, pref, False ) );
				}
				else{
					fprintf( StdErr, "xgraph::__GetEnv(): recursive variable '%s' (%s) == '%s'\n",
						n, name, &env[1]
					);
					fflush( StdErr);
				}
			}
			if( !set ){
				if( env[0]== '\\' ){
					set= ( &env[1]);
				}
				else{
					set= ( env);
				}
			}
		}
		  /* recursiveness	*/
		next_set= __GetEnv( name, pref, True );
		return( (next_set)? next_set : set );
	}
	if( getenv("GETENV_DEBUG") ){
		fprintf( StdErr, "xgraph::__GetEnv(\"%s\";\"%s\",\"%s\") = NULL\n", name, n, pref );
	}
	return( NULL);
}

/* return the value of variable
 * 1) (env) _<n>
 * 2) (disk) _<n>
 * 3) (env) <n>
 * 4) (disk) <n>
 */
char *_GetEnv( char *n)		/* find an env. var in EnvDir:<n>	*/
{  char *getenv();

	errno= 0;
/* by having __GetEnv() check prefix "" first, and then,
 \ if set, calling __GetEnv(n, prefix) recursively, one
 \ can get a real hierarchy!
 */
	return( __GetEnv( n, "_", False) );
}

int GetEnvDir()
{	char *ed;
	
	ed= EnvDir;					/* save EnvDir path	*/
	EnvDir= "";
	if( ( EnvDir= _GetEnv( "ENVDIR"))){
		strcpy( _EnvDir, EnvDir);		/* save the path we	found	*/
		EnvDir= _EnvDir;				/* set EnvDir to the saved path	*/
	}
	else
		EnvDir= ed;				/* restore default path	*/
	return( 1);
}

char *GetEnv( char *n)
{
	GetEnvDir();
	return( _GetEnv( n));
}

/* find an env. var in memory (set)	*/
/* or in env: (setenv)				*/
char *_SetEnv( char *n, char *v)
{	static char fail_env[]= "_SetEnv=failed (no memory)";
	char *nv;

	if( !n ){
		return(NULL);
	}
	if( !v ){
		v= "";
	}
	if( !(nv= calloc( strlen(n)+strlen(v)+2, 1L)) ){
		putenv( fail_env);
		return( NULL);
	}
	sprintf( nv, "%s=%s", n, v);
	putenv( nv);
	return( _GetEnv(n) );
}

char *SetEnv( char *n, char *v)
{
	GetEnvDir();
	return( _SetEnv( n, v));
}

int streq( char *a, char *b, int n )
{ int r= 0;
	if( b && *b && a && strncmp( b, "RE^", 3)== 0 && b[strlen(b)-1]== '$' ){
		if( re_comp( &b[2] ) ){
			fprintf( StdErr, "streq(\"%s\",rexp): can't compile regular expr. \"%s\" (%s)\n",
				a, &b[2], serror()
			);
			r= 0;
		}
		else{
			r= re_exec( a );
		}
	}
	else{
		if( n>= 0 ){
			r= !strncmp( a, b, n );
		}
		else{
			r= !strcmp( a, b );
		}
	}
	return(r);
}
