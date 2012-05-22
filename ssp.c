#include <stdlib.h>
#include <stdio.h>
#include <errno.h>

extern int errno;
#if !defined(linux) && !defined(__MACH__) && !defined(__CYGWIN__)
	extern char *sys_errlist[];
#endif
#ifndef __CYGWIN__
#	define serror() ((errno>0&&errno<sys_nerr)?sys_errlist[errno]:"invalid errno")
#else
#	define serror()	strerror(errno)
#endif


int ssp( FILE *fp )
{  int c;
	  /* Skip all blank lines at the beginning of the file	*/
	do{
		c= getc(fp);
	} while( c!= EOF && c== '\n' );
	if( c== EOF ){
		return(0);
	}
	while( c!= EOF ){
		putchar(c);
		if( (c= getc(fp))== '\n' ){
			 /* first newline on this line: output it.	*/
			putchar( c);
			if( (c= getc(fp))== '\n' ){
				 /* yet another: this is a blank line. Output it.	*/
				putchar( c);
				 /* now skip all next blank lines.	*/
				while( (c= getc(fp))== '\n' );
			}
		}
		  /* to beginning of loop, with maybe a new, non-blank line.	*/
	}
	return(1);
}

int join( FILE *fp, int tabbed)
{  int c, ll= 0;
	do{
		if( (c= getc(fp))== '\n' ){
			if( (c= getc(fp))== '\n' ){
				if( tabbed ){
				  /* Empty line(s) insert as many tabs between the foldings	*/
					do{
						putchar( '\t');
					} while( (c= getc(fp))== '\n' && c!= EOF );
				}
				else{
					putchar( c);
					do{
						putchar( c);
					} while( (c= getc(fp))== '\n' && c!= EOF );
				}
				if( c!= EOF ){
					putchar(c);
					ll= 1;
				}
				else{
					ll= 0;
				}
			}
			else if( c!= EOF ){
				putchar( ' ');
				ll+= 1;
				if( isspace((unsigned char)c) ){
					do{
						c= getc(fp);
					} while( c!= EOF && isspace((unsigned char)c) );
				}
				if( c!= EOF ){
					putchar( c );
					ll+= 1;
				}
			}
		}
		else if( c!= EOF ){
			putchar( c);
			ll+= 1;
		}
	} while( c!= EOF );
	return(1);
}


main( int argc, char **argv )
{ FILE *fp;
  int i= 1;
  extern char *rindex( const char *, char);
  char *progname= rindex( argv[0], '/');
	if( progname ){
		progname++;
	}
	else{
		progname= argv[0];
	}
	if( argc> 1 ){
		while( argc> 1 ){
			if( (fp= fopen( argv[i], "r")) ){
				if( strcmp( progname, "ssp")== 0 ||
					strcmp( progname, "sspace")== 0
				){
					ssp( fp );
				}
				else if( strcmp( progname, "join")== 0 ){
					join( fp, 0 );
				}
				else if( strcmp( progname, "jointabbed")== 0 ){
					join( fp, 1 );
				}
				else{
					fprintf( stderr, "%s: don't know what to do when called like this...!\n", progname );
				}
				fclose( fp );
			}
			else{
				fprintf( stderr, "%s: can't open \"%s\": %s\n",
					progname, argv[i], serror()
				);
			}
			i+= 1;
			argc-= 1;
		}
	}
	else{
		if( strcmp( progname, "ssp")== 0 ||
			strcmp( progname, "sspace")== 0
		){
			ssp( stdin );
		}
		else if( strcmp( progname, "join")== 0 ){
			join( stdin, 0 );
		}
		else if( strcmp( progname, "jointabbed")== 0 ){
			join( stdin, 1 );
		}
		else{
			fprintf( stderr, "%s: don't know what to do when called like this...!\n", progname );
		}
	}
	exit( 0 );
}
