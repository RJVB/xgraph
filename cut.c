/*
 * Copyright (c) 1989 The Regents of the University of California.
 * All rights reserved.
 *
 * This code is derived from software contributed to Berkeley by
 * Adam S. Moskowitz of Menlo Consulting and Marciano Pitargue.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *	This product includes software developed by the University of
 *	California, Berkeley and its contributors.
 * 4. Neither the name of the University nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

#ifndef _POSIX2_LINE_MAX
#	define	_POSIX2_LINE_MAX 1024
#endif

#ifndef REGISTER
#	define REGISTER /* */
#endif

int _posix2_line_max= _POSIX2_LINE_MAX;

#ifndef lint
char copyright[] =
"@(#) Copyright (c) 1989 The Regents of the University of California.\n\
 All rights reserved.\n";
#endif /* not lint */

#ifndef lint
static char sccsid[] = "@(#)cut.c	5.4 (Berkeley) 10/30/90";
#endif /* not lint */

#include <stdlib.h>

#include <limits.h>
#include <stdio.h>
#include <ctype.h>
#include <errno.h>

extern int errno;
#if !defined(linux) && !defined(__MACH__)
	extern char *sys_errlist[];
#endif

int	cflag;
char	dchar;
int	dflag;
int	fflag;
int	sflag;

char *positions;
char Substitute_Char, substitute_char= '\0', *Substitute_Switch= NULL;

main(argc, argv)
	int argc;
	char **argv;
{
	extern char *optarg;
	extern int errno, optind;
	FILE *fp;
	int ch, last_opt, (*fcn)(), c_cut(), f_cut();
	char *strerror();
	int subst_switch_on= 0;

	dchar = '\t';			/* default delimiter is \t */

	if( !(positions= calloc( _posix2_line_max+1, 1)) ){
		exit(1);
	}

	last_opt= 0;
	while( (ch = getopt(argc, argv, "c:d:S:M:f:s01l:")) != EOF){
		switch(ch) {
			case 'c':
				fcn = c_cut;
				get_list(optarg);
				cflag = 1;
				break;
			case 'd':
				dchar = *optarg;
				dflag = 1;
				break;
			case 'S':
				substitute_char = *optarg;
				Substitute_Char = *optarg;
				break;
			case 'M':
				Substitute_Switch = optarg;
				break;
			case '0':
				if( last_opt== 'M' ){
					subst_switch_on= 1;
				}
				else{
					fprintf( stderr, "cut: -0 option not defined for previous option -%c\n", last_opt );
				}
				break;
			case '1':
				if( last_opt== 'M' ){
					subst_switch_on= 0;
				}
				else{
					fprintf( stderr, "cut: -1 option not defined for previous option -%c\n", last_opt );
				}
				break;
			case 'f':
				get_list(optarg);
				fcn = f_cut;
				fflag = 1;
				break;
			case 's':
				sflag = 1;
				break;
			case 'l':{
			  int l= atoi(optarg);
				if( l ){
					_posix2_line_max= l;
					if( !( positions= realloc( positions, _posix2_line_max+1)) ){
						fprintf( stderr, "cut: can't allocate %d byte linebuffer (%s)\n",
							_posix2_line_max, sys_errlist[errno]
						);
						exit(1);
					}
				}
				else{
					fprintf( stderr, "cut: invalid linelength specified (-l %s=%d), using default (%d)\n",
						optarg, l, _posix2_line_max
					);
				}
				break;
			}
			case '?':
			default:
				usage();
		}
		last_opt= ch;
	}
	argc -= optind;
	argv += optind;

	if (fflag) {
		if (cflag){
			usage();
		}
	} else if (!cflag || dflag || sflag){
		usage();
	}

	if( subst_switch_on== 0 && Substitute_Switch ){
		substitute_char= 0;
	}

	if( Substitute_Char && cflag ){
		fprintf( stderr, "cut: -S option ignored in combination with -c option\n");
		substitute_char= 0;
		Substitute_Char= 0;
	}

	if (*argv){
		for (; *argv; ++argv) {
			if (!(fp = fopen(*argv, "r"))) {
				(void)fprintf(stderr,
				    "cut: %s: %s\n", *argv, strerror(errno));
				exit(1);
			}
			fcn(fp, *argv);
		}
	}
	else{
		fcn(stdin, "stdin");
	}
	exit(0);
}

int autostart, autostop, maxval;

get_list(list)
	char *list;
{
	REGISTER char *pos;
	REGISTER int setautostart, start, stop;
	char *p, *strtok();

	/*
	 * set a byte in the positions array to indicate if a field or
	 * column is to be selected; use +1, it's 1-based, not 0-based.
	 * This parser is less restrictive than the Draft 9 POSIX spec.
	 * POSIX doesn't allow lists that aren't in increasing order or
	 * overlapping lists.  We also handle "-3-5" although there's no
	 * real reason too.
	 */
	for (; p = strtok(list, ", \t"); list = NULL) {
		setautostart = start = stop = 0;
		if (*p == '-') {
			++p;
			setautostart = 1;
		}
		if (isdigit(*p)) {
			start = stop = strtol(p, &p, 10);
			if (setautostart && start > autostart)
				autostart = start;
		}
		if (*p == '-') {
			if (isdigit(p[1]))
				stop = strtol(p + 1, &p, 10);
			if (*p == '-') {
				++p;
				if (!autostop || autostop > stop)
					autostop = stop;
			}
		}
		if (*p)
			badlist("illegal list value");
		if (!stop || !start)
			badlist("values may not include zero");
		if (stop > _posix2_line_max) {
			/* positions used rather than allocate a new buffer */
			(void)sprintf(positions, "%d too large (max %d)",
			    stop, _posix2_line_max);
			badlist(positions);
		}
		if (maxval < stop)
			maxval = stop;
		for (pos = positions + start; start++ <= stop; *pos++ = 1);
	}

	/* overlapping ranges */
	if (autostop && maxval > autostop)
		maxval = autostop;

	/* set autostart */
	if (autostart)
		memset(positions + 1, '1', autostart);
}

/* ARGSUSED */
c_cut(fp, fname)
	FILE *fp;
	char *fname;
{
	REGISTER int ch, col;
	REGISTER char *pos;

	for (;;) {
		pos = positions + 1;
		for (col = maxval; col; --col) {
			if ((ch = getc(fp)) == EOF){
				return;
			}
			if( ch == '\n' ){
				break;
			}
			if (*pos++){
				putchar(ch);
			}
		}
		if (ch != '\n'){
			if (autostop){
				while ((ch = getc(fp)) != EOF && ch != '\n'){
					putchar(ch);
				}
			}
			else{
				while ((ch = getc(fp)) != EOF && ch != '\n');
			}
		}
		putchar('\n');
	}
}

f_cut(fp, fname)
	FILE *fp;
	char *fname;
{
	REGISTER int ch, field, isdelim;
	REGISTER char *pos, *p, sep;
	int sep_output;
#ifdef __GNUC__
	char lbuf[_posix2_line_max + 2];
#else
	char *lbuf= calloc( _posix2_line_max+1, 1);

	if( !lbuf ){
		fprintf( stderr, "cut: can't allocate %d temp buffer (%s)\n",
			_posix2_line_max, sys_errlist[errno]
		);
		exit(1);
	}
#endif

	for (sep = dchar; fgets(lbuf, _posix2_line_max+1, fp);) {
	  int eol= 0;
		if( Substitute_Switch ){
		  int len= strlen(lbuf);
		  char c= 0;
			if( lbuf[len-1]== '\n' ){
				c= lbuf[len-1];
				lbuf[len-1]= '\0';
			}
			if( !strcmp( Substitute_Switch, lbuf) ){
				substitute_char= (substitute_char)? '\0' : Substitute_Char;
			}
			if( c ){
				lbuf[len-1]= c;
			}
		}
		  /* reset the separator output switch at the beginning of each line	*/
		sep_output= 0;
		for (isdelim = 0, p = lbuf; ; ++p) {
			if (!(ch = *p)) {
				(void)fprintf(stderr,
				    "cut: %s: line too long.\n", fname);
				exit(1);
			}
			/* this should work if newline is delimiter */
			if (ch == sep){
				isdelim = 1;
			}
			if (ch == '\n') {
				if (!isdelim && !sflag){
					(void)printf("%s", lbuf);
				}
				break;
			}
		}
		if (!isdelim)
			continue;

		pos = positions + 1;
		for (field = maxval, p = lbuf; field; --field, ++pos) {
			if (*pos) {
/* why output the separator?????	*/
/* 960122 RJB: Of course. You wouldn't want all cut columns to be
 \ glued against each other into one big column! Only the first column
 \ shouldn't be preceded by the separator!
 */
				if( sep_output ){
					putchar(sep); 
				}
				sep_output++;
				if( !eol ){
					while( (ch = *p++) != '\n' && ch != sep){
						putchar(ch);
					}
				}
				else{
					putchar( substitute_char );
				}
			}
			else if( !eol ){
				while ((ch = *p++) != '\n' && ch != sep);
			}
			  /* end of the line?	*/
			if( ch == '\n' && !substitute_char ){
				break;
			}
			if( ch == '\n' ){
				eol= 1;
			}
		}
		if( ch != '\n'){
			if (autostop) {
				if (sep_output){
					putchar(sep); 
				}
				if( !eol ){
					for (; (ch = *p) != '\n'; ++p){
						putchar(ch);
					}
				}
				else{
					putchar( substitute_char );
				}
			} else{
				  /* so, what's this line supposed to do??!
				   \ outcommented RJB 960404
				   */
/* 				for (; (ch = *p) != '\n'; ++p);	*/
			}
		}
		putchar('\n');
	}
#ifndef __GNUC__
	free( lbuf );
#endif
}

badlist(msg)
	char *msg;
{
	(void)fprintf(stderr, "cut: [-cf] list: %s.\n", msg);
	exit(1);
}

usage()
{
	(void)fprintf(stderr,
"usage:\tcut -c list [file1 ...]\n\tcut -f list [-s] [-d delim] [-S subst] [-M subst_switch] [-l maxlinelength] [-0|-1] [file ...]\n");
	exit(1);
}
