#ifndef _ASC_TABLE_H
#define _ASC_TABLE_H

/* 20020428: moved the builtin-functions table to ascanfc-table.c . The Makefile generates a corresponding headerfile
 \ that contains all the necessary (= known) callback routines (identified through their use of the ASCB_ARGLIST macro!!)
 \
 \ No source file should need to be recompiled when this file changes (as it contains only function definitions).
 \
 \ This header file is generated automatically by the Makefile. Do not edit!
 \
 */


static pragma_unused char *ath= "$Id: @(#) " __FILE__ " [" __DATE__ "," __TIME__ "] $";

#ifndef _ASCANFC_C
/* 	CAUTION: the arguments have to be doubles, even constants (the compiler won't know what to promote to...)	*/
	extern double ASCB_call( ASCANF_CALLBACK( (*function) ), int *success, int level, char *expr, int max_argc, int argc, ... );
#endif
