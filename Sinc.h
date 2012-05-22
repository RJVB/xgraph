#ifndef _SINC_H
#	define _SINC_H

#include "defun.h"

typedef enum SincType { SINC_STRING=0xABCDEF09, SINC_FILE=0x4AF3BD8C } SincType;

typedef enum SincString{ SString_Fixed, SString_Dynamic, SString_Global } SincString;
extern int SincString_Behaviour;

typedef struct Sinc{
	union file_or_string{
		char *string;
		FILE *file;
	} sinc;
	SincType type;
	long _cnt, _base, _tlen, alloc_len;
	SincString behaviour;
} Sinc;


DEFUN( Sinc_string, ( Sinc *sinc, char *string, long _cnt, long base), Sinc* );
DEFUN( Sinc_string_behaviour, ( Sinc *sinc, char *string, long _cnt, long base, SincString behaviour), Sinc* );
extern int SincString_Behaviour;
DEFUN( SincAllowExpansion, (Sinc *sinc), int );
DEFUN( Sinc_file, ( Sinc *sinc, FILE *file, long cnt, long base), Sinc* );
DEFUN( Sinc_base, ( Sinc *sinc, long base), Sinc* );
DEFUN( Sputs, ( char *text, Sinc *sinc), int );
DEFUN( SSputs, ( char *text, Sinc *sinc), Sinc* );
DEFUN( Sputc, ( int a_char, Sinc *sinc), int );
DEFUN( SSputc, ( int a_char, Sinc *sinc), Sinc* );
DEFUN( Sflush, (Sinc *sinc), int );
DEFUN( Srewind, (Sinc *sinc), int );

#define Seof(Sinc)	(Sinc->_cnt>0 && Sinc->_base>= Sinc->_cnt)
#define Serror(Sinc)	(!Sinc || !Sinc->sinc.string || (Sinc->type!=SINC_STRING && Sinc->type!=SINC_FILE))
#define _Sflush(Sinc)	{if(Sinc->type==SINC_FILE)Flush_File(Sinc->sinc.file);}

#ifndef ENODATA
#	define ENODATA	ENOENT
#endif

#endif
