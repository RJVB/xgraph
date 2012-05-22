#ifndef _XFREE_H

#ifdef __cplusplus
extern "C" {
#endif

	extern FILE *StdErr;
	extern void _xfree(void *x, char *file, int lineno);
#	ifdef _DATA_SET_H
		extern void _xfree_setitem(void *x, char *file, int lineno, DataSet *this_set );
#	endif

#	ifndef NDEF_XFREE
#		define xfree(x) {_xfree((void*)(x),(char*)__FILE__,__LINE__);(x)=NULL;}
#	endif //NDEF_XFREE

#	ifndef NDEF_XFREE_SETITEM
#		define xfree_setitem(x,this_set) {_xfree_setitem((void*)(x),__FILE__,__LINE__,this_set);(x)=NULL;}
#	endif //NDEF_XFREE_SETITEM

#ifdef __cplusplus
}
#endif

#define _XFREE_H
#endif
