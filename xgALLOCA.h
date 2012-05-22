#ifndef ALLOCA

#ifdef __cplusplus
	extern "C" {
#endif

	extern int local_buf_size;
#	define LMAXBUFSIZE	local_buf_size

/* NB: some gcc versions (linux 2.9.?) don't like a declaration of the type 
 \ type x[len+1]
 \ claiming that the size of x is too large. Bug or feature, it can be remedied
 \ by using x[len+2] instead of x[len+1], *or* by x[LEN], where LEN=len+1.
 */

/* Set this to some value (in Mb) if you want to perform size-checking on the size argument passed to
 \ the ALLOCA() macro (defined in xgALLOCA.h; expands to a system-dependent interface to the alloca()
 \ memory allocator). The need for this will depend on how your system handles alloca allocation failures.
 \ The IRIX alloca() routine (and gcc's too) is guaranteed not to fail -- i.e. it always returns a non-null
 \ pointer. If however you ask for some huge amount of memory (e.g. -1 :)), the process will be killed by
 \ the system, without leaving any leeway to the programme to handle the situation gracefully. Also,
 \ utilities like Electric Fence do not trace allocations using alloca-like allocators.
 \ The size checking offered by xgraph does not alter this, but it will print an informative message whenever
 \ more memory is requested, which allows to trace the location where the allocation was made.
 \ It will also likely slow down the programme.
 \ To activate, set CHKLCA_MAX to something (e.g. 128 to warn about allocations larger than 128Mb). Don't
 \ remove the #ifndef pair around the definition; this makes it possible to do file-specific activation.
 */
#ifndef CHKLCA_MAX
#	ifdef __CYGWIN__
		// 20100331: mystery, my way of calling into alloca() fails on cygwin unless I call through check_alloca_size()...
#		define CHKLCA_MAX	-1
#	else
#		define CHKLCA_MAX	0
#	endif
#endif

#if CHKLCA_MAX > 0

#	ifdef __GNUC__
inline
#	endif
  /* 20020501 RJVB */
static size_t check_alloca_size(size_t n, size_t s, size_t max, char *file, int lineno )
{
	if( n*s > (max << 20) ){
		fprintf( stderr, "\ncheck_alloca_size(%lu,%lu): request for %g>%luMb in %s, line %d!\n\n",
			n, s, ( (n*s) / (double)( 1 << 20) ), max,
			file, lineno
		);
		fflush( stderr );
	}
	return(n);
}

#	if defined(__GNUC__) && !defined(DEBUG)
#		define CHKLCA(len,type)	check_alloca_size((size_t)(len),sizeof(type),CHKLCA_MAX,__FILE__,__LINE__)
#	else
			/* 20040122 this macro had 1 instead of sizeof(type); for non GNU compilers??? */
#		define CHKLCA(len,type)	check_alloca_size((size_t)(len),sizeof(type),CHKLCA_MAX,__FILE__,__LINE__)
#	endif

#elif CHKLCA_MAX < 0

#	ifdef __GNUC__
__attribute__ ((noinline))
#	endif
static void handle_neg_size(long *n)
{
	fprintf( stderr, "\ncheck_alloca_size_dum(%lu): request for signed length <= 0\n\n",
		*n
	);
	fflush( stderr );
}

#	ifdef __GNUC__
__attribute__ ((noinline))
#	endif
static size_t check_alloca_size_dum(size_t n)
{
	if( ((long) n) <= 0 ){
		handle_neg_size( (long*) &n );
	}
	return(n);
}

#	define CHKLCA(len,type)	check_alloca_size_dum((size_t)len)

#else

#	define CHKLCA(len,type)	len

#endif

extern void *xgalloca(unsigned int n, char *file, int linenr);

#define HAS_alloca

#	if defined(__GNUC__) && !defined(DEBUG) /* && !defined(__MACH__) */
#		define XGALLOCA(x,type,len,xlen,dum)	type x[CHKLCA(len,type)]; int xlen= sizeof(x)
#		define ALLOCA(x,type,len,xlen)	type x[CHKLCA(len,type)]
#		define _ALLOCA(x,type,len,xlen)	type x[CHKLCA(len,type)]; int xlen= sizeof(x)
#		define __ALLOCA(x,type,len,xlen)	type x[CHKLCA(len,type)]; int xlen= sizeof(x)
#		define GCA()	/* */
#		define __GCA()	/* */
#	else
	/* #	define LMAXBUFSIZE	MAXBUFSIZE*3	*/
#		ifndef STR
#		define STR(name)	# name
#		endif
		extern void *XGalloca(void **ptr, int items, int *alloced_items, int size, char *name);
#		define XGALLOCA(x,type,len,xlen,dum)	static int xlen= -1; static type *x=NULL; type *dum=(type*) XGalloca((void**)&x,CHKLCA(len,type),&xlen,sizeof(type),STR(x))

#		if defined(HAS_alloca)
#			define __ALLOCA(x,type,len,xlen)	int xlen=(CHKLCA(len,type))* sizeof(type); type *x= (type*) alloca(xlen)
#			define __GCA()	/* */
#		else
#			define __ALLOCA(x,type,len,xlen)	int xlen=(CHKLCA(len,type))* sizeof(type); type *x= (type*) xgalloca(xlen,__FILE__,__LINE__)
#			define __GCA()	xgalloca(0,__FILE__,__LINE__)
#		endif
#		if defined(HAS_alloca) && !defined(DEBUG)
#			include <alloca.h>
#			define ALLOCA(x,type,len,xlen)	__ALLOCA(x,type,CHKLCA(len,type),xlen)
#			define _ALLOCA(x,type,len,xlen)	__ALLOCA(x,type,CHKLCA(len,type),xlen)
#			define GCA()	/* */
#		else
#			define ALLOCA(x,type,len,xlen)	int xlen=(CHKLCA(len,type))* sizeof(type); type *x= (type*) xgalloca(xlen,__FILE__,__LINE__)
#			define _ALLOCA(x,type,len,xlen)	ALLOCA(x,type,CHKLCA(len,type),xlen)
#			define GCA()	xgalloca(0,__FILE__,__LINE__)
#		endif

#	endif

  /* should work alike on all platforms (this is getting a mess!!!) 	*/
#define _XGALLOCA(x,type,len,xlen)	int xlen=(CHKLCA(len,type))* sizeof(type); type *x= (type*) xgalloca(xlen,__FILE__,__LINE__)
#define _GCA()	xgalloca(0,__FILE__,__LINE__)

#include "xfree.h"

#if !defined(sgiKK) && !defined(alloca)
#	define _REALLOCA(x,type,len,xlen)	xlen=(CHKLCA(len,type))* sizeof(type); x= (type*) xgalloca(xlen,__FILE__,__LINE__)
#else
#	define _REALLOCA(x,type,len,xlen)	xlen=(CHKLCA(len,type))* sizeof(type); x= (type*) xgalloca(xlen,__FILE__,__LINE__)
#endif

#if !defined(__GNUC__) && !defined(HAS_alloca)
#	undef alloca
#	define alloca(n) xgalloca(CHKLCA((n),char),__FILE__,__LINE__)
#endif

#ifdef __cplusplus
	}
#endif

#endif
