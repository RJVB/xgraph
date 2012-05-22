/* 
	alloca -- (mostly) portable public-domain implementation -- D A Gwyn
	20020501: renamed to xgalloca by RJVB

	last edit:	86/05/30	rms
	   include config.h, since on VMS it renames some symbols.

	This implementation of the PWB library alloca() function,
	which is used to allocate space off the run-time stack so
	that it is automatically reclaimed upon procedure exit, 
	was inspired by discussions with J. Q. Johnson of Cornell.

	It should work under any C implementation that uses an
	actual procedure stack (as opposed to a linked list of
	frames).  There are some preprocessor constants that can
	be defined when compiling for your specific system, for
	improved efficiency; however, the defaults should be okay.

	The general concept of this implementation is to keep
	track of all alloca()-allocated blocks, and reclaim any
	that are found to be deeper in the stack than the current
	invocation.  This heuristic does not reclaim storage as
	soon as it becomes invalid, but it will do so eventually.

	As a special case, alloca(0) reclaims storage without
	allocating any.  It is a good idea to use alloca(0) in
	your main control loop, etc. to force garbage collection.
*/
#ifndef lint
static char	SCCSid[] = "$Id: @(#)alloca.c	1.1 $";	/* for the "what" utility */
#endif

/* #ifdef HAVE_CONFIG_H	*/
#	include "config.h"
/* #endif	*/

/* IDENTIFY( "alloca implementations" );	*/

#ifndef _XGRAPH_H
#	include <stdio.h>
#	include <stdlib.h>

#	ifndef StdErr
		extern FILE *StdErr;
#	endif
#endif

#include "Macros.h"

/* 20020418: a different way of allocation, that should never cause alignment problems: */
#undef RJVB

#ifdef emacs
#	ifdef static
/* actually, only want this if static is defined as ""
   -- this is for usg, in which emacs must undefine static
   in order to make unexec workable
   */
#		ifndef STACK_DIRECTION
you
lose
-- must know STACK_DIRECTION at compile-time
#		endif /* STACK_DIRECTION undefined */
#	endif /* static */
#endif /* emacs */

extern void	free();

/*
	Define STACK_DIRECTION if you know the direction of stack
	growth for your system; otherwise it will be automatically
	deduced at run-time.

	STACK_DIRECTION > 0 => grows toward higher addresses
	STACK_DIRECTION < 0 => grows toward lower addresses
	STACK_DIRECTION = 0 => direction of growth unknown
*/

#if defined(__APPLE__) && defined(__PPC__) && !defined(__GNUC__)
#	define STACK_DIRECTION	1
#endif

#ifndef STACK_DIRECTION
#	define	STACK_DIRECTION	0		/* direction unknown */
#endif

#if STACK_DIRECTION != 0

#	define	STACK_DIR	STACK_DIRECTION	/* known at compile-time */

#else	/* STACK_DIRECTION == 0; need run-time code */

static int	stack_dir = 0;		/* 1 or -1 once known */
#	define	STACK_DIR	stack_dir

static void find_stack_direction (/* void */)
{
  static char	*addr = NULL;	/* address of first
				   `dummy', once known */
  auto char	dummy;		/* to get stack address */

  if (addr == NULL)
    {				/* initial entry */
	void (*stdir)()= find_stack_direction;
		addr = &dummy;

		(*stdir) (); /* recurse once */
    }
  else{				/* second entry */
    if (&dummy > addr)
      stack_dir = 1;		/* stack grew upward */
    else
      stack_dir = -1;		/* stack grew downward */
#ifdef DEBUG
	 fprintf( StdErr, "xgalloca()::StackDirection: %s\n", (stack_dir<0)? "down" : "up" ); fflush(StdErr);
#endif
  }
}

#endif	/* STACK_DIRECTION == 0 */

/*
	An "alloca header" is used to:
	(a) chain together all alloca()ed blocks;
	(b) keep track of stack depth.

	It is very important that sizeof(header) agree with malloc()
	alignment chunk size.  The following default should work okay.
	20020418 RJVB: the default was just sizeof(double); this would of course
	\ not work since the struct{}h in the header was larger than a single double.
	\ Increasing the alignment correction to 2 doubles corrects this.
*/

#ifndef	ALIGN_SIZE
#	define	ALIGN_SIZE	2*sizeof(double)
#endif

#ifdef RJVB
 /* A header definition that has no alignment problems, but requires 2 allocations: */
typedef struct header{
	struct{
		struct header *next;
		char *deep;
		unsigned long size;
	} h;
	void *memory;
} header;
#else
typedef union hdr{
  char	align[ALIGN_SIZE];	/* to force sizeof(header) */
  struct
    {
      union hdr *next;		/* for chaining headers */
      char *deep;		/* for stack depth measure */
	 unsigned long size;
    } h;
} header;
#endif

/*
	alloca( size ) returns a pointer to at least `size' bytes of
	storage which will be automatically reclaimed upon exit from
	the procedure that called alloca().  Originally, this space
	was supposed to be taken from the current stack frame of the
	caller, but that method cannot be made to work for some
	implementations of C, for example under Gould's UTX/32.
*/

static header *last_alloca_header = NULL; /* -> last alloca header */
static header *first_alloca_header = NULL; /* -> first alloca header */

__attribute__ ((noinline)) void *xgalloca (size, file, lineno)			/* returns pointer to storage */
     unsigned	int size;		/* # bytes to allocate */
	char *file;
	int lineno;
{
  auto char	probe;		/* probes stack depth: */
  register char	*depth = &probe;

#if STACK_DIRECTION == 0
  if (STACK_DIR == 0){		/* unknown growth direction */
   void (*stdir)()= find_stack_direction;
    (*stdir) ();
  }
#endif

				/* Reclaim garbage, defined as all alloca()ed storage that
				   was allocated from deeper in the stack than currently. */

  {
    register header	*hp;	/* traverses linked list */

    hp = last_alloca_header;
    while( hp ){
      if( (STACK_DIR > 0 && hp->h.deep > depth)
	  || (STACK_DIR < 0 && hp->h.deep < depth)
	    /* RJVB: de-allocate everything upto and INCLUDING the current stack frame */
	  || (size==0 && hp->h.deep== depth)
	){
	  header	*np = hp->h.next;

#if DEBUG > 1
	if( size ){
	  int i;
#ifdef RJVB
	  char *c= hp->memory;
#else
	  char *c= (char*) hp+ sizeof(header);
#endif
 		fprintf(StdErr,"xgalloca(%d)-%s,line %d: Freeing 0x%lx[%lu] < ", size, file, lineno, c, hp->h.size ); fflush(StdErr);
		for( i= 0; i< sizeof(int); i++ ){
 			fprintf( StdErr, "%x ", c[i] );
		}
 		fprintf( StdErr, "> \"" );
		for( i= 0; i< sizeof(int); i++ ){
 			fprintf( StdErr, "%c", c[i] );
		}
 		fputs( "\"\n", StdErr );
	}
#endif
#ifdef RJVB
	  free( hp->memory );
	  hp->memory= NULL;
#endif
	  free ((void *) hp);	/* collect garbage */

	  hp = np;		/* -> next header */
	}
      else
	break;			/* rest are not deeper */
    }

    last_alloca_header = hp;	/* -> last valid storage */
    if( !last_alloca_header ){
	    first_alloca_header= NULL;
    }
  }

  if (size == 0)
    return NULL;		/* no allocation required */

  /* Allocate combined header + user data storage. */

  {
#ifdef RJVB
    register void *	new = malloc( sizeof (header) );
	if( new ){
		((header*)new)->memory= calloc( 1, size );
	}
#else
    register void *	new = calloc (1, sizeof (header) + size);
#endif
    /* address of header */

    if( !new ){
	    return(NULL);
    }

    ((header *)new)->h.next = last_alloca_header;
    ((header *)new)->h.deep = depth;
    ((header *)new)->h.size = size;
    if( !last_alloca_header ){
	    first_alloca_header= (header*) new;
    }

    last_alloca_header = (header *)new;
#if DEBUG > 1
    fprintf( StdErr, "xgalloca(%d,%s::%d): 0x%lx->0x%lx ... ->0x%lx->0x%lx\n",
		size, file, lineno,
    		new, ((header*)new)->h.next, first_alloca_header, first_alloca_header->h.next
	);
#endif

#ifdef RJVB
	return( ((header*)new)->memory );
#else
    /* User storage begins just after header. */
    return (void *)((char *)new + sizeof(header));
#endif
  }
}

#include <signal.h>

/* gcc has dynamical array-allocation, meaning that one can specify, say, 
 \ double foo[arg1];
 \ as a local variable. foo is than allocated as an array of 100 doubles, on the
 \ stack (probably...). Very convenient, although no error-checking is done (in case
 \ the variable arg1 is very large, e.g.). The routine used for this used to be called
 \ alloca(), which I simulate here. This routine can be called with the macro ALLOCA, which
 \ than makes sure the proper calling sequence is generated, to (re)allocate a static, local
 \ variable (yes, I know static variables must be assigned constant values at compile-time. I do.
 \ NULL... And I define an extra, dummy variable (sorry.. have to specify the name yourself..)
 \ which is initialised with the function-call to alloca()).
 \ Passed a pointer to your static variable, the (new) number of items, the previous number of items
 \ (stored in a static var. that you have to specify), and the size per item, this routine either allocates
 \ (using calloc) or reallocates (using realloc) the requested memory. Drawback: this is static memory that
 \ is not de-allocated upon exit of the variable's scope. As far as I can see, this can't be done automatically.
 \ Advantage: it gives more information, and exits in a (possibly nice..) way when things do go wrong. The disadvantage
 \ of that is of course that it may be less convenable to use for allocations other than of the type just described.
 \ Note: under gcc, the ALLOCA() macro just expands to a "normal", dynamic allocation of a local variable.
 */
void *XGalloca(void **ptr, int items, int *alloced_items, int size, char *name)
{ extern int debugFlag;
	if( *alloced_items<= 0 || items> *alloced_items || !ptr ){
	  void *old= *ptr;
		if( !*ptr ){
			*ptr= calloc( items, size );
			if( debugFlag ){
				fprintf( StdErr, "XGalloca(): \"%s\" %d items, size %d at 0x%lx\n",
					name, items, items* size, *ptr
				);
			}
		}
		else{
			*ptr= realloc( *ptr, items* size);
			if( debugFlag ){
				fprintf( StdErr, "XGalloca(): reallocated \"%s\", %d items, size %d at 0x%lx\n",
					name, items, items* size, *ptr
				);
			}
		}
		if( !*ptr ){
			fprintf( StdErr, "XGalloca(): can't realloc mem \"%s\" (0x%lx) (%d items) to %d items of size %d (%s)\n",
				name, *ptr, *alloced_items, items, size, serror()
			);
			raise( SIGHUP );
			  /* If that doesn't exit us nicely, we just do:	*/
			exit(-10);
		}
		else{
			*alloced_items= items;
		}
	}
	return( *ptr );
}

#ifdef __CYGWIN__0

void handle_neg_size(long *n)
{
	fprintf( stderr, "\ncheck_alloca_size_dum(%lu): request for signed length <= 0\n\n",
		n
	);
	fflush( stderr );
}

#endif

