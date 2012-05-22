#ifndef _COMPILED_ASCANF_H
#define _COMPILED_ASCANF_H


/* !!! See the explanation of this flag in ascanf.h !!! */
/* Set to 1 to activate, 0 to deactivate. When not defined, the lines
 \ immediately after will activate it for platforms where it has an advantage.
 \ 20020419: this feature is always on for procedure calls with arguments.
 \ 20020421: tentatively put it back. It makes an ever-so-small difference (improvement).
 */
#define ASCANF_FORM_ARGVALS 1

#ifndef ASCANF_FORM_ARGVALS
// 20101019: the exclusion of gcc from supporting ASCANF_FORM_ARGVALS was probably for an OLD version...!
// #	ifndef __GNUC__
#		define ASCANF_FORM_ARGVALS 1
// #	endif
#endif


/* !!! See the explanation of this flag in ascanf.h !!! */
#define VERBOSE_CONSTANTS_EVALUATION 1



/* #include "dymod.h"	*/

/* 20020322: This header file contains some definitions that have to do with compiled
 \ ascanf code. In principle, one should NOT use this lightly. The only compelling
 \ reason I can see is when you want to access the <parent> field, which can be used
 \ to obtain the (name of the) calling function.
 */

/* Structure for making a compiled representation of an ascanf expression.
 \ 'value' contains the last evaluated expression value; 'fun' contains a
 \ pointer to an ascanf_Function, if this is a function expression. 'type'
 \ is mainly for debugging (is 'fun' correctly NULL/non-NULL?). If 'fun'
 \ is non-NULL, 'args' points to a list of <argc> Compiled_Forms that make up the
 \ arguments to this function. 'cdr' points to the next same-level value or
 \ expression. 'last_cdr' points to the last non-NULL expression (for easy
 \ appending). 'ok' contains the last evaluation returncode.
 \\
 */
typedef struct Compiled_Form{
	double value;
	ascanf_Function *fun;
	special_ascanf_Functions special_fun;
	char *expr;
	ascanf_type type;
	int store, sign, negate, take_address, take_usage, ok, argc, alloc_argc, list_of_constants, last_value, direct_eval,
		DyMod_Dependency, level, empty_arglist;
	struct Compiled_Form *args, *cdr;
	struct Compiled_Form *top, *last_cdr;
	struct Compiled_Form *parent;
	  /* af: currently only used for code belonging to a procedure: it points to the ascanf_Function entry 
	   \ in the exported tables that allow the user to reference the procedure. It allows to register a
	   \ Compiled_Form as a client of a DyMod, and in particular to delete the procedure when the DyMod
	   \ is unloaded.
	   */
	ascanf_Function *af;
	  /* 20050111: if take_address or take_usage, pointer contains the variable pointed to by value. */
	ascanf_Function *pointer;
	  /* 20061117: support for local variables. */
	ascanf_Function *vars_local_Functions;
	int local_Functions;
/* #if ASCANF_FORM_ARGVALS == 1	*/
	double *argvals;
/* #endif	*/
	struct DyModDependency *DyMod_DependList;
#ifdef XG_DYMOD_SUPPORT
	struct DyModLists *dymod;
#endif
	// 20080708
	double last_eval_time;
} Compiled_Form;

typedef struct Compiled_Form _Compiled_Form;

#ifndef ASCB_FRAME_EXPRESSION
#	undef AH_EXPR
#	define AH_EXPR	((__ascb_frame->compiled)? __ascb_frame->compiled->expr : NULL)
#endif

extern double af_ArgList_address;
#define SET_AF_ARGLIST(args,argc)	{\
	if( !(af_ArgList->array && (args) && (argc)==1 && *(args)== af_ArgList_address) ){ \
		af_ArgList->array= (args);\
		af_ArgList->value= af_ArgList->array[(af_ArgList->last_index= 0)]; \
		af_ArgList->N= (argc); \
	} \
}

extern void _ascanf_xfree(void *x, char *file, int lineno, ascanf_Function *af );
#define ascanf_xfree(af,x) {_ascanf_xfree((void*)(x),__FILE__,__LINE__,(af));(x)=NULL;}

extern int compile_procedure_code( int *n, char *source, double *result, ascanf_Function *af, int *level );
extern int call_compiled_ascanf_function( int Index, char **s, double *result, int *ok, char *caller, Compiled_Form **Form, int *level );

#endif
