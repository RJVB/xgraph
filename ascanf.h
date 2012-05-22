#ifndef _ASCANF_H
#define _ASCANF_H

#include "pragmas.h"

/* Header file for the ascanf functionality, that allows the reading (a la scanf) and parsing of arrays.
 \ These arrays can be nested expressions (Lisp-like) of almost arbitrary complexity.
 */

/* user-controllable compile flags and macros: */

/* The default maximum number of elements that can be read in a single call. Can be changed at runtime */
#define AMAXARGSDEFAULT		256

/* Whether or not to use the "new" callback parameter passing mechanism, or the original one. The original
 \ one is probably faster, but is much less flexible. There should be no need to undefine this flag.
 */
#define ASCANF_ALTERNATE

/* 20020413: A flag allowing debugging of the ascanf argument handling in compiled expressions. With this
 \ flag defined and ascanf_verbose==2 (do $verbose[2]), each callback prints the expression that invocated
 \ it before it executes its code. This allows to trace callbacks that misbehavingly write or read arguments
 \ that were not passed and thus not allocated. More functionality may be implemented.
 \ There is of course some (time) penalty associated with this this feature...
 \ NB: compiling with the DEBUG switch defined (as done by -g when using gccopt) has the same effect!!
 */
#undef ASCANF_ARG_DEBUG

/* 20020416: a flag that controls whether automatically created "local" (internal) variables (arrays, strings,...)
 \ have a unique name or not. Normally, these variables' names are identical to their (initial) value. This
 \ can cause name collisions; problably not a problem for strings, but more likely to be problematic in the case
 \ of automatic arrays. Setting this flag causes a simple serial number to be appended to the names, ensuring that
 \ such collisions will not happen.
 \ Value 1: only applied to arrays.
 \ Value 2: also applied to strings.
 */
#define ASCANF_AUTOVARS_UNIQUE 1

/* 20020417: when this flag is defined, the array of doubles that holds the arguments for each compiled
 \ is allocated at compile time, and stored in the frame (Compiled_Form) that "describes" that expression.
 \ This prevents the need for runtime allocation and de-allocation, at the expense of larger memory requirements.
 \ Note that this will not always make a huge difference (probably especially not when using gcc which has an
 \ efficient builtin alloca()-like function for allocation on the stack).
 \ 20020418: default to off. I have not found any advantage at all when using gcc or other compilers having
 \ efficient on-the-stack-allocation (alloca). Such allocation appears not to take significant time for the
 \ amounts typically required, and apparently yields memory that can be accessed faster (because activating
 \ the option often results in slower execution!). Things may be different on machines that do not have
 \ the required alloca() implementation, in which case the provided alloca.c can be used (which uses calloc()).
 */
/* !!! This flag is set in compiled_ascanf.h !!! */

/* 20020601: VERBOSE_CONSTANTS_EVALUATION, a flag controlling whether or not verbose mode exists in the
 \ evaluation of constants lists. Setting this flag makes verbose output of expressions containing
 \ constants lists (i.e. with $UseConstantsLists True) somewhat more cryptic as names of the constants
 \ (variables and arrays) do not show up in the output, only their evaluated value. However, it speeds up
 \ evaluation by 2-3%... (in non-verbose mode; this is due to not checking the verbose flag...!)
 */
/* !!! This flag is set in compiled_ascanf.h !!! */




/* end user-controllable section; below here, there is not really anything to meddle with! */



#include <stdint.h>

#include <math.h>

#ifdef __GNUC__
#	include <sys/types.h>
/* #	include <varargs.h>	*/
/* #	define _STDARG_H	*/
#else
/* #	include <varargs.h>	*/
#endif

#if defined(__APPLE_CC__)
#	define Boolean	macBoolean
/* #	include <Accelerate/Accelerate.h>	*/
#	undef Boolean
#	undef pixel
#endif

#include <time.h>

#include "Sinc.h"

#ifdef __cplusplus
	extern "C" {
#endif

/* extern char *index();	*/

#ifdef _MAC_MPWC_
#	define fgets _fgets
#endif

#define ASCANF_DATA_COLUMNS	4

#define VAR_CHANGED	'N'
#define VAR_REASSIGN	'R'
#define VAR_UNCHANGED	'o'

#define CHANGED_FIELD_OFFSET	(2*sizeof(unsigned short))
#define CHANGED_FIELD(vars)	(((vars)->changed)?(vars)->changed->flags:NULL)
#define CHANGED_FIELD_INDEX(vars,i)	(((vars)->changed)?(vars)->changed->flags[(i)]:0)

#define RESET_CHANGED_FLAG(ch,i) if(ch){\
		ch[i]= VAR_UNCHANGED;\
	}

#define SET_CHANGED_FLAG(ch,i,a,b,scanf_ret) if(ch){\
		if( scanf_ret== EOF)\
			ch[i]= VAR_UNCHANGED;\
		else if( a== b)\
			ch[i]= VAR_REASSIGN;\
		else if( a!= b)\
			ch[i]= VAR_CHANGED;\
	}

#define SET_CHANGED_FIELD(vars,i,val) (((vars)->changed)?(vars)->changed->flags[i]=(val):0)
#define RESET_CHANGED_FIELD(v,i) SET_CHANGED_FIELD(v,i,VAR_UNCHANGED)

#define LEFT_BRACE	'{'
#define RIGHT_BRACE '}'
#define REFERENCE_OP	'$'
#define REFERENCE_OP2	'{'

#define SET_INTERNAL	"set internal"
#define SHOW_INTERNAL	"show internal"

#ifdef VARS_DEBUG
#	define POP_TRACE(depth)	fprintf( stderr, "popto(%s,%d)=%d\n",\
		CX_Trace_StackTop->func, depth-1, pop_trace_stackto(depth-1))
#	define PUSH_TRACE(fun,newdepth)	fprintf( stderr, "push(%s)=%d\n",\
		fun, newdepth=push_trace_stack(__FILE__,CX_Time(),fun,__LINE__))
#else
#	define POP_TRACE(depth)	pop_trace_stackto(depth-1)
#	define PUSH_TRACE(fun,newdepth)	newdepth=push_trace_stack(__FILE__,CX_Time(),fun,__LINE__)
#endif

/* the following read an array of values
 * from s in  array of length *elems. Format: a,b,c,..
 * return EOF if less than *elems elements in s.
 * *elems is updated to reflect the number of items read.
 */

extern int local_buf_size;
#define ASCANF_FUNCTION_BUF	local_buf_size
#ifdef __GNUC__
#else
/* #	define ASCANF_FUNCTION_BUF	1024	*/
#endif

extern int Ascanf_Max_Args;
/* Defined while compiling an expression:	*/
extern int ascanf_SyntaxCheck;
#define ASCANF_MAX_ARGS		Ascanf_Max_Args
#define AMAXARGS			-1

#ifndef MIN
#	define MIN(a,b)                (((a)<(b))?(a):(b))
#endif
#define Function_args(f)	((f)?(((f)->Nargs>=0)? MIN((f)->Nargs,ASCANF_MAX_ARGS) : ASCANF_MAX_ARGS):0)

typedef enum ascanf_Function_type { NOT_EOF=0 , NOT_EOF_OR_RETURN,
	_ascanf_value, _ascanf_function, _ascanf_variable, _ascanf_array, _ascanf_procedure,
	_ascanf_simplestats, _ascanf_simpleanglestats, _ascanf_python_object,
	_ascanf_novariable, _ascanf_types
} ascanf_Function_type;

typedef ascanf_Function_type ascanf_type;
extern char *ascanf_type_name[_ascanf_types];


#ifdef ASCANF_ALTERNATE

typedef struct ascanf_Callback_Frame{
	double *args;
	double *result;
	int *level;
	struct Compiled_Form *compiled;
	struct ascanf_Function *self;
#if defined(ASCANF_ARG_DEBUG)
	  /* Optional fields must come at the end to avoid problems!	*/
	char *expr;
#	define ASCB_FRAME_EXPRESSION	1
#endif
} ascanf_Callback_Frame;

#ifdef ASCB_FRAME_EXPRESSION
#	define AH_EXPR	__ascb_frame->expr
#else
#	define AH_EXPR	NULL
#endif

#	if defined(ASCANF_ARG_DEBUG) || defined(DEBUG)
  /* A routine that will return frame->expr (if non-null), and print it on StdErr. The stub argument
   \ is only there to receive a pointer to the local variable (theExpr below) in which the expression is
   \ to be stored. This prevents gcc from complaining about theExpr being unused (it still is of course...
   \ but gcc is too braindead to understand that :))
   */
extern char *_callback_expr(ascanf_Callback_Frame *frame, char *fn, int lnr, char **stub);
#		define ASCB_FRAME_RESULT	double *result= __ascb_frame->result;\
	char *theExpr=(ascanf_verbose>1)? _callback_expr(__ascb_frame,__FILE__, __LINE__,&theExpr) : AH_EXPR;
#		define ASCB_FRAME_SHORT	double *args= __ascb_frame->args, *result= __ascb_frame->result;\
	char *theExpr=(ascanf_verbose>1)? _callback_expr(__ascb_frame,__FILE__, __LINE__,&theExpr) : AH_EXPR;
#		define ASCB_FRAME	double *args= __ascb_frame->args, *result= __ascb_frame->result;\
	int *level= __ascb_frame->level;\
	char *theExpr=(ascanf_verbose>1)? _callback_expr(__ascb_frame,__FILE__, __LINE__,&theExpr) : AH_EXPR;

#	else

#		define ASCB_FRAME_RESULT	double *result= __ascb_frame->result;
#		define ASCB_FRAME_SHORT	double *args= __ascb_frame->args, *result= __ascb_frame->result;
#		define ASCB_FRAME	double *args= __ascb_frame->args, *result= __ascb_frame->result; int *level= __ascb_frame->level;
#	endif

#	define ASCB_ARGLIST	ascanf_Callback_Frame *__ascb_frame
#	define ASCB_ARGUMENTS	__ascb_frame
#	define ASCB_COMPILED	__ascb_frame->compiled
#	define ASCB_LEVEL	__ascb_frame->level

#else

#	define ASCB_FRAME_RESULT	/* ascanf_CallBack_Frame (result only) not used	*/
#	define ASCB_FRAME_SHORT	/* ascanf_Callback_Frame (short form) not used	*/
#	define ASCB_FRAME	/* ascanf_Callback_Frame not used	*/
#	define ASCB_ARGLIST	double *args, double *result, int *level
#	define ASCB_ARGUMENTS	args, result, level
#	define ASCB_COMPILED	NULL
#	define ASCB_LEVEL	level

#endif

#define ASCANF_CALLBACK(name)	int name ( ASCB_ARGLIST )

typedef enum special_ascanf_Functions {
	direct_fun=0,
	ifelse_fun, switch_fun0, switch_fun, dowhile_fun, whiledo_fun, DCL_fun, DEPROC_fun,
	Delete_fun, EditPROC_fun, for_to_fun, for_toMAX_fun, AND_fun, OR_fun,
	matherr_fun, global_fun, verbose_fun, no_verbose_fun, IDict_fun, comment_fun, popup_fun,
	systemtime_fun, systemtime_fun2, SHelp_fun,
	not_a_function
} special_ascanf_Functions;

typedef struct ascanf_Function ascanf_Function;
typedef struct ascanf_NameSpace ascanf_NameSpace;

struct ascanf_NameSpace{
	char *name;
	size_t name_length;
	long hash;
	ascanf_Function *dict;
	ascanf_NameSpace *car, *cdr;
};

struct ascanf_Function{
	char *name;
/* 	int (*function)(double *args,double *result, int *level);	*/
	ASCANF_CALLBACK( (*function) );
	int Nargs;
	ascanf_Function_type type;
	char *usage;
	  /* Whether an _ascanf_variable with name[0]=='$' can be changed:	*/
	int dollar_variable;
	  /* Should this function be called when compiling, to check syntax?	*/
	int SyntaxCheck;
	  /* more or less internal fields:	*/
	int name_length;
	long hash;
	  /* Fields for _ascanf_variable's: check vars_ascanf_Functions[] initialisations if ever
	   \ something changes down here!
	   */
	int store, sign;
	double value, *array;
	int *iarray;
	struct Compiled_Form *procedure;
	ascanf_Function *local_scope, *accessHandler;
/* 20010329: I blindly put the *label field here, which of course caused initialisation problems with N etc...	*/
/* 	char *label;	*/
	int assigns, reads, N, last_index, links,
		  /* Take the address of this one, or is this the address of another one (someday..)	*/
		take_address, is_address, is_usage, take_usage,
		  /* Negate ("booleanise") this one's value:	*/
		negate,
		  /* marks an internal variable:	*/
		internal, user_internal;
	short
		  /* Whether this is a procedure that will post a dialog to allow pre-execution modification its code	*/
		dialog_procedure, procedure_separator;
	  /* Old value, used only when accesHandler is initialised:	*/
	double old_value,
		  /* accesHandler parameters:	*/
		aH_par[3];
	int aH_flags[2];
	FILE *fp;
	int fp_is_pipe;
	char *fp_mode, *label;
	struct CustomFont *cfont;
	struct SimpleStats *SS;
	struct SimpleAngleStats *SAS;
	void *PyObject;
	struct {
		void *self;
		ascanf_Function **selfaf;
	} PyAOself;
	char *PyObject_Name;
	ascanf_Function *PyObject_ReturnVar;
	int PythonHasReturnVar, PythonEvalType;
	special_ascanf_Functions special_fun;
#ifdef XG_DYMOD_SUPPORT
	struct DyModLists *dymod;
#endif
	  /* 20031005: Currently only used for _ascanf_array (re)allocation, when set. */
	void* (*malloc)(size_t size);
	void (*free)(void *memory);
#ifdef ASCANF_64BIT_COMPAT
	  /* 20031016 */
	unsigned int sID;
#endif
	double own_address;
	  /* 20050404: sourceArray field moved here. It had messed up the field order used in static initialisations... */
	ascanf_Function *sourceArray;
	  /* 20080709: for direct linking to DataSet data: */
	struct{
		double *dataColumn;
		short set_nr, col_nr;
	} linkedArray;
	  /* 20051105: */
	ascanf_NameSpace *nameSpace;
	  /* next function	*/
	ascanf_Function *car, *cdr;
};

#ifdef SAFE_FOR_64BIT
	typedef uint32_t address32;
	extern void register_ascanf_Address( ascanf_Function *af, address32 repr );
#else
	typedef void* address32;
	extern void register_ascanf_Address( ascanf_Function *af );
#endif

extern int Delete_Variable( ascanf_Function *af );
extern int Delete_Internal_Variable( char *name, ascanf_Function *entry );
extern ascanf_Function *verify_ascanf_Address( address32 p, ascanf_Function_type type );
extern void delete_ascanf_Address( address32 af );
extern int register_VariableNames( int yesno );
extern void register_VariableName( ascanf_Function *af );
extern ascanf_Function *get_VariableWithName( char *name, int exhaustive );
extern void delete_VariableName( char *name );
// 20100610: brute-force method using the algorithm also used by the compiler/parser:
extern ascanf_Function *find_ascanf_function( char *name, double *result, int *ok, char *caller );
extern void register_DoubleWithIndex( double value, long idx );
extern long get_IndexForDouble( double value );
extern void delete_IndexForDouble( double value );
extern int register_LinkedArray_in_List( void **lst, ascanf_Function *af );
extern int unregister_LinkedArray( ascanf_Function *af );
extern int remove_LinkedArray_from_List( void **lst, ascanf_Function *af );
extern ascanf_Function *walk_LinkedArray_List( void **lst, void **iterator );
extern FILE *register_FILEsDescriptor( FILE *fp );
extern FILE *get_FILEForDescriptor( int fd );
extern void delete_FILEsDescriptor( FILE *fp );

extern double take_ascanf_address( ascanf_Function *af );
extern pragma_malloc ascanf_Function *parse_ascanf_address( double a, int this_type, char *caller, int verbose, int *take_usage );
extern double AccessHandler(ascanf_Function *af, char *caller, int *level, struct Compiled_Form *form, char *expr, double *result);

extern ascanf_Function *Procedure_From_Code( void *code );

// calling a callback "easily" from C:
#ifndef _ASCANFC_C
/* 	CAUTION: the arguments have to be doubles, even constants (the compiler won't know what to promote to...)	*/
	extern double ASCB_call( ASCANF_CALLBACK( (*function) ), int *success, int level, char *expr, int max_argc, int argc, ... );
#endif

extern ascanf_Function *af_ArgList, *ascanf_XGOutput;
extern double ascanf_PointerPos[2], ascanf_ArgList[1];
extern int ascanf_update_ArgList;
extern int ascanf_arg_error;

extern char d3str_format[16];
extern int ascanf_use_greek_inf;
extern char *ad2str( double val, const char *format, char **Buf );

extern int ascanf_verbose;

extern int Print_Form( FILE *fp, struct Compiled_Form **Form, int print_results, int pp, char *whdr, char *wpref, char *wrapstr, int ff );

#if !defined(strdup) && !defined(__cplusplus)
	extern char *strdup();
#endif

#ifndef CLIP_EXPR_CAST
/* A safe casting/clipping macro.	*/
#	define CLIP_EXPR_CAST(ttype,var,stype,expr,low,high)	{stype clip_expr_cast_lvalue=(expr); if(clip_expr_cast_lvalue<(low)){\
		(var)=(ttype)(low);\
	}else if(clip_expr_cast_lvalue>(high)){\
		(var)=(ttype)(high);\
	}else{\
		(var)=(ttype)clip_expr_cast_lvalue;}}
#endif

  /* ASCANF_TRUE(): a macro that returns the Boolean value of a double (an ascanf variable). Any
   \ double that is not 0 and not a NaN is True, all others are False.
   */
#define ASCANF_TRUE(dbl_x)	(((dbl_x) && !isNaN((dbl_x)))? True : False)
#define ASCANF_FALSE(dbl_x)	!ASCANF_TRUE(dbl_x)
extern int ASCANF_TRUE_(double x);

#define ASCANF_ARG_TRUE(i)	((ascanf_arguments>(i)) && ASCANF_TRUE(args[(i)]))

extern int Resize_ascanf_Array_force;
extern void* (*ascanf_array_malloc)(size_t size);
extern void (*ascanf_array_free)(void *memory);
  /* When a dymod containing the free and/or malloc routines for ascanf_arrays is unloaded, remaining arrays
   \ need to be converted back to regular arrays. ascanf_Arrays2Regular() does that.
   */
extern int ascanf_Arrays2Regular(void *malloc, void *free);

#ifdef DEBUG
#	ifdef __GNUC__
	__attribute__((used))
#	endif
	static double ASCANF_ARRAY_ELEM_SET( ascanf_Function *af, int i, double v )
	{
		if( af->iarray ){
			af->iarray[i] = (int) v;
		}
		else{
			af->array[i] = v;
		}
		return(v);
	}

#	ifdef __GNUC__
	__attribute__((used))
#	endif
	static double ASCANF_IARRAY_ELEM( ascanf_Function *af, int i )
	{
		return( af->iarray[i] );
	}
#	ifdef __GNUC__
	__attribute__((used))
#	endif
	static double ASCANF_DARRAY_ELEM( ascanf_Function *af, int i )
	{
		return( af->array[i] );
	}
#	define ASCANF_ARRAY_ELEM(af,i)	(((af)->iarray)? ASCANF_IARRAY_ELEM(af,i) : ASCANF_DARRAY_ELEM(af,i))

#else

#	if defined(__GNUC__) && (__GNUC__<3)
#		define ASCANF_ARRAY_ELEM_SET(af,i,v)	(((af)->iarray)? (af)->iarray[i] : (af)->array[i])=(v)
#	else
#		define ASCANF_ARRAY_ELEM_SET(af,i,v)	if((af)->iarray){ (af)->iarray[i]=(v); } else { (af)->array[i]=(v); }
#	endif
#	define ASCANF_ARRAY_ELEM(af,i)	(((af)->iarray)? (af)->iarray[i] : (af)->array[i])
#endif
#define ASCANF_ARRAY_ELEM_OP(af,i,op,v)	if((af)->iarray){ (af)->iarray[i] op (v); } else { (af)->array[i] op (v); }

#ifdef __cplusplus
}
#endif

#endif
