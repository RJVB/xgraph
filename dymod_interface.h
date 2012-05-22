#ifndef _DYMOD_INTERFACE_H

#include "pragmas.h"

#include "xtb/xtb.h"

#ifndef ELAPSED_H
struct Time_Struct;
#endif

typedef struct DyMod_Interface{
	void *XGraphHandle;
	size_t sizeof_DyMod_Interface, sizeof_ascanf_Function, sizeof_Compiled_Form, sizeof_LocalWin, sizeof_DataSet, sizeof_Python_Interface;
	DyModLists **p_DyModList;
	FILE **p_StdErr, **p_NullDevice;
	struct LocalWin **p_ActiveWin, **p_StubWindow_ptr;
	char *p_ascanf_separator, **p_ascanf_emsg;
	int *p_ascanf_arg_error, *p_ascanf_arguments, *p_ascanf_escape, *p_ascanf_exit, *p_ascanf_interrupt,
		*p_ascanf_SyntaxCheck, *p_ascanf_verbose, *p_Unloaded_Used_Modules,
		*p_debugFlag, *p_debugLevel,
		*p_MaxSets, *p_setNumber;
	double *p_ascanf_progn_return, **p_ascanf_setNumber;
	void **p_ascanf_array_malloc;
	void **p_ascanf_array_free;
	int *p_Resize_ascanf_Array_force;
	struct DataSet **p_AllSets;
	unsigned long *p_ascanf_window;
	struct ascanf_Function **p_af_ArgList;
	double *p_af_ArgList_address;
	int *p_ascanf_update_ArgList;
	int *p_Ascanf_Max_Args;
	int *p_SwapEndian, *p_EndianType;
	int *p_scriptVerbose;
	char **p_PrefsDir;

	void **p_disp;
	int *p_RemoteConnection;

	int *p_StartUp;

	struct DM_Python_Interface **p_dm_python;


	  /* "constants" (pointers in the main programme) */
	char *p_d3str_format;
	struct SimpleStats *p_EmptySimpleStats;
	struct SimpleAngleStats *p_EmptySimpleAngleStats;

	  /* Functions/routines, in no particular order */
	int (*p_xtb_error_box)( unsigned long parent, char *mesg, char *title);
	int (*p_ascanf_CheckFunction)( struct ascanf_Function *af );
	int (*p_add_ascanf_functions)( struct ascanf_Function *array, int n, char *caller);
	int (*p_add_ascanf_functions_with_autoload)( ascanf_Function *array, int n, char* libname, char *caller);
	int (*p_remove_ascanf_functions)( struct ascanf_Function *array, int n, int force);
	int (*p_Copy_preExisting_Variable_and_Delete)( struct ascanf_Function *af, char *label );
	char* (*p_XGstrdup)(const char *string);
	int (*p_XGstrcmp)(const char *a, const char *b);
	int (*p_XGstrncmp)(const char *a, const char *b, size_t n);
	int (*p_XGstrcasecmp)(const char *a, const char *b);
	int (*p_XGstrncasecmp)(const char *a, const char *b, size_t n);
	int (*p_Check_Doubles_Ascanf)( struct ascanf_Function *af, char *label, int warn );
	ascanf_Function* (*p_parse_ascanf_address)( double a, int this_type, char *caller, int verbose, int *take_usage );
	double (*p_take_ascanf_address)( struct ascanf_Function *af );
	ascanf_Function* (*p_get_VariableWithName)( char *name, int exhaustive );
	double (*p_AccessHandler)( struct ascanf_Function *af, char *caller, int *level, struct Compiled_Form *form, char *expr, double *result );
	int (*p_Auto_LoadDyMod)( DyModAutoLoadTables *Table, int N, char *fname );
	int (*p_Auto_LoadDyMod_LastPtr)( DyModAutoLoadTables *Table, int N, char *fname, DyModLists **last );
	char* (*p_d2str)(double, const char*, char*);
	char* (*p_ad2str)(double, const char*, char**);
	int (*p_fascanf2)( int *n, char *s, double *a, int separator);
	int (*p_ascanf_Variable)( ASCB_ARGLIST );
	int (*p_Resize_ascanf_Array)( struct ascanf_Function *af, int N, double *result );
	char* (*p_tildeExpand)(char *out, const char *in);
	int (*p_IncludeFile)( struct LocalWin *rwi, FILE *strm, char *Fname, int event, char *skip_to );
	void (*p__xfree)(void *x, char *file, int lineno );
	char* (*p__callback_expr)( struct ascanf_Callback_Frame *__ascb_frame, char *fn, int lnr, char **stub );
	void* (*p__XGrealloc)( void* ptr, size_t n, char *name, char *size);
	void* (*p_xgalloca)(unsigned int n, char *file, int linenr);
	char* (*p_GetEnv)( char *n);
	char* (*p_SetEnv)( char *n, char *v);
	int (*p_ascanf_check_event)(char *caller);
	int (*p_evaluate_procedure)( int *n, ascanf_Function *proc, double *args, int *level );
	int (*p_q_Permute)( void *a, int n, int size, int quick);
	double (*p_Elapsed_Since)( struct Time_Struct *then, int update );
	double (*p_Elapsed_Since_HR)( struct Time_Struct *then, int update );
	int (*p__DiscardedPoint)( struct LocalWin *wi, struct DataSet *set, int idx);
	int16_t* (*p_SwapEndian_int16)( int16_t *data, int N);
	int* (*p_SwapEndian_int)( int *data, int N);
	int32_t* (*p_SwapEndian_int32)( int32_t *data, int N);
	float* (*p_SwapEndian_float)( float *data, int N);
	double* (*p_SwapEndian_double)( double *data, int N);
	int (*p_StringCheck)( char *text, int len, char *file, int linenr);
	struct XGStringList* (*p_XGStringList_Delete)( struct XGStringList *list );
	struct XGStringList* (*p_XGStringList_AddItem)( struct XGStringList *list, char *text );
	struct XGStringList* (*p_XGStringList_FindItem)( struct XGStringList *list, char *text, int *item );

	double (*p_SS_Mean)( struct SimpleStats *SS);
	double (*p_SS_St_Dev)( struct SimpleStats *SS);
	char* (*p_SS_sprint_full)( char *buffer, char *format, char *sep, double min_err, struct SimpleStats *a);
	struct SimpleStats* (*p_SS_Add_Data)(struct SimpleStats *a, long count, double sum, double weight);
	struct SimpleAngleStats* (*p_SAS_Add_Data)(struct SimpleAngleStats *a, long count, double sum, double weight, int convert);
	double (*p_SAS_Mean)(struct SimpleAngleStats *SS);
	double (*p_SAS_St_Dev)(struct SimpleAngleStats *SS);
	char* (*p_SAS_sprint_full)( char *buffer, char *format, char *sep, double min_err, struct SimpleAngleStats *a);
	struct Sinc* (*p_Sinc_string_behaviour)( struct Sinc *sinc, char *string, long cnt, long base, enum SincString behaviour );
	int (*p_Sflush)( struct Sinc *sinc );
	int (*p_Sputs)( char *text, struct Sinc *sinc );
	char* (*p_concat)(char *first, ...);
	char* (*p_concat2)(char *first, ...);

	struct LocalWin * (*p_aWindow)( struct LocalWin *w );
	char * (*p_parse_codes)( char *T );
	char * (*p_ParseTitlestringOpcodes)( struct LocalWin *wi, int idx, char *title, char **parsed_end );
	char * (*p_strrstr)(const char *a, const char *b);
	char * (*p_xg_re_comp)(const char *pat);
	int (*p_xg_re_exec)(const char *lp);
	char * (*p_xtb_input_dialog)( Window parent, char *text, int tilen, int maxlen, char *mesg, char *title,
		int modal,
		char *hlp_label, xtb_hret (*p_hlp_btn)(Window,int,xtb_data),
		char *hlp_label2, xtb_hret (*p_hlp_btn2)(Window,int,xtb_data),
		char *hlp_label3, xtb_hret (*p_hlp_btn3)(Window,int,xtb_data)
	);
	int (*p_RedrawNow)( struct LocalWin *wi );
	int (*p_RedrawSet)( int set_nr, Boolean doit );
	int (*p_new_param_now)( char *ExprBuf, double *val, int N);
	// 20081203:
	int (*p_Ascanf_AllocMem)( int elems );

	void ***p_gnu_rl_event_hook;
	int *p_Num_Windows;
} DyMod_Interface;

#ifdef XG_DYMOD_IMPORT_MAIN

static pragma_unused const char *ident= "@(#) " __FILE__ ": code and definitions for dynamically loading xgraph's symbols from inside a DyMod";

#ifdef XG_DYMOD_IMPORT_MAIN_STATIC
#	define StAtIc	static
#else
#	define StAtIc	/* */
#endif


/* extern DyMod_Interface *DMBase;	*/

#ifndef _DYMOD_C

/* Variables that can change */
#define DyModList	(*(DMBase->p_DyModList))
#define StdErr	(*(DMBase->p_StdErr))
#define NullDevice	(*(DMBase->p_NullDevice))
#define ActiveWin	(*(DMBase->p_ActiveWin))
#define StubWindow_ptr	(*(DMBase->p_StubWindow_ptr))
#define debugFlag	(*(DMBase->p_debugFlag))
#define debugLevel	(*(DMBase->p_debugLevel))
#define ascanf_separator	(*(DMBase->p_ascanf_separator))
#define ascanf_emsg	(*(DMBase->p_ascanf_emsg))
#define ascanf_escape	(*(DMBase->p_ascanf_escape))
#define ascanf_exit		(*(DMBase->p_ascanf_exit))
#define ascanf_interrupt	(*(DMBase->p_ascanf_interrupt))
#define ascanf_arg_error	(*(DMBase->p_ascanf_arg_error))
#define ascanf_arguments	(*(DMBase->p_ascanf_arguments))
#define ascanf_SyntaxCheck (*(DMBase->p_ascanf_SyntaxCheck))
#define ascanf_verbose	(*(DMBase->p_ascanf_verbose))
#define ascanf_setNumber	(*(DMBase->p_ascanf_setNumber))
#define ascanf_progn_return (*(DMBase->p_ascanf_progn_return))
#define Unloaded_Used_Modules	(*(DMBase->p_Unloaded_Used_Modules))
#define ascanf_array_malloc	(*(DMBase->p_ascanf_array_malloc))
#define ascanf_array_free	(*(DMBase->p_ascanf_array_free))
#define Resize_ascanf_Array_force	(*(DMBase->p_Resize_ascanf_Array_force))
#define MaxSets	(*(DMBase->p_MaxSets))
#define setNumber	(*(DMBase->p_setNumber))
#define AllSets	(*(DMBase->p_AllSets))
#define ascanf_window	(*(DMBase->p_ascanf_window))
#define af_ArgList	(*(DMBase->p_af_ArgList))
#define af_ArgList_address	(*(DMBase->p_af_ArgList_address))
#define ascanf_update_ArgList	(*(DMBase->p_ascanf_update_ArgList))
#define Ascanf_Max_Args	(*(DMBase->p_Ascanf_Max_Args))
#define SwapEndian	(*(DMBase->p_SwapEndian))
#define EndianType	(*(DMBase->p_EndianType))
#define scriptVerbose	(*(DMBase->p_scriptVerbose))
#define PrefsDir	(*(DMBase->p_PrefsDir))

#define disp		((Display*)*(DMBase->p_disp))
#define RemoteConnection	(*(DMBase->p_RemoteConnection))

#define StartUp		(*(DMBase->p_StartUp))
#define dm_python		(*(DMBase->p_dm_python))

/* Constants: */
#define d3str_format	(DMBase->p_d3str_format)
#define EmptySimpleStats	(*(DMBase->p_EmptySimpleStats))
#define EmptySimpleAngleStats	(*(DMBase->p_EmptySimpleAngleStats))

/* Functions (constants too...) */
#define xtb_error_box	(DMBase->p_xtb_error_box)
#define ascanf_CheckFunction	(DMBase->p_ascanf_CheckFunction)
#define add_ascanf_functions	(DMBase->p_add_ascanf_functions)
#define add_ascanf_functions_with_autoload	(DMBase->p_add_ascanf_functions_with_autoload)
#define remove_ascanf_functions	(DMBase->p_remove_ascanf_functions)
#define Copy_preExisting_Variable_and_Delete	(DMBase->p_Copy_preExisting_Variable_and_Delete)
#define XGstrdup	(DMBase->p_XGstrdup)
#define XGstrcmp	(DMBase->p_XGstrcmp)
#define XGstrncmp	(DMBase->p_XGstrncmp)
#define XGstrcasecmp	(DMBase->p_XGstrcasecmp)
#define XGstrncasecmp	(DMBase->p_XGstrncasecmp)
#define Check_Doubles_Ascanf	(DMBase->p_Check_Doubles_Ascanf)
#define parse_ascanf_address	(DMBase->p_parse_ascanf_address)
#define take_ascanf_address	(DMBase->p_take_ascanf_address)
#define get_VariableWithName	(DMBase->p_get_VariableWithName)
#define AccessHandler	(DMBase->p_AccessHandler)
#define Auto_LoadDyMod	(DMBase->p_Auto_LoadDyMod)
#define Auto_LoadDyMod_LastPtr	(DMBase->p_Auto_LoadDyMod_LastPtr)
#define d2str	(DMBase->p_d2str)
#define ad2str	(DMBase->p_ad2str)
#define fascanf2	(DMBase->p_fascanf2)
#define ascanf_Variable	(DMBase->p_ascanf_Variable)
#define tildeExpand	(DMBase->p_tildeExpand)
#define IncludeFile	(DMBase->p_IncludeFile)
#define _xfree	(DMBase->p__xfree)
#define _callback_expr	(DMBase->p__callback_expr)
#define _XGrealloc	(DMBase->p__XGrealloc)
#define Resize_ascanf_Array	(DMBase->p_Resize_ascanf_Array)
#define xgalloca	(DMBase->p_xgalloca)
#define GetEnv	(DMBase->p_GetEnv)
#define SetEnv	(DMBase->p_SetEnv)
#define ascanf_check_event (DMBase->p_ascanf_check_event)
#define evaluate_procedure (DMBase->p_evaluate_procedure)
#define q_Permute (DMBase->p_q_Permute)
#define Elapsed_Since	(DMBase->p_Elapsed_Since)
#define Elapsed_Since_HR	(DMBase->p_Elapsed_Since_HR)
#define _DiscardedPoint	(DMBase->p__DiscardedPoint)
#define SwapEndian_int16	(DMBase->p_SwapEndian_int16)
#define SwapEndian_int	(DMBase->p_SwapEndian_int)
#define SwapEndian_int32	(DMBase->p_SwapEndian_int32)
#define SwapEndian_float	(DMBase->p_SwapEndian_float)
#define SwapEndian_double	(DMBase->p_SwapEndian_double)
#define StringCheck	(DMBase->p_StringCheck)
#define XGStringList_AddItem	(DMBase->p_XGStringList_AddItem)
#define XGStringList_Delete	(DMBase->p_XGStringList_Delete)
#define XGStringList_FindItem	(DMBase->p_XGStringList_FindItem)
#define SS_Mean	(DMBase->p_SS_Mean)
#define SS_Mean	(DMBase->p_SS_Mean)
#define SS_St_Dev	(DMBase->p_SS_St_Dev)
#define SS_sprint_full	(DMBase->p_SS_sprint_full)
#define SS_Add_Data	(DMBase->p_SS_Add_Data)
#define SAS_Add_Data	(DMBase->p_SAS_Add_Data)
#define SAS_Mean	(DMBase->p_SAS_Mean)
#define SAS_St_Dev	(DMBase->p_SAS_St_Dev)
#define SAS_sprint_full	(DMBase->p_SAS_sprint_full)
#define SS_St_Dev	(DMBase->p_SS_St_Dev)
#define SS_sprint_full	(DMBase->p_SS_sprint_full)
#define SS_Add_Data	(DMBase->p_SS_Add_Data)
#define SAS_Add_Data	(DMBase->p_SAS_Add_Data)
#define SAS_Mean	(DMBase->p_SAS_Mean)
#define SAS_St_Dev	(DMBase->p_SAS_St_Dev)
#define SAS_sprint_full	(DMBase->p_SAS_sprint_full)
#define Sinc_string_behaviour	(DMBase->p_Sinc_string_behaviour)
#define Sflush	(DMBase->p_Sflush)
#define Sputs	(DMBase->p_Sputs)
#define concat2	(DMBase->p_concat2)
#define concat	(DMBase->p_concat)

#define	aWindow	(DMBase->p_aWindow)
#define	parse_codes	(DMBase->p_parse_codes)
#define	ParseTitlestringOpcodes	(DMBase->p_ParseTitlestringOpcodes)
#define	strrstr	(DMBase->p_strrstr)
#define	xg_re_comp	(DMBase->p_xg_re_comp)
#define	xg_re_exec	(DMBase->p_xg_re_exec)
#define	xtb_input_dialog	(DMBase->p_xtb_input_dialog)
#define	RedrawNow		(DMBase->p_RedrawNow)
#define	RedrawSet		(DMBase->p_RedrawSet)
#define	new_param_now		(DMBase->p_new_param_now)
#define	Ascanf_AllocMem	(DMBase->p_Ascanf_AllocMem)
#define	gnu_rl_event_hook	(*(DMBase->p_gnu_rl_event_hook))
#define	Num_Windows		(*(DMBase->p_Num_Windows))

static pragma_unused int DyMod_API_Check( DyMod_Interface *base )
{ int ok= 0;
  void *prog;
	if( base->XGraphHandle== (prog= dlopen( NULL, RTLD_GLOBAL|RTLD_LAZY )) ){
		if( base->sizeof_DyMod_Interface== sizeof(DyMod_Interface)
#ifdef _ASCANF_H
			&& base->sizeof_ascanf_Function== sizeof(ascanf_Function)
#endif
#ifdef _COMPILED_ASCANF_H
			&& base->sizeof_Compiled_Form== sizeof(Compiled_Form)
#endif
#ifdef _XGRAPH_H
			&& base->sizeof_LocalWin== sizeof(LocalWin)
#endif
#ifdef _DATA_SET_H
			&& base->sizeof_DataSet== sizeof(DataSet)
#endif
#ifdef _PYTHONINTERFACE_H
			&& base->sizeof_Python_Interface== sizeof(DM_Python_Interface)
#endif
		){
			ok= 1;
		}
	}
	return(ok);
}

static pragma_unused int DyMod_API_Check2( DyMod_Interface *base )
{ int ok= 0;
  void *prog;
	if( base->XGraphHandle== (prog= dlopen( NULL, RTLD_GLOBAL|RTLD_LAZY )) ){
		  // allow for expansion of the DyMod_Interface structure, which is safe as long as it's
		  // done by appending
		if( base->sizeof_DyMod_Interface>= sizeof(DyMod_Interface)
#ifdef _ASCANF_H
			&& base->sizeof_ascanf_Function== sizeof(ascanf_Function)
#endif
#ifdef _COMPILED_ASCANF_H
			&& base->sizeof_Compiled_Form== sizeof(Compiled_Form)
#endif
#ifdef _XGRAPH_H
			&& base->sizeof_LocalWin== sizeof(LocalWin)
#endif
#ifdef _DATA_SET_H
			&& base->sizeof_DataSet== sizeof(DataSet)
#endif
#ifdef _PYTHONINTERFACE_H
			&& base->sizeof_Python_Interface== sizeof(DM_Python_Interface)
#endif
		){
			ok= 1;
		}
	}
	return(ok);
}


#endif

#else

#	define MODULE_INTERFACE_VARIABLES	/* */
#	define __XGRAPH_FUNCTION(ptr,name)	/* */
#	define __XGRAPH_VARIABLE(ptr,name)	/* */
#	define XGRAPH_ATTACH()	/* */

	extern double ascanf_progn_return;

#endif

static pragma_unused char *xg_lm_errmsg;

#	define XGRAPH_FUNCTION(ptr,name)	if(DMBase && DMBase->XGraphHandle){ void **p=(void**)&(ptr); \
		*p = dlsym(DMBase->XGraphHandle, name);\
	} \
	if( (xg_lm_errmsg= (char*)dlerror()) || !DMBase->XGraphHandle || !ptr ){ \
		fprintf( StdErr, "%s: Error retrieving xgraph::%s: %s (result=0x%lx)\n", \
			__FILE__, name, xg_lm_errmsg, ptr \
		); \
		return(DM_Error); \
	} \
	else if( debugFlag ){ \
		fprintf( StdErr, "%s: loaded '%s' from main programme (0x%lx).\n",  \
			__FILE__, name, ptr \
		); \
	}

#define XGRAPH_VARIABLE(ptr,name)	XGRAPH_FUNCTION(ptr, name)

#	define DYMOD_FUNCTION(dm,ptr,sname)	if((dm) && (dm)->handle) ptr= dlsym((dm)->handle, sname);\
	if( (xg_lm_errmsg= (char*)dlerror()) || !(dm)->handle || !ptr ){ \
		fprintf( StdErr, "%s: Error retrieving %s::%s: %s (result=0x%lx)\n", \
			__FILE__, (dm)->name, sname, xg_lm_errmsg, ptr \
		); \
		return(DM_Error); \
	} \
	else if( debugFlag ){ \
		fprintf( StdErr, "%s: loaded '%s' from %s (0x%lx).\n",  \
			__FILE__, sname, (dm)->name, ptr \
		); \
	}

#define DYMOD_VARIABLE(dm,ptr,name)	DYMOD_FUNCTION(dm,ptr,name)

extern struct DyMod_Interface *Init_DyMod_Interface( struct DyMod_Interface *base );

#ifdef DYMOD_MAIN
#	define wrong_dymod_loaded(funcname,caller,dymodname)	__wrong_dymod_loaded__(__FILE__,XG_IDENTIFY(),funcname,caller,dymodname)

static void __wrong_dymod_loaded__( char *fname, char *xgid, char *funcname, char *caller, char *dymodname )
{
#if defined(__MACH__) || defined(__APPLE_CC__)
  const char *pname= "(failed)", *mname= "(failed)";
  void *maddr= dlsym(NULL, "main");
  Dl_info minfo, info;
  extern DyModTypes initDyMod( INIT_DYMOD_ARGUMENTS );

	if( dladdr( initDyMod, &minfo ) ){
		mname= minfo.dli_fname;
	}
	if( (maddr= dlsym(RTLD_DEFAULT, "main"))
		|| (maddr= dlsym(RTLD_DEFAULT, "Py_Main"))
		|| (maddr= dlsym(RTLD_DEFAULT, "NSApplicationMain"))
	){
		if( dladdr( maddr, &info) ){
			pname= info.dli_fname;
		}
	}
#endif
	fprintf( stderr,
		"%s::%s: it seems your %s process is loading the wrong %s shared library.\n"
		" Under Mac OS 10.4 (and higher?) this probably means it was found somewhere in your custom $DYLD_LIBRARY_PATH\n"
		" rather than in the location it is actually being searched for (even if an absolute path to it was given).\n"
		" This message comes from an XGraph \"dymod\", \"%s\".\n"
#if defined(__MACH__) || defined(__APPLE_CC__)
		" Guessed calling process name: %s, found shared module: %s\n"
#endif
		, fname, funcname, caller, dymodname
		, xgid
#if defined(__MACH__) || defined(__APPLE_CC__)
		, pname
		, mname
#endif
	);
}

#endif

#define _DYMOD_INTERFACE_H
#endif
