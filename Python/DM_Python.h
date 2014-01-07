#ifndef _PYTHON_H

#ifdef __PYTHON_MODULE_SRC__

	extern DyMod_Interface *Python_DM_Base;
#	define DMBase	Python_DM_Base

		extern ascanf_Function* (*Create_Internal_ascanfString_ptr)( char *string, int *level );
		extern int (*show_ascanf_functions_ptr)( FILE *fp, char *prefix, int do_bold, int lines );
// 		extern int (*new_param_now_ptr)( char *ExprBuf, double *val, int N);
		extern int (*ascanf_call_method_ptr)( ascanf_Function *af, int argc, double *args, double *result, int *retval, ascanf_Callback_Frame *__ascb_frame, int alloc_largs );
		extern double (*ascanf_WaitForEvent_h_ptr)( int type, char *message, char *caller );
		extern ascanf_Function* (*find_ascanf_function_ptr)( char *name, double *result, int *ok, char *caller );
		extern int (*register_VariableNames_ptr)( int );
// 		extern ascanf_Function* (*get_VariableWithName_ptr)( char *name );
		extern void (*register_VariableName_ptr)( ascanf_Function *af );
		extern int (*Delete_Variable_ptr)( ascanf_Function *af );
		extern int (*Delete_Internal_Variable_ptr)( char *name, ascanf_Function *entry );
		extern void (*realloc_Xsegments_ptr)();
		extern void (*realloc_points_ptr)( DataSet *this_set, int allocSize, int force );
		extern double** (*realloc_columns_ptr)( DataSet *this_set, int ncols );
		extern void (*Check_Columns_ptr)(DataSet *this_set);
		extern void (*_ascanf_RedrawNow_ptr)(int unsilence, int all, int update);
		extern int (*Handle_An_Events_ptr)( int level, int CheckFirst, char *caller, Window win, long mask);
		extern char* (*AscanfTypeName_ptr)( int type );
		extern char* (*ULabel_pixelCName_ptr)(UserLabel*, int* );
		extern char* (*ColumnLabelsString_ptr)( DataSet *set, int column, char *newstr, int new, int nCI, int *ColumnInclude );
		extern int (*LinkSet2_ptr)( DataSet *this_set, int set_link );
		extern void (*grl_HandleEvents_ptr)();
		extern ascanf_Function *(*Create_Internal_ascanfString_ptr)( char *string, int *level );
		extern ascanf_Function *(*Create_ascanfString_ptr)( char *string, int *level );
		extern double (*DBG_SHelp_ptr)( char *string, int internal );

		extern int  *ascanf_check_int_ptr;
		extern char ***Argv_ptr;
		extern char **TBARprogress_header_ptr, **TBARprogress_header2_ptr;
		extern int *maxitems_ptr;
		extern int *AlwaysUpdateAutoArrays_ptr;
		extern int *ascanf_AutoVarWouldCreate_msg_ptr;
		extern char **ULabelTypeNames_ptr;
		extern int  *ascanf_Functions_ptr;
		extern ascanf_Function *vars_ascanf_Functions_ptr;
		extern unsigned int *grl_HandlingEvents_ptr;

#		define Create_Internal_ascanfString	(*Create_Internal_ascanfString_ptr)
#		define show_ascanf_functions			(*show_ascanf_functions_ptr)
// #		define new_param_now				(*new_param_now_ptr)
#		define ascanf_call_method			(*ascanf_call_method_ptr)
#		define ascanf_WaitForEvent_h			(*ascanf_WaitForEvent_h_ptr)
#		define find_ascanf_function			(*find_ascanf_function_ptr)
#		define register_VariableNames			(*register_VariableNames_ptr)
// #		define get_VariableWithName			(*get_VariableWithName_ptr)
#		define register_VariableName			(*register_VariableName_ptr)
#		define Delete_Variable				(*Delete_Variable_ptr)
#		define Delete_Internal_Variable		(*Delete_Internal_Variable_ptr)
#		define realloc_Xsegments				(*realloc_Xsegments_ptr)
#		define realloc_points				(*realloc_points_ptr)
#		define realloc_columns				(*realloc_columns_ptr)
#		define Check_Columns				(*Check_Columns_ptr)
#		define _ascanf_RedrawNow				(*_ascanf_RedrawNow_ptr)
#		define Handle_An_Events				(*Handle_An_Events_ptr)
#		define AscanfTypeName				(*AscanfTypeName_ptr)
#		define ULabel_pixelCName				(*ULabel_pixelCName_ptr)
#		define ColumnLabelsString			(*ColumnLabelsString_ptr)
#		define LinkSet2					(*LinkSet2_ptr)
#		define grl_HandleEvents				(*grl_HandleEvents_ptr)
#		define Create_Internal_ascanfString	(*Create_Internal_ascanfString_ptr)
#		define Create_ascanfString			(*Create_ascanfString_ptr)
#		define DBG_SHelp					(*DBG_SHelp_ptr)

#		define Argv						(*Argv_ptr)
#		define ascanf_check_int				(*ascanf_check_int_ptr)
#		define TBARprogress_header			(*TBARprogress_header_ptr)
#		define TBARprogress_header2			(*TBARprogress_header2_ptr)
#		define maxitems					(*maxitems_ptr)
#		define AlwaysUpdateAutoArrays			(*AlwaysUpdateAutoArrays_ptr)
#		define ascanf_AutoVarWouldCreate_msg	(*ascanf_AutoVarWouldCreate_msg_ptr)
#		define ULabelTypeNames				(ULabelTypeNames_ptr)
#		define ascanf_Functions				(*ascanf_Functions_ptr)
#		define vars_ascanf_Functions			(vars_ascanf_Functions_ptr)
#		define grl_HandlingEvents			(*grl_HandlingEvents_ptr)

#	include "Python/PyObjects.h"

	extern PyObject *AscanfPythonModule;
	extern PyObject *XGraphPythonModule;
	extern PyObject *XG_PythonError;

	extern ascanf_Function *Py_getNamedAscanfVariable( char *name );
	extern PyObject *Py_ImportVariableFromAscanf( ascanf_Function **af, char **name, int Ndims, npy_intp *dim, int deref, int force_address );
	extern PyObject *Py_ExportVariableToAscanf( PyAscanfObject *pao, char* name, PyObject *var,
										int force, int IDict, int as_pobj, char *label, int retVar );

	extern char *PyObject_Name( PyObject *var );
	extern ascanf_Function *make_ascanf_python_object( ascanf_Function *af, PyObject *var, char *caller );

	// 20140107: function declaration so that clang/C99 will allow the function to be accessible from outside Python.c even
	// if it is defined inline.
	extern int python_check_interrupted();
#	define CHECK_INTERRUPTED()	if( python_check_interrupted() ){ return(NULL); }

#else /* !__PYTHON_MODULE_SRC__ */

#endif /* __PYTHON_MODULE_SRC__ */


#include "Python/PythonInterface.h"

#define _PYTHON_H
#endif

