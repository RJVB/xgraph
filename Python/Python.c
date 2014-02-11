#include "config.h"
IDENTIFY( "ascanf library module for interfacing with Python" );

#ifndef XG_DYMOD_SUPPORT
#error You need to define XG_DYMOD_SUPPORT in config.h - and compile xgraph with it!!!
#endif


#ifdef linux
#	define __USE_GNU
#endif

#include "Python/Python_headers.h"
#include "Python/Py_InitModule.h"

#ifdef __APPLE__
#	ifdef RJVB
#		ifdef PYTHON25
#			include <Python2.5/eval.h>
#			include <Python2.5/ceval.h>
#		elif PYTHON24
#			include <Python2.4/eval.h>
#			include <Python2.4/ceval.h>
#		elif PYTHON26
#			include <Python2.6/eval.h>
#			include <Python2.6/ceval.h>
#		elif PYTHONsys
#			include <Python/compile.h>
#			include <Python/eval.h>
#			include <Python/ceval.h>
#		else
#			include <Python/compile.h>
#			include <Python/eval.h>
#			include <Python/ceval.h>
#		endif
#	else
#		ifdef PYTHON23
#			include <Python/compile.h>
#			include <Python/eval.h>
#			include <Python/ceval.h>
#		else
#			include <Python/eval.h>
#			include <Python/ceval.h>
#		endif
#	endif
#	define __ACCELERATE__
#elif defined(__CYGWIN__) || defined(linux)
#	ifdef PYTHON25
#		include <python2.5/eval.h>
#		include <python2.5/ceval.h>
#	elif PYTHON26
#		include <python2.6/eval.h>
#		include <python2.6/ceval.h>
#	endif
#else
#	include <eval.h>
#	include <ceval.h>
#endif

#if !defined(Py_RETURN_NONE)	// Python 2.3?!
#	define Py_RETURN_NONE return Py_INCREF(Py_None), Py_None
#	define Py_EnterRecursiveCall(s)	0
#	define Py_LeaveRecursiveCall()	/**/
#endif //PYTHON23

#if PY_MAJOR_VERSION >= 3 && PY_MINOR_VERSION > 2
#	define _PyUnicode_AsDefaultEncodedString(unistring,dum)	PyUnicode_AsUTF8String((unistring))
#endif
// PyUnicode_AsUTF8String

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <signal.h>
#include <libgen.h>
#include <locale.h>

  /* Get the dynamic module definitions:	*/
#include "dymod.h"

extern FILE *StdErr;

#include "copyright.h"

  /* Include a whole bunch of headerfiles. Not all of them are strictly necessary, but if
   \ we want to have fdecl.h to know all functions we possibly might want to call, this
   \ list is needed.
   */
#include "xgout.h"
#include "xgraph.h"
#include "new_ps.h"
#include "xtb/xtb.h"

#include "NaN.h"

#include "fdecl.h"

  /* get the ascanf definitions:	*/
#include "ascanf.h"
  /* If we want to be able to access the "expression" field in the callback argument, we need compiled_ascanf.h .
   \ If we don't include it, we will just get 'theExpr= NULL' ... (this applies to when -DDEBUG)
   */
#include "compiled_ascanf.h"
#include "ascanfc-table.h"

  /* For us to be able to access the calling programme's internal variables, the calling programme should have
   \ had at least 1 object file compiled with the -rdynamic flag (gcc 2.95.2, linux). Under irix 6.3, this
   \ is the default for the compiler and/or the OS (gcc).
   \ On some other systems, XG_DYMOD_IMPORT_MAIN should be defined (see config.h).
   */

// include PythonInterface.h here, so that dymod_interface.h can activate the size-checking code:
#include "Python/PythonInterface.h"

#define DYMOD_MAIN
#include "dymod_interface.h"
// static DyMod_Interface DMBaseMem;
DyMod_Interface *Python_DM_Base= NULL;

static int initialised= False, RVN, pythonActive= 0;

static ascanf_Function *ascanf_VarLabel = NULL;

#define __PYTHON_MODULE_SRC__
#include "Python/DM_Python.h"

	ascanf_Function* (*Create_Internal_ascanfString_ptr)( char *string, int *level );
	int (*show_ascanf_functions_ptr)( FILE *fp, char *prefix, int do_bold, int lines );
// 	int (*new_param_now_ptr)( char *ExprBuf, double *val, int N);
	int (*ascanf_call_method_ptr)( ascanf_Function *af, int argc, double *args, double *result, int *retval, ascanf_Callback_Frame *__ascb_frame, int alloc_largs );
	double (*ascanf_WaitForEvent_h_ptr)( int type, char *message, char *caller );
	ascanf_Function* (*find_ascanf_function_ptr)( char *name, double *result, int *ok, char *caller );
	int (*register_VariableNames_ptr)( int );
// 	ascanf_Function* (*get_VariableWithName_ptr)( char *name, int exhaustive );
	void (*register_VariableName_ptr)( ascanf_Function *af );
	int (*Delete_Variable_ptr)( ascanf_Function *af );
	int (*Delete_Internal_Variable_ptr)( char *name, ascanf_Function *entry );
	void (*realloc_Xsegments_ptr)();
	void (*realloc_points_ptr)( DataSet *this_set, int allocSize, int force );
	double** (*realloc_columns_ptr)( DataSet *this_set, int ncols );
	void (*Check_Columns_ptr)(DataSet *this_set);
	void (*_ascanf_RedrawNow_ptr)(int unsilence, int all, int update);
	int (*Handle_An_Events_ptr)( int level, int CheckFirst, char *caller, Window win, long mask);
	char* (*AscanfTypeName_ptr)( int type );
	char* (*ULabel_pixelCName_ptr)(UserLabel*, int* );
	char* (*ColumnLabelsString_ptr)( DataSet *set, int column, char *newstr, int new, int nCI, int *ColumnInclude );
	int (*LinkSet2_ptr)( DataSet *this_set, int set_link );
	void (*grl_HandleEvents_ptr)();
	ascanf_Function *(*Create_Internal_ascanfString_ptr)( char *string, int *level );
	ascanf_Function *(*Create_ascanfString_ptr)( char *string, int *level );
	DyModLists *(*LoadDyMod_ptr)( char *Name, int flags, int no_dump, int auto_unload );
	int (*UnloadDyMod_ptr)( char *Name, int *load_count, int force );
	char *(*ascanf_index_ptr)( char *s, char c, int *instring );
	DyModAutoLoadTables *(*Add_LoadDyMod_ptr)( DyModAutoLoadTables *target, int *target_len, DyModAutoLoadTables *source, int n );
	int (*IOImport_Data_ptr)( const char *ioModuleName, char *filename );
	FILE* (*get_FILEForDescriptor_ptr)( int fd );
	double (*DBG_SHelp_ptr)( char *string, int internal );

	int  *ascanf_check_int_ptr;
	char ***Argv_ptr;
	char **TBARprogress_header_ptr, **TBARprogress_header2_ptr;
	int *maxitems_ptr;
	int *AlwaysUpdateAutoArrays_ptr;
	int *ascanf_AutoVarWouldCreate_msg_ptr;
	char **ULabelTypeNames_ptr;
	int  *ascanf_Functions_ptr;
	ascanf_Function *vars_ascanf_Functions_ptr;
	unsigned int *grl_HandlingEvents_ptr;
	DyModAutoLoadTables *AutoLoadTable_ptr;
	int *AutoLoads_ptr, *DyModsLoaded_ptr;

#	define LoadDyMod					(*LoadDyMod_ptr)
#	define UnloadDyMod					(*UnloadDyMod_ptr)
#	define ascanf_index					(*ascanf_index_ptr)
#	define Add_LoadDyMod				(*Add_LoadDyMod_ptr)
#	define IOImport_Data				(*IOImport_Data_ptr)
#	define get_FILEForDescriptor			(*get_FILEForDescriptor_ptr)
#	define AutoLoadTable				(AutoLoadTable_ptr)
#	define AutoLoads					(*AutoLoads_ptr)
#	define DyModsLoaded					(*DyModsLoaded_ptr)


PyObject *AscanfPythonModule= NULL;
PyObject *AscanfPythonDictionary= NULL;
PyObject *XGraphPythonModule= NULL;
PyObject *XGraphPythonDictionary= NULL;
PyObject *PenPythonModule= NULL;

#ifdef IS_PY3K
	static struct PyModuleDef AscanfModuleDef, XGraphModuleDef, PenModuleDef;
#endif

PyObject *ReturnValue= NULL;

PyObject *MainModule= NULL;
PyObject *MainDictionary= NULL;
PyObject *SysModule= NULL;
PyObject *SysDictionary= NULL;

DM_Python_Interface _DM_Python_Interface_;

// reference value for the readline idle handler at the time Python-Shell was called. Its base value
// is -1 to signal that we're not actually in interactive Python code.
int grl_HandlingEvents_ref = -1;

int inInteractiveShell();
#ifdef __GNUC__
inline
#endif
int inInteractiveShell()
{
	return ( grl_HandlingEvents_ref>=0 && grl_HandlingEvents != grl_HandlingEvents_ref );
}

int Check_External_Availability()
{
	return !inInteractiveShell();
}

#if PY_MAJOR_VERSION >= 3 && PY_MINOR_VERSION < 2
//----------------- copied verbatim from the Python 3.2.3 distro
#ifdef MS_WINDOWS
#  include <windows.h>
#endif

/* Decode a byte string from the locale encoding with the
   surrogateescape error handler (undecodable bytes are decoded as characters
   in range U+DC80..U+DCFF). If a byte sequence can be decoded as a surrogate
   character, escape the bytes using the surrogateescape error handler instead
   of decoding them.

   Use _Py_wchar2char() to encode the character string back to a byte string.

   Return a pointer to a newly allocated wide character string (use
   PyMem_Free() to free the memory) and write the number of written wide
   characters excluding the null character into *size if size is not NULL, or
   NULL on error (conversion or memory allocation error).

   Conversion errors should never happen, unless there is a bug in the C
   library. */
wchar_t*
_Py_char2wchar(const char* arg, size_t *size)
{
    wchar_t *res;
    size_t argsize = mbstowcs(NULL, arg, 0);
    size_t count;
    unsigned char *in;
    wchar_t *out;
    mbstate_t mbs;
    if (argsize != (size_t)-1) {
        res = (wchar_t *)PyMem_Malloc((argsize+1)*sizeof(wchar_t));
        if (!res)
            goto oom;
        count = mbstowcs(res, arg, argsize+1);
        if (count != (size_t)-1) {
            wchar_t *tmp;
            /* Only use the result if it contains no
               surrogate characters. */
            for (tmp = res; *tmp != 0 &&
                         (*tmp < 0xd800 || *tmp > 0xdfff); tmp++)
                ;
            if (*tmp == 0) {
                if (size != NULL)
                    *size = count;
                return res;
            }
        }
        PyMem_Free(res);
    }
    /* Conversion failed. Fall back to escaping with surrogateescape. */
    /* Try conversion with mbrtwoc (C99), and escape non-decodable bytes. */

    /* Overallocate; as multi-byte characters are in the argument, the
       actual output could use less memory. */
    argsize = strlen(arg) + 1;
    res = (wchar_t*)PyMem_Malloc(argsize*sizeof(wchar_t));
    if (!res)
        goto oom;
    in = (unsigned char*)arg;
    out = res;
    memset(&mbs, 0, sizeof mbs);
    while (argsize) {
        size_t converted = mbrtowc(out, (char*)in, argsize, &mbs);
        if (converted == 0)
            /* Reached end of string; null char stored. */
            break;
        if (converted == (size_t)-2) {
            /* Incomplete character. This should never happen,
               since we provide everything that we have -
               unless there is a bug in the C library, or I
               misunderstood how mbrtowc works. */
            fprintf(stderr, "unexpected mbrtowc result -2\n");
            PyMem_Free(res);
            return NULL;
        }
        if (converted == (size_t)-1) {
            /* Conversion error. Escape as UTF-8b, and start over
               in the initial shift state. */
            *out++ = 0xdc00 + *in++;
            argsize--;
            memset(&mbs, 0, sizeof mbs);
            continue;
        }
        if (*out >= 0xd800 && *out <= 0xdfff) {
            /* Surrogate character.  Escape the original
               byte sequence with surrogateescape. */
            argsize -= converted;
            while (converted--)
                *out++ = 0xdc00 + *in++;
            continue;
        }
        /* successfully converted some bytes */
        in += converted;
        argsize -= converted;
        out++;
    }
    if (size != NULL)
        *size = out - res;
    return res;
oom:
    fprintf(stderr, "out of memory\n");
    return NULL;
}

/* Encode a (wide) character string to the locale encoding with the
   surrogateescape error handler (characters in range U+DC80..U+DCFF are
   converted to bytes 0x80..0xFF).

   This function is the reverse of _Py_char2wchar().

   Return a pointer to a newly allocated byte string (use PyMem_Free() to free
   the memory), or NULL on conversion or memory allocation error.

   If error_pos is not NULL: *error_pos is the index of the invalid character
   on conversion error, or (size_t)-1 otherwise. */
char*
_Py_wchar2char(const wchar_t *text, size_t *error_pos)
{
    const size_t len = wcslen(text);
    char *result = NULL, *bytes = NULL;
    size_t i, size, converted;
    wchar_t c, buf[2];

    if (error_pos != NULL)
        *error_pos = (size_t)-1;

    /* The function works in two steps:
       1. compute the length of the output buffer in bytes (size)
       2. outputs the bytes */
    size = 0;
    buf[1] = 0;
    while (1) {
        for (i=0; i < len; i++) {
            c = text[i];
            if (c >= 0xdc80 && c <= 0xdcff) {
                /* UTF-8b surrogate */
                if (bytes != NULL) {
                    *bytes++ = c - 0xdc00;
                    size--;
                }
                else
                    size++;
                continue;
            }
            else {
                buf[0] = c;
                if (bytes != NULL)
                    converted = wcstombs(bytes, buf, size);
                else
                    converted = wcstombs(NULL, buf, 0);
                if (converted == (size_t)-1) {
                    if (result != NULL)
                        PyMem_Free(result);
                    if (error_pos != NULL)
                        *error_pos = i;
                    return NULL;
                }
                if (bytes != NULL) {
                    bytes += converted;
                    size -= converted;
                }
                else
                    size += converted;
            }
        }
        if (result != NULL) {
            *bytes = 0;
            break;
        }

        size += 1; /* nul byte at the end */
        result = PyMem_Malloc(size);
        if (result == NULL)
            return NULL;
        bytes = result;
    }
    return result;
}
//---------------------------------------------------------------
#endif	// python 3.0 or 3.1

/* Attention: this function is very expensive, it slows down execution of a typical statement almost 20x!! */

void Python_SysArgv0(char *argv0, char *argv1)
{ int argc= (argv1)? 2 : 1;
#ifdef IS_PY3K
  wchar_t *argv[3];
	argv[0]= _Py_char2wchar( (argv0)? argv0 : "xgraph", NULL );
	argv[1]= (argv1)? _Py_char2wchar( argv1, NULL ) : NULL;
	argv[2]= NULL;
#else
  char *argv[3];

	argv[0]= (argv0)? argv0 : "xgraph";
	argv[1]= (argv1)? argv1 : NULL;
	argv[2]= NULL;
#endif
#if PY_MAJOR_VERSION <= 2 && PY_MINOR_VERSION < 7
	PySys_SetArgv( argc, argv );
#else
	PySys_SetArgvEx( argc, argv, False );
#endif
}

#ifdef IS_PY3K
	void Py2Sys_SetArgv( int argc, char *argv[] )
	{ wchar_t **wargv;
	  int i;
		if( argc > 0 && (wargv = calloc(argc, sizeof(wchar_t*))) ){
			for( i = 0 ; i < argc ; i++ ){
				if( argv[i] ){
					wargv[i] = _Py_char2wchar( argv[i], NULL );
				}
			}
			PySys_SetArgvEx( argc, wargv, False );
		}
	}
#else
	void Py2Sys_SetArgv( int argc, char *argv[] )
	{
#if PY_MAJOR_VERSION <= 2 && PY_MINOR_VERSION < 7
		PySys_SetArgv( argc, argv );
#else
		PySys_SetArgvEx( argc, argv, False );
#endif
	}
#endif

void Python_INCREF(PyObject *obj)
{
	Py_XINCREF(obj);
}

void Python_DECREF(PyObject *obj)
{
	Py_XDECREF(obj);
}

int Python_CheckSignals()
{
	return( PyErr_CheckSignals() );
}

void Python_SetInterrupt()
{
	PyErr_SetInterrupt();
}

#ifdef IS_PY3K
#	define Python_No_fclosing	0
#else
int Python_No_fclosing( FILE *fp )
{
	if( !(debugFlag || scriptVerbose)
		&& (fp== stdin || fp== stdout || fp== stderr || fp== StdErr)
	){
		return(0);
	}
	if( initialised> 0 ){
		if( (PyErr_Warn( PyExc_Warning, "ignoring attempt to close a file opened elsewhere!" )) ){
			return(1);
		}
		else{
			return(0);
		}
	}
	else{
		fprintf( StdErr, " (ignoring attempt to close a file opened elsewhere!) " );
		return(0);
	}
}
#endif

#ifdef IS_PY3K
	typedef PyObject		XGPyCodeObject;
#else
	typedef PyCodeObject	XGPyCodeObject;
#endif

typedef struct Compiled_Python{
	XGPyCodeObject *cexpr;
	char *code;
	int start;
} Compiled_Python;
static Compiled_Python *Compiled_ExpressionList= NULL;
static size_t Compiled_ExpressionListEntries=0;

int ascanf_PythonCompile ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  ascanf_Function *pexpr;
	ascanf_arg_error= 0;
	set_NaN(*result);
	if( !args || ascanf_arguments< 1 ){
		ascanf_emsg= " (need at least 1 argument)";
		ascanf_arg_error= 1;
	}
	else{
		if( (pexpr= parse_ascanf_address( args[0], 0, "ascanf_PythonEval", (int) ascanf_verbose, NULL ))
			&& pexpr->usage
		){
			if( !ascanf_SyntaxCheck ){
			  int enr= -1, i;
				for( i= 0; i< Compiled_ExpressionListEntries && enr<0 ; i++ ){
					if( !Compiled_ExpressionList[i].cexpr || strcmp(Compiled_ExpressionList[i].code, pexpr->usage)== 0 ){
						enr= i;
					}
				}
				if( enr< 0 || enr>= Compiled_ExpressionListEntries ){
				  int n= Compiled_ExpressionListEntries+ 2;
					if( !(Compiled_ExpressionList=
							(Compiled_Python*) realloc( (void*) Compiled_ExpressionList, n* sizeof(Compiled_Python) ))
					){
						fprintf( StdErr, " (can't [re]allocate 2 more (%d total) compiled Python exprs (%s)) ",
							n, serror()
						);
						ascanf_arg_error= 1;
						Compiled_ExpressionListEntries= 0;
					}
					else{
						for( i= Compiled_ExpressionListEntries; i< n; i++ ){
							memset( &Compiled_ExpressionList[i], 0, sizeof(Compiled_Python) );
						}
						enr= Compiled_ExpressionListEntries;
						Compiled_ExpressionListEntries= n;
					}
				}
				if( !ascanf_arg_error ){
					PyErr_Clear();
					if( strchr( pexpr->usage, '=' ) || strchr(pexpr->usage, ';') ){
						Compiled_ExpressionList[enr].start= Py_file_input;
					}
					else{
						  /* see _Evaluate_Python_Expr(): this is really just a lucky guess we're in the best case! */
						Compiled_ExpressionList[enr].start= Py_eval_input;
					}
#if 0
					Compiled_ExpressionList[enr].cexpr= Py_CompileString( pexpr->usage, "<ascanf>",
						Compiled_ExpressionList[enr].start );
#else
					{ void *n;
					  /* Inspired by _Evaluate_Python_Expr() and the code for PyRun_String(): parse the code string first.
					   \ If this fails (n==NULL), try again with Py_file_input.
					   \ This probably means that the runtime checks below are redundant...
					   */
						if( !(n= PyParser_SimpleParseString(pexpr->usage, Compiled_ExpressionList[enr].start )) ){
							if( ascanf_verbose ){
								fprintf( StdErr, " (PyParser_SimpleParseString with Py_eval_input returned NULL, trying with Py_file_input) ");
								if( ascanf_verbose> 1 && PyErr_Occurred() ){
									PyErr_Print();
								}
							}
							PyErr_Clear();
							Compiled_ExpressionList[enr].start= Py_file_input;
							n= PyParser_SimpleParseString(pexpr->usage, Compiled_ExpressionList[enr].start );
						}
						if( n ){
#if PY_MAJOR_VERSION > 2 || (PY_MAJOR_VERSION == 2 && PY_MINOR_VERSION > 4)
/* #ifdef CHECK should be made against Python version macros. */
							Compiled_ExpressionList[enr].cexpr= (XGPyCodeObject*) PyNode_Compile( n, "<ascanf>" );
#else
							Compiled_ExpressionList[enr].cexpr= (XGPyCodeObject*) PyNode_CompileFlags( n, "<ascanf>", NULL );
#endif
							PyNode_Free(n);
						}
					}
#endif
					if( PyErr_Occurred() ){
						PyErr_Print();
						ascanf_arg_error= 1;
						memset( &Compiled_ExpressionList[enr], 0, sizeof(Compiled_Python) );
					}
					else{
						Compiled_ExpressionList[enr].code= strdup( pexpr->usage );
						*result= enr;
					}
				}
			}
		}
	}
	return( !ascanf_arg_error );
}

int ascanf_PythonDestroyCExpr ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  int i;
	ascanf_arg_error= 0;
	set_NaN(*result);
	if( !args || ascanf_arguments< 1 ){
		ascanf_emsg= " (need at least 1 argument)";
		ascanf_arg_error= 1;
	}
	else{
		for( i= 0; i< ascanf_arguments; i++ ){
			if( args[i]>= 0 && args[i]< Compiled_ExpressionListEntries ){
				Py_XDECREF( Compiled_ExpressionList[ (int) args[i] ].cexpr );
				Compiled_ExpressionList[ (int) args[i] ].cexpr= NULL;
				xfree( Compiled_ExpressionList[ (int) args[i] ].code );
				*result= i;
			}
			else{
				fprintf( StdErr, " (cexpr number out of bounds [0,%d>) ", Compiled_ExpressionListEntries );
			}
		}
	}
	return( !ascanf_arg_error );
}

int ascanf_PythonPrintCExpr ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  int i;
	ascanf_arg_error= 0;
	set_NaN(*result);
	if( !args || ascanf_arguments< 1 ){
		ascanf_emsg= " (need at least 1 argument)";
		ascanf_arg_error= 1;
	}
	else{
		for( i= 0; i< ascanf_arguments; i++ ){
			if( args[i]>= 0 && args[i]< Compiled_ExpressionListEntries ){
				fprintf( StdErr, "## compiled Python expression #%d: %s\n",
					(int) args[i], Compiled_ExpressionList[ (int) args[i] ].code
				);
				*result= i;
			}
			else{
				fprintf( StdErr, " (cexpr number out of bounds [0,%d>) ", Compiled_ExpressionListEntries );
			}
		}
	}
	return( !ascanf_arg_error );
}

int Run_Python_Expr( char *expr )
{ int r, pA= pythonActive;
	if( !Py_EnterRecursiveCall( "Run_Python_Expr():" STRING(__LINE__) ) ){
		pythonActive+= 1;
		r= PyRun_SimpleString(expr);
		pythonActive= pA;
		Py_LeaveRecursiveCall();
	}
	return(r);
}

int Import_Python_File( char *filename, char *sourceFile, int unlink_afterwards, int ispy2k )
{ int ret= 0;
  int pA= pythonActive;
	if( filename[0]== '|' ){
	  char *cname= &filename[1];
	  FILE *fp;
		while( isspace(*cname) && *cname ){
			cname++;
		}
		if( cname ){
			fp= popen(cname, "r");
		}
		if( fp ){
			pythonActive+= 1;
			if( sourceFile ){
				Python_SysArgv0(sourceFile, "");
				ret= PyRun_AnyFile( fp, sourceFile );
			}
			else{
				Python_SysArgv0(cname, "");
				ret= PyRun_AnyFile( fp, cname );
			}
			Python_SysArgv0(NULL, NULL);
			pythonActive= pA;
			pclose(fp);
		}
		else{
			ascanf_emsg= " (impossible command) ";
			fprintf( StdErr, " (can't execute '%s' (%s)) ", &filename[1], serror() );
			ret= 1;
		}
	}
	else{
	  char *fname= NULL;
	  FILE *fp= fopen( (fname= tildeExpand( fname, filename )), "r");
		if( fname ){
		  char *dir = dirname(fname), *bname = basename(fname);
		  char *altFName = concat( dir,
#ifdef IS_PY3K
		  						"/py3k/",
#else
								"/py2k/",
#endif
								bname, NULL
						);
import_py_file:
			if( fp ){
				pythonActive+= 1;
				if( sourceFile ){
#ifdef IS_PY3K
					if( ispy2k ){
					  char *cmd = concat(
							"try:\n"
								"\tfrom lib2to3.main import main as convert2to3\n"
								"\tconvert2to3( 'lib2to3.fixes', ['-w','-n','--no-diffs','", sourceFile, "'] )\n"
							"except:\n"
								"\tprint( 'Warning: failed to convert python2 code to python3!', file=sys.stderr )\n"
							"\n", NULL );
						if( cmd ){
							fclose(fp);
							PyRun_SimpleString(cmd);
							xfree(cmd);
							fp = fopen( sourceFile, "r" );
						}
					}
#endif
					Python_SysArgv0(sourceFile, "");
					ret= PyRun_AnyFile( fp, sourceFile );
				}
				else{
#ifdef IS_PY3K
					if( ispy2k ){
					  char *cmd = concat(
					  	"try:\n"
							"\tprint('convert2to3(\"lib2to3.fixes\", [\"-w\",\"-n\",\"--no-diffs\",\"", fname, "\"])', file=sys.stderr )\n"
							"\tfrom lib2to3.main import main as convert2to3\n"
							"\tconvert2to3( 'lib2to3.fixes', ['-w','-n','--no-diffs','", fname, "'] )\n"
						"except:\n"
							"\tprint( 'Warning: failed to convert python2 code to python3!', file=sys.stderr )\n"
						"\n", NULL );
						if( cmd ){
							fclose(fp);
							PyRun_SimpleString(cmd);
							xfree(cmd);
							fp = fopen( fname, "r" );
						}
					}
#endif
					Python_SysArgv0(fname, "");
					ret= PyRun_AnyFile( fp, fname );
				}
				Python_SysArgv0(NULL, NULL);
				pythonActive= pA;
				if( strcmp( fname, "/dev/tty") ){
					fclose(fp);
					if( unlink_afterwards ){
						unlink(fname);
					}
				}
			}
			else{
				if( !sourceFile && altFName ){
					if( (fp = fopen( altFName, "r" )) ){
						xfree(fname);
						fname = altFName;
						altFName = NULL;
						goto import_py_file;
					}
				}
				ascanf_emsg= " (file doesn't exist) ";
				fprintf( StdErr, " (can't open file '%s' (%s) (%s)) ", filename, fname, serror() );
				ret= 1;
			}
			xfree(fname);
		}
	}
	fflush(StdErr);
	fflush(stdout);
	return( ret );
}

// 20101023: importing a file from outside the Python environment can cause a crash if we're in an interactive
// shell, so we make sure that there's a distinct entry path and do the appropriate checking.
int Import_Python_File_wrapper( char *filename, char *sourceFile, int unlink_afterwards, int ispy2k )
{
	if( inInteractiveShell() ){
		fprintf( StdErr, "## ignoring Import_Python_File(\"%s\",\"%s\") while in a Python-Shell!\n",
			(filename)? filename : "<NULL>",
			(sourceFile)? sourceFile : "<NULL>"
		);
		return 0;
	}
	return Import_Python_File( filename, sourceFile, unlink_afterwards, ispy2k );
}

int ascanf_PythonEval ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  ascanf_Function *pexpr;
  int i;
	ascanf_arg_error= 0;
	set_NaN(*result);
	if( !args || ascanf_arguments< 1 ){
		ascanf_emsg= " (need at least 1 argument)";
		ascanf_arg_error= 1;
	}
	else{
		for( i= 0; i< ascanf_arguments; i++ ){
			if( (pexpr= parse_ascanf_address( args[i], 0, "ascanf_PythonEval", (int) ascanf_verbose, NULL ))
				&& pexpr->usage
			){
				if( !ascanf_SyntaxCheck ){
					*result= Run_Python_Expr( pexpr->usage );
				}
			}
		}
	}
	return( !ascanf_arg_error );
}

static int init_gtkcons()
{ static char called= 0, failure= 0;
  int pA= pythonActive;
	if( !called ){
		pythonActive+= 1;
		PyRun_SimpleString(
			"try:\n"
				"\timport os, pygtk\n"
				"\tpygtk.require('1.2')\n"
				"\ttry:\n"
					"\t\timport gtkcons\n"
					"\t\txgraph.gtkconsole=lambda: gtkcons.gtk_console(globals())\n"
				"\texcept:\n"
//	 				"\t\tos.write(sys.stderr.fileno(), 'Cannot load gtkcons\\n')\n"
#ifdef IS_PY3K
					"\t\tprint('Cannot load gtkcons', file=sys.stderr)\n"
#else
					"\t\tprint >>sys.stderr, 'Cannot load gtkcons'\n"
#endif
			"except:\n"
// 				"\tos.write(sys.stderr.fileno(), 'Cannot load pygtk\\n')\n"
#ifdef IS_PY3K
				"\tprint('Cannot load pygtk', file=sys.stderr)\n"
#else
				"\tprint >>sys.stderr, 'Cannot load pygtk'\n"
#endif
		);
		pythonActive= pA;
		called= 1;
		failure= 0;
	}
	return(1);
}

static int init_tkcons()
{ static char called= 0, failure= 0;
  int pA= pythonActive;
	if( !called ){
		pythonActive+= 1;
		PyRun_SimpleString(
			"try:\n"
				"\tfrom TkConsoleX11 import TkConsole\n"
				"\tdef __idle_cons__(cons):\n"
					"\t\txgraph.__idle_input_handler__()\n"
					"\t\tif cons.idle_callback:\n"
						"\t\t\tcons.install_idle_callback(cons.idle_callback)\n"
				"\txgraph.tkconsole=lambda: TkConsole(globals(),__idle_cons__,50)\n"
			"except:\n"
				"\ttry:\n"
					"\t\tfrom TkConsole import TkConsole\n"
					"\t\tdef __idle_cons__(cons):\n"
						"\t\t\txgraph.__idle_input_handler__()\n"
						"\t\t\tif cons.idle_callback:\n"
							"\t\t\t\tcons.install_idle_callback(cons.idle_callback)\n"
					"\t\txgraph.tkconsole=lambda: TkConsole(globals(),__idle_cons__,50)\n"
				"\texcept:\n"
					"\t\tdel xgraph.tkconsole\n"
// 					"\t\tos.write(sys.stderr.fileno(), 'Cannot load TkConsole\\n')\n"
#ifdef IS_PY3K
					"\t\tprint('Cannot load TkConsole', file=sys.stderr)\n"
#else
					"\t\tprint >>sys.stderr, 'Cannot load TkConsole'\n"
#endif
		);
		pythonActive= pA;
		called= 1;
		failure= 0;
	}
	return(1);
}

static int init_interact()
{ static char called= 0, failure= 0;
  int pA= pythonActive;
	if( isatty(fileno(stdin)) && isatty(fileno(stdout)) ){
		if( !called ){
			pythonActive+= 1;
			PyRun_SimpleString( "import code" );
			PyRun_SimpleString( "xgraph.interact=code.interact" );
// 			PyRun_SimpleString( "xgraph.interact.__doc__='Call the code.interact embedded shell'" );
			pythonActive= pA;
			called= 1;
			failure= 0;
		}
		return(1);
	}
	else{
		fprintf( StdErr, "Not on an interactive terminal, not starting any embedded interactive shells!\n" );
		if( !failure ){
			pythonActive+= 1;
			PyRun_SimpleString( "import os" );
#ifdef IS_PY3K
			PyRun_SimpleString( "xgraph.interact=lambda: print('Not on an interactive terminal',file=sys.stderr)" );
#else
			PyRun_SimpleString( "xgraph.interact=lambda: os.write(sys.stderr.fileno(), 'Not on an interactive terminal\\n')" );
#endif
			pythonActive= pA;
			failure= 1;
			called= 0;
		}
		return(0);
	}
}

static int init_ipshell()
{ static char called= 0, failure= 0;
  int pA= pythonActive;
	if( isatty(fileno(stdin)) && isatty(fileno(stdout)) ){
		if( !called ){
			pythonActive+= 1;
			PyRun_SimpleString( "try:\n"
								// attempt to load with the newer API
								"\timport IPython\n"
								"\ttry:\n"
								"\t\txgraph.ipshell=IPython.terminal.embed.InteractiveShellEmbed()\n"
								"\texcept:\n"
								"\t\txgraph.ipshell=IPython.frontend.terminal.embed.InteractiveShellEmbed()\n"
								"\txgraph.ipshell.banner='Entering xgraph.ipshell IPython shell'\n"
								"\txgraph.ipshell.exit_msg='Leaving xgraph.ipshell IPython shell'\n"
							"except:\n"
								// fall back onto the API that existed until IPython 0.10
								"\tfrom IPython.Shell import IPythonShellEmbed\n"
								"\txgraph.ipshell=IPythonShellEmbed()\n"
								"\txgraph.ipshell.set_banner('Entering xgraph.ipshell IPython shell')\n"
								"\txgraph.ipshell.set_exit_msg('Leaving xgraph.ipshell IPython shell')\n"
							"\n"
			);
			pythonActive= pA;
			called= 1;
			failure= 0;
		}
		return(1);
	}
	else{
		fprintf( StdErr, "Not on an interactive terminal, not starting any embedded IPython shells!\n" );
		if( !failure ){
			pythonActive+= 1;
			PyRun_SimpleString( "import os" );
#ifdef IS_PY3K
			PyRun_SimpleString( "xgraph.ipshell=lambda: print('Not on an interactive terminal',file=sys.stderr)" );
#else
			PyRun_SimpleString( "xgraph.ipshell=lambda: os.write(sys.stderr.fileno(), 'Not on an interactive terminal\\n')" );
#endif
			PyRun_SimpleString( "xgraph.ipshell.restore_system_completer=lambda:None" );
			pythonActive= pA;
			failure= 1;
			called= 0;
		}
		return(0);
	}
}

int in_IPShell= 0;
static void (*current_PyOS_InputHook)() = NULL;

static int Python_grl_HandleEvents()
{ static int active = 0;

	if( !active ){
		active = 1;
		(*grl_HandleEvents)();
		if( current_PyOS_InputHook && (void*) current_PyOS_InputHook != (void*) Python_grl_HandleEvents
		   && (void*) current_PyOS_InputHook != (void*) grl_HandleEvents
		){
			(*current_PyOS_InputHook)();
		}
		active = 0;
	}
	return 0;
}

/* 20090317: store and restore rl_event_hook in this function? */
int open_PythonShell( double *arg, int *result )
{ int Result;
  void *prev;
  int pA= pythonActive, gHEr= grl_HandlingEvents_ref, ret = 0;
  void *reh;
	if( !result ){
		result = &Result;
	}
	if( gnu_rl_event_hook ){
		reh= *gnu_rl_event_hook;
	}
	grl_HandlingEvents_ref = grl_HandlingEvents;
	if( arg && arg[0]< 0 ){
// 		prev= signal( SIGINT, SIG_DFL );
		 // !! don't allow pythonActive to be incremented here, as we have our own flag!
		if( Num_Windows && !StartUp ){
//			PyOS_InputHook= (gnu_rl_event_hook && *gnu_rl_event_hook)? *gnu_rl_event_hook : grl_HandleEvents;
			// 20110131: let's try to support currently installed hooks... grl_HandleEvents will now call
			// gnu_rl_event_hook if it's a valid handler (and not 'self').
			current_PyOS_InputHook = (void*) PyOS_InputHook;
			PyOS_InputHook = Python_grl_HandleEvents;
		}
		if( !Py_EnterRecursiveCall( "ascanf_PythonShell():" STRING(__LINE__) ) ){
			PyRun_SimpleString( "import xgraph, ascanf" );
			pythonActive-= 1;
			*result= Run_Python_Expr( "import code\nxgraph.interact=code.interact\nxgraph.interact()" );
			ret = 1;
			pythonActive= pA;
			Py_LeaveRecursiveCall();
		}
		if( gnu_rl_event_hook && Num_Windows && !StartUp ){
			*gnu_rl_event_hook= reh;
		}
// 		signal( SIGINT, prev);
	}
	else{
 		if( !(isatty(fileno(stdin)) && isatty(fileno(stdout))) ){
			if( !Py_EnterRecursiveCall( "ascanf_PythonShell():" STRING(__LINE__) ) ){
				PyRun_SimpleString( "import xgraph, ascanf" );
				if( init_tkcons() || init_gtkcons() ){
					if( Num_Windows && !StartUp ){
//						PyOS_InputHook= (gnu_rl_event_hook && *gnu_rl_event_hook)? *gnu_rl_event_hook : grl_HandleEvents;
						current_PyOS_InputHook = (void*) PyOS_InputHook;
						PyOS_InputHook = Python_grl_HandleEvents;
					}
					 // !! don't allow pythonActive to be incremented here, as we have our own flag!
					pythonActive-= 1;
					*result= Run_Python_Expr(
						"try:\n"
							"\txgraph.tkconsole()\n"
						"except:\n"
							"\txgraph.gtkconsole()"
					);
					ret = 1;
					pythonActive= pA;
					if( gnu_rl_event_hook && Num_Windows && !StartUp ){
						*gnu_rl_event_hook= reh;
					}
				}
				Py_LeaveRecursiveCall();
			}
		}
		else if( !Py_EnterRecursiveCall( "ascanf_PythonShell():" STRING(__LINE__) ) ){
			PyRun_SimpleString( "import xgraph, ascanf" );
			if( init_ipshell() ){
				if( !arg || ASCANF_TRUE(arg[0]) ){
				  int iIS= in_IPShell;
					  /* IPython doesn't install its own SIGINT handler once signal(SIGINT,...) has been called, even if
					   \ the handler has been (re)set to the default, SIG_DFL. Unloading ourselves while IPython is active
					   \ will result in a crash, so the only option we seem to have is to ignore SIGINT while in the interactive
					   \ shell.
					   */
					prev= signal( SIGINT, SIG_IGN );
					in_IPShell+= 1;
					if( Num_Windows && !StartUp ){
//						PyOS_InputHook= (gnu_rl_event_hook && *gnu_rl_event_hook)? *gnu_rl_event_hook : grl_HandleEvents;
						current_PyOS_InputHook = (void*) PyOS_InputHook;
						PyOS_InputHook = Python_grl_HandleEvents;
					}
					 // !! don't allow pythonActive to be incremented here, as we have our own flag!
					pythonActive-= 1;
					*result= Run_Python_Expr( "xgraph.ipshell()\n"
										// attempt the cleanup code which existed until 0.10
										"try:\n"
											"\txgraph.ipshell.restore_system_completer()\n"
										"except:\n"
											"\tpass\n"
										"\n"
					);
					ret = 1;
					pythonActive= pA;
					if( gnu_rl_event_hook && Num_Windows && !StartUp ){
						*gnu_rl_event_hook= reh;
					}
					in_IPShell= iIS;
					  /* Restore the previous SIGINT handler (typically ExitProgramme()) */
					signal( SIGINT, prev);
				}
			}
			Py_LeaveRecursiveCall();
		}
	}
	grl_HandlingEvents_ref = gHEr;
	return( 1 );
}

int ascanf_PythonShell ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  int res;
	ascanf_arg_error= 0;
	if( open_PythonShell( args, &res ) ){
		*result = (double) res;
	}
	else{
		set_NaN(*result);
	}
	return !ascanf_arg_error;
}

int ascanf_PythonEvalCExpr ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  int i;
	ascanf_arg_error= 0;
	set_NaN(*result);
	if( !args || ascanf_arguments< 1 ){
		ascanf_emsg= " (need at least 1 argument)";
		ascanf_arg_error= 1;
	}
	else{
		for( i= 0; i< ascanf_arguments; i++ ){
		  XGPyCodeObject *cexpr;
		  int enr;
			enr= (int) args[i];
			if( enr >= 0 && enr < Compiled_ExpressionListEntries
				&& (cexpr= Compiled_ExpressionList[enr].cexpr)
			){
			  PyObject *res;
			  int pA= pythonActive;
				if( ascanf_verbose ){
					fprintf( StdErr, "## %s\n",
						Compiled_ExpressionList[ enr ].code
					);
				}
				PyErr_Clear();
				pythonActive+= 1;
				res= PyEval_EvalCode( cexpr, MainDictionary, MainDictionary );
				if( !res && Compiled_ExpressionList[enr].start== Py_eval_input ){
					PyErr_Clear();
					Compiled_ExpressionList[enr].start= Py_file_input;
					Py_XDECREF( Compiled_ExpressionList[enr].cexpr );
					if( (Compiled_ExpressionList[enr].cexpr= (XGPyCodeObject*) Py_CompileString( Compiled_ExpressionList[enr].code, "<ascanf>",
						Compiled_ExpressionList[enr].start ))
					){
						PyErr_Clear();
						res= PyEval_EvalCode( Compiled_ExpressionList[enr].cexpr, MainDictionary, MainDictionary );
					}
					else{
						ascanf_arg_error= 1;
						memset( &Compiled_ExpressionList[enr], 0, sizeof(Compiled_Python) );
					}
				}
				pythonActive= pA;
				if( PyErr_Occurred() ){
					PyErr_Print();
					PyErr_Clear();
					*result= 0;
				}
				else{
					*result= 1;
				}
				Py_XDECREF(res);
			}
			else{
				fprintf( StdErr, " (illegal cexpr %s) ", ad2str(args[i], d3str_format, NULL) );
			}
		}
	}
	return( !ascanf_arg_error );
}

int Get_Python_ReturnValue(double *result)
{
	PyErr_Clear();
	if( (ReturnValue= PyMapping_GetItemString( AscanfPythonDictionary, "ReturnValue" )) ){
		*result= PyFloat_AsDouble(ReturnValue);
		Py_XDECREF(ReturnValue);
		return(1);
	}
	else{
		if( PyErr_Occurred() ){
			PyErr_Print();
		}
		return(0);
	}
}

PyObject *_Evaluate_Python_Expr( char *expr, int *evalType )
{ PyObject *ret= NULL;
  int pA= pythonActive;

	PyErr_Clear();
	pythonActive+= 1;
	if( *evalType!= 0 ){
		ret= PyRun_String(expr, *evalType, MainDictionary, MainDictionary );
	}
	else
	{
		if( strchr(expr, '=') || strchr(expr, ';') ){
			ret= PyRun_String(expr, Py_file_input, MainDictionary, MainDictionary );
			*evalType= Py_file_input;
		}
		else{
			if( !(ret= PyRun_String(expr, Py_eval_input, MainDictionary, MainDictionary )) ){
#if 0
				if( ascanf_verbose ){
					fprintf( StdErr, " (PyRun_String with Py_eval_input returned NULL, trying with Py_file_input) ");
				}
#endif
				PyErr_Clear();
				ret= PyRun_String(expr, Py_file_input, MainDictionary, MainDictionary );
				*evalType= Py_file_input;
			}
			else{
				*evalType= Py_eval_input;
			}
		}
	}
	pythonActive= pA;
	if( PyErr_Occurred() ){
		PyErr_Print();
	}
	return(ret);
}

static
#ifdef __GNUC__
inline
#endif
void _python_expression_value( PyObject *retval, double *result, char *exportName, ascanf_Function **af, char *descr )
{
	 // 20100504: numerical values are returned as such, and no attempt is made to return them in an ascanf variable
	if( PyInt_Check(retval) || PyLong_Check(retval) || PyFloat_Check(retval) ){
		*result= PyFloat_AsDouble(retval);
	}
	else{
	  int aspobj= False;
retry_as_pobj:;
		if( ExportVariableToAscanf( NULL, exportName, retval, True, True, aspobj, af ) ){
			if( (*af)->type== _ascanf_variable && !(*af)->is_address && !(*af)->is_usage && !(*af)->take_usage ){
				*result= (*af)->value;
			}
			else{
				*result= (*af)->own_address;
			}
			if( (*af)->usage ){
				if( descr && !(*af)->is_usage && !(*af)->take_usage ){
					xfree((*af)->usage);
					(*af)->usage= XGstrdup(descr);
				}
			}
		}
		  // 20100504: this may be an unsupported retval type. Rather than bailing out, we attempt first to
		  // return it as a PythonObject
		else if( !aspobj ){
			aspobj = True;
			goto retry_as_pobj;
		}
	}
}

int Python_Expression_Value( PyObject *retval, double *result, char *descr, ascanf_Function *exportAf, char *exportName )
{ int ok;

	if( retval ){
		if( result ){
		  ascanf_Function *af= exportAf;
			if( !exportName ){
				exportName = "$Python-Call-Result";
			}
			PyErr_Clear();
			ok= 1;
			if( retval== Py_None ){
				if( ascanf_verbose ){
					fprintf( StdErr, " (expression returned None, returning ascanf.ReturnValue) " );
				}
				if( (ReturnValue= PyMapping_GetItemString( AscanfPythonDictionary, "ReturnValue" )) ){
#if 0
					if( ExportVariableToAscanf( NULL, exportName, ReturnValue, True, True, False, &af ) ){
						if( af->type== _ascanf_variable && !af->is_address && !af->is_usage && !af->take_usage ){
							*result= af->value;
						}
						else{
							*result= af->own_address;
						}
						if( af->usage ){
							if( descr && !af->is_usage && !af->take_usage ){
								xfree(af->usage);
								af->usage= XGstrdup(descr);
							}
						}
					}
					else{
						*result= PyFloat_AsDouble(ReturnValue);
					}
#else
					_python_expression_value( ReturnValue, result, exportName, &af, descr );
#endif
					Py_XDECREF(ReturnValue);
				}
				else{
					ok= 0;
				}
			}
			else{
				_python_expression_value( retval, result, exportName, &af, descr );
			}
			if( PyErr_Occurred() ){
				PyErr_Print();
				ok= 0;
			}
			Py_XDECREF(retval);
		}
		return(ok);
	}
	else{
		return(0);
	}
}

int Evaluate_Python_Expr( char *expr, double *result )
{ int dum= 0;
	return( Python_Expression_Value( _Evaluate_Python_Expr(expr,&dum), result, expr, NULL, NULL ) );
}

int ascanf_PythonEvalValue ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  ascanf_Function *pexpr;
  int i;
	ascanf_arg_error= 0;
	set_NaN(*result);
	if( !args || ascanf_arguments< 1 ){
		ascanf_emsg= " (need at least 1 argument)";
		ascanf_arg_error= 1;
	}
	else{
		for( i= 0; i< ascanf_arguments; i++ ){
			if( (pexpr= parse_ascanf_address( args[i], 0, "ascanf_PythonEvalValue", (int) ascanf_verbose, NULL ))
				&& pexpr->usage
			){
				if( !ascanf_SyntaxCheck ){
					ascanf_arg_error= !Python_Expression_Value( _Evaluate_Python_Expr(pexpr->usage ,&pexpr->PythonEvalType),
										result, pexpr->usage, NULL, NULL );
				}
			}
		}
	}
	return( !ascanf_arg_error );
}

PyObject *_Evaluate_Python_CExpr( Compiled_Python *CP )
{ PyObject *res= NULL;
  int pA= pythonActive;

	PyErr_Clear();
	pythonActive+= 1;
	res= PyEval_EvalCode( CP->cexpr, MainDictionary, MainDictionary );
	if( !res && CP->start== Py_eval_input ){
		if( ascanf_verbose ){
			fprintf( StdErr, " (PyEval_EvalCode with Py_eval_input returned NULL, trying with Py_file_input) ");
		}
		PyErr_Clear();
		CP->start= Py_file_input;
		Py_XDECREF( CP->cexpr );
		if( (CP->cexpr= (XGPyCodeObject*) Py_CompileString( CP->code, "<ascanf>", CP->start )) ){
			PyErr_Clear();
			res= PyEval_EvalCode( CP->cexpr, MainDictionary, MainDictionary );
		}
		else{
			ascanf_arg_error= 1;
			memset( CP, 0, sizeof(Compiled_Python) );
		}
	}
	pythonActive= pA;
	if( PyErr_Occurred() ){
		PyErr_Print();
		PyErr_Clear();
		res= NULL;
	}
	return(res);
}

int ascanf_PythonEvalValueCExpr ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  int i;
	ascanf_arg_error= 0;
	set_NaN(*result);
	if( !args || ascanf_arguments< 1 ){
		ascanf_emsg= " (need at least 1 argument)";
		ascanf_arg_error= 1;
	}
	else{
		for( i= 0; i< ascanf_arguments; i++ ){
		  int enr;
			enr= (int) args[i];
			if( enr >= 0 && enr < Compiled_ExpressionListEntries
				&& Compiled_ExpressionList[enr].cexpr
			){
			  PyObject *res;
				if( ascanf_verbose ){
					fprintf( StdErr, "## %s\n",
						Compiled_ExpressionList[enr].code
					);
				}
				res= _Evaluate_Python_CExpr( &Compiled_ExpressionList[enr] );
				if( !res ){
					set_NaN(*result);
					ascanf_arg_error= 1;
				}
				else{
					ascanf_arg_error= !Python_Expression_Value( res, result, Compiled_ExpressionList[enr].code, NULL, NULL );
				}
			}
			else{
				fprintf( StdErr, " (illegal cexpr %s) ", ad2str(args[i], d3str_format, NULL) );
			}
		}
	}
	return( !ascanf_arg_error );
}

int ascanf_PythonEvalFile ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  ascanf_Function *pexpr;
  int i;
	ascanf_arg_error= 0;
	set_NaN(*result);
	if( !args || ascanf_arguments< 1 ){
		ascanf_emsg= " (need at least 1 argument)";
		ascanf_arg_error= 1;
	}
	else{
		for( i= 0; i< ascanf_arguments; i++ ){
			if( (pexpr= parse_ascanf_address( args[i], 0, "ascanf_PythonEvalFile", (int) ascanf_verbose, NULL ))
				&& pexpr->usage
			){
				if( !ascanf_SyntaxCheck ){
					*result= Import_Python_File( pexpr->usage, NULL, 0, False );
				}
			}
		}
	}
	return( !ascanf_arg_error );
}

int _ascanf_PythonCall( ascanf_Function *pfunc, int argc, double *args, double *result )
{ int startArg, ret= 0;
  PyObject *func= NULL, *pargs;
  const char *fname= NULL;
  int pA= pythonActive;
  char *exportName = "$Python-Call-Result";

	if( pfunc->PythonHasReturnVar ){
		if( argc < 1 || !(pfunc->PyObject_ReturnVar = parse_ascanf_address( args[0], 0,
				"ascanf_PythonCall", (int) ascanf_verbose, NULL ))
		){
			fprintf( StdErr, " (must specify a pointer to a return variable in the first argument for this Python object!)== " );
			ascanf_arg_error = 1;
			return(0);
		}
		else{
			exportName = pfunc->PyObject_ReturnVar->name;
			startArg = 1;
		}
// 		argc -= 1;
	}
	else{
		startArg = 0;
	}

	if( pfunc->usage && *pfunc->usage && (pfunc->type!= _ascanf_python_object || pfunc->take_usage) ){
		if( (func= _Evaluate_Python_Expr( pfunc->usage, &pfunc->PythonEvalType )) ){
		 /*  20090113: if we passed a string that contains a "full" function call (e.g. "foo(1,2)"),
		  \ it will now have been evaluated, and func will actually point to the call's result,
		  \ NOT to a callable object!
		  */
			pythonActive+= 1;
			fname= PyEval_GetFuncName(func);
			pythonActive= pA;
		}
	}
	else{
		func= pfunc->PyObject;
		fname= pfunc->PyObject_Name;
		Py_XINCREF(func);
	}
	if( func ){
		  // 20100507: it's unclear why, but calling Py_EnterRecursiveCall() from a readline idle handler while
		  // we're in a Python-Shell provokes a crash. We attempt to prevent that by comparing the current
		  // readline idle handler recursion counter with the value it had when Python-Shell was called.
// 		if( grl_HandlingEvents_ref>=0 && grl_HandlingEvents != grl_HandlingEvents_ref )
		if( inInteractiveShell() )
		{
		  static const char *last_fname = NULL;
			if( fname != last_fname ){
				fprintf( StdErr, "## not calling %s (= %s) while in a Python-Shell!\n", pfunc->name, fname );
				last_fname = fname;
			}
			return(!ascanf_arg_error);
		}
		if( PyCallable_Check(func) ){
			if( !(pargs= PyTuple_New( argc-startArg )) ){
				if( PyErr_Occurred() ){
					PyErr_Print();
				}
			}
			if( pargs ){
			  int i, j;
				  /* we don't check if func is callable. Calling it will fail if not, and set/print a sensible
				   \ error message (hard to compose ourselves, as there's no good way to get an object's name)
				   */
				for( j= 0, i= startArg; i< argc; i++, j++ ){
				  ascanf_Function *af;
				  PyObject *arg;
					if( (af= parse_ascanf_address( args[i], 0, "ascanf_PythonCall", (int) ascanf_verbose, NULL )) ){
						arg= Py_ImportVariableFromAscanf( &af, &af->name, 0, NULL, 0, 0 );
					}
					else{
						arg= PyFloat_FromDouble(args[i]);
					}
					PyTuple_SetItem(pargs, j, arg );
				}
				if( ascanf_verbose ){
					fprintf( StdErr, " (calling Python function %s (%s) with %d argument(s)) ",
						pfunc->name, (fname)? fname : "??", argc-startArg
					);
					fflush( StdErr );
				}
				Py_XINCREF(pargs);
				if( !Py_EnterRecursiveCall( "_ascanf_PythonCall():" STRING(__LINE__) ) ){
					pythonActive+= 1;
					ret= Python_Expression_Value( PyObject_CallObject( func, pargs ), result,
						(pfunc->usage)? pfunc->usage : pfunc->name, pfunc->PyObject_ReturnVar, exportName );
					pythonActive= pA;
					Py_LeaveRecursiveCall();
				}
				if( PyErr_Occurred() ){
					PyErr_Print();
				}
				Py_XDECREF(pargs);
			}
		}
		else{
			if( ascanf_verbose ){
				fprintf( StdErr, " (%s already called, returned a(n) %s, collecting result(s)) ",
					pfunc->name, (fname)? fname : "??"
				);
				fflush( StdErr );
			}
			pythonActive+= 1;
			ret= Python_Expression_Value( func, result,
				(pfunc->usage)? pfunc->usage : pfunc->name, pfunc->PyObject_ReturnVar, exportName );
			pythonActive= pA;
			if( PyErr_Occurred() ){
				PyErr_Print();
			}
		}
		Py_XDECREF(func);
	}
	return( ret );
}

int ascanf_PythonCall ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  ascanf_Function *pfunc;
	ascanf_arg_error= 0;
	set_NaN(*result);
	if( !args || ascanf_arguments< 1 ){
		ascanf_emsg= " (need at least 1 argument)";
		ascanf_arg_error= 1;
	}
	else{
	  int take_usage;
		if( (pfunc= parse_ascanf_address( args[0], 0, "ascanf_PythonCall", (int) ascanf_verbose, &take_usage ))
			&& (pfunc->type== _ascanf_python_object || pfunc->usage)
		){ int tu= pfunc->take_usage;
			pfunc->take_usage= take_usage;
			ascanf_arg_error= !_ascanf_PythonCall( pfunc, ascanf_arguments-1, &args[1], result );
			pfunc->take_usage= tu;
		}
		else{
			fprintf( StdErr, " (PythonCall: first argument must be a string or a pointer to a callable PyObject) ");
			ascanf_emsg= "invalid argument";
			ascanf_arg_error= 1;
		}
	}
	return( !ascanf_arg_error );
}

static void InitModules(int);

int ascanf_PythonReInit ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
	ascanf_arg_error= 0;
	set_NaN(*result);
	if( args && ASCANF_TRUE(args[0]) ){
		InitModules(0);
	}
	return( !ascanf_arg_error );
}

static ascanf_Function Python_Function[] = {
	{ "Python-Call", ascanf_PythonCall, AMAXARGS, NOT_EOF_OR_RETURN,
		"Python-Call[PObj|name[,args,..]]: call the Python object referenced by PObj\n"
		" or <name>, passing the optionally specified arguments. Returns the return\n"
		" value of the expression, or the value of the Python variable ascanf.ReturnValue.\n"
		" The return value is a number when the Python callable returns a numerical scalar,\n"
		" a pointer to an ascanf object (typically an array) if possible, and a PObj otherwise\n"
		" (a non-homogenous list, for instance the elements of which can be accessed through\n"
		" ad-hoc Python functions).\n"
		" The last returned value(s) is/are available also in $Python-Call-Result, a pointer\n"
		" to which is returned if the return value is not a scalar. When calling with aPObj,\n"
		" and if <returnVar> was true when the PObj was obtained, the first in the <args> list\n"
		" must be a pointer to an ascanf variable that will hold the return value instead of\n"
		" $Python-Call-Result.\n"
		" A PObj can be obtained by calling ascanf.ExportVariable with as_PObj=True. They\n"
		" can also be called directly (e.g. PObj[1,2,3] or PObj[,] for passing 0 arguments).\n"
		" PObjs must be declared before compiling code that calls them directly.\n"
	},
	{ "Python-Eval", ascanf_PythonEval, AMAXARGS, NOT_EOF_OR_RETURN,
		"Evaluate all string arguments using the embedded Python interpreter.\n"
	},
	{ "Python-EvalValue", ascanf_PythonEvalValue, AMAXARGS, NOT_EOF_OR_RETURN,
		"Evaluate all string arguments using the embedded Python interpreter.\n"
		" Returns the return value of the last expression, or the value of the.\n"
		" Python variable ascanf.ReturnValue .\n"
	},
	{ "Python-EvalFile", ascanf_PythonEvalFile, AMAXARGS, NOT_EOF_OR_RETURN,
		"Process all the specified files using the embedded Python interpreter.\n"
	},
	{ "Python-Compile", ascanf_PythonCompile, 1, NOT_EOF,
		"Compile the string arguments using the embedded Python\n"
		" and return an compiled expression (cexpr) ID.\n"
	},
	{ "Python-DelCompiled", ascanf_PythonDestroyCExpr, AMAXARGS, NOT_EOF_OR_RETURN,
		"Release all the compiled Python expressions specified by the ID codes\n"
	},
	{ "Python-PrintCompiled", ascanf_PythonPrintCExpr, AMAXARGS, NOT_EOF_OR_RETURN,
		"Print all the specified compiled expressions.\n"
	},
	{ "Python-EvalCompiled", ascanf_PythonEvalCExpr, AMAXARGS, NOT_EOF_OR_RETURN,
		"Evaluate all the specified compiled expressions.\n"
	},
	{ "Python-EvalValueCompiled", ascanf_PythonEvalValueCExpr, AMAXARGS, NOT_EOF_OR_RETURN,
		"Evaluate all the specified compiled expressions and return the value of the last.\n"
	},
	{ "Python-Shell", ascanf_PythonShell, 1, NOT_EOF_OR_RETURN,
		"Fire up an interactive Python shell. Currently, this only works when not detached.\n"
		" from the calling terminal.\n"
		" Call with a negative argument to use the interact() method from the (standard) code package,\n"
		" otherwise the more advanced IPython package will be used.\n"
		" Call with a single False argument to just initialise the shell (makes available\n"
		" xgraph.ipshell() to Python code).\n"
		" NB: currently, even initialising the IPython shell interferes with correct error reporting.\n"
	},
	{ "Python-ReInit", ascanf_PythonReInit, 1, NOT_EOF,
		"Python-ReInit[True]: re-initialise part of the Python functionality (recreate ascanf and xgraph modules).\n"
		" A use-at-your-own risk function!\n"
	},
};
static int Python_Functions= sizeof(Python_Function)/sizeof(ascanf_Function);

static void af_initialise( DyModLists *new, char *label )
{ ascanf_Function *af= Python_Function;
  static char called= 0;
  int i;
  char buf[64];

	for( i= 0; i< Python_Functions; i++, af++ ){
		if( !called ){
			if( af->name ){
				af->name= XGstrdup( af->name );
			}
			else{
				sprintf( buf, "Function-%d", i );
				af->name= XGstrdup( buf );
			}
			if( af->usage ){
				af->usage= XGstrdup( af->usage );
			}
			ascanf_CheckFunction(af);
			if( af->function!= ascanf_Variable ){
				set_NaN(af->value);
			}
			if( label ){
				af->label= XGstrdup( label );
			}
			Check_Doubles_Ascanf( af, label, True );
		}
		af->dymod= new;
	}
	called+= 1;
}

PyObject *XG_PythonError= NULL;

#if 0
### PyArg_ParseTuple format argument:
	s (string or Unicode object) [const char *] Convert a Python string or Unicode object to a C pointer to a character
string. You must not provide storage for the string itself; a pointer to an existing string is stored into
the character pointer variable whose address you pass. The C string is NUL-terminated. The Python string
must not contain embedded NUL bytes; if it does, a TypeError exception is raised. Unicode objects are
converted to C strings using the default encoding. If this conversion fails, a UnicodeError is raised.
	s# (string, Unicode or any read buffer compatible object) [const char *, int] This variant on 's' stores into
two C variables, the first one a pointer to a character string, the second one its length. In this case the Python
string may contain embedded null bytes. Unicode objects pass back a pointer to the default encoded string
version of the object if such a conversion is possible. All other read-buffer compatible objects pass back a
reference to the raw internal data representation.
	z (string or None) [const char *] Like 's', but the Python object may also be None, in which case the C
pointer is set to NULL.
	z# (string or None or any read buffer compatible object) [const char *, int] This is to 's#' as 'z' is to 's'.
	b/B : signed/unsigned char (also: c)
	h/H : signed/unsigned short
	i/I
	l/k : long/unsigned long
	L/K : long long/unsigned long long
	f/d : float/double
	O : PyObject*
	O! : ReferenceTypePyObject*,PyObject* initialises PyObject if the argument matches the reference type, else TypeError
	S : PyStringObject* or PyObject*  : obtain a reference from a string object

	format codes after a | indicate optional arguments
	: ends the format list, indicates the name of the function in raised errors
#endif

int python_check_int=100, python_check_now=0;

inline int python_check_interrupted()
{
	if( ascanf_interrupt< 0 ){
		ascanf_interrupt= 0;
	}
	if( ascanf_escape< 0 ){
		ascanf_escape= 0;
	}
	if( python_check_int ){
		if( python_check_now== 0 ){
			ascanf_check_event( "python_check_interrupted()" );
		}
		python_check_now= (python_check_now+1) % python_check_int;
	}
	if( ascanf_interrupt ){
 		PyErr_SetString( XG_PythonError, "ascanf interrupt requested" );
// 		PyErr_SetString(  PyExc_KeyboardInterrupt, "ascanf interrupt requested" );
		ascanf_interrupt= -1;
		return(1);
	}
	else if( ascanf_escape ){
 		PyErr_SetString( XG_PythonError, "(ascanf) evaluation needs to be interrupted" );
// 		PyErr_SetString(  PyExc_KeyboardInterrupt, "(ascanf) evaluation needs to be interrupted" );
		ascanf_escape= -1;
		return(1);
	}
	else{
		return(0);
	}
}

#ifdef DEBUG
static PyObject* python_ListVars(PyObject *self, PyObject *args)
{
	CHECK_INTERRUPTED();
	if( !PyArg_ParseTuple(args, ":ListVariables") ){
		return NULL;
	}
	show_ascanf_functions( StdErr, "\t", True, 1 );
	Py_RETURN_NONE;
}
#endif

static PyObject* python_AscanfEval(PyObject *self, PyObject *args, PyObject *kw )
{ char *expr;
  char *kws[]= { "expression", "dereference", "N", "verbose", NULL };
  double *result;
  int i, N= 1, r, deref= 0, verbose=0;
  PyObject *ret = NULL, *retTuple = NULL;
	CHECK_INTERRUPTED();
	if( !PyArg_ParseTupleAndKeywords(args, kw, "s|iii:Eval", kws, &expr, &deref, &N, &verbose ) || !N ){
		return( NULL );
	}
	if( N < 0 ){
		N = 1;
	}
	if( (result = (double*) malloc( N * sizeof(double) )) ){
	  int aV = ascanf_verbose;
		for( i = 0 ; i < N ; i++ ){
			set_NaN(result[i]);
		}
		if( verbose ){
			ascanf_verbose = 1;
		}
		new_param_now( NULL, NULL, 0 );
		r = new_param_now( expr, result, N );
		if( verbose <= 1 ){
			ascanf_verbose = aV;
		}
		if( r <= 0 ){
			Py_RETURN_NONE;
		}
		else if( r > N ){
			fprintf( StdErr, "## Warning: %d expressions were evaluated instead of N=%d!\n", r, N );
			r = N;
		}
		if( r > 1 ){
			if( !(retTuple = PyTuple_New(r)) ){
				fprintf( StdErr, "## failure allocating %d element return tuple (%s)\n", r, serror() );
				PyErr_NoMemory();
				ascanf_verbose = aV;
				return NULL;
			}
		}
		for( i = 0 ; i < r ; i++ ){
			if( deref ){
			  int take_usage;
			  ascanf_Function *af= parse_ascanf_address( result[i], 0, "python_AscanfEval", (int) ascanf_verbose, &take_usage );
				if( af ){
					deref= abs(deref);
					ret = ( Py_ImportVariableFromAscanf( &af, &af->name, 0, NULL, deref-1, 0 ) );
				}
			}
			else{
				ret = ( Py_BuildValue("d", result[i]) );
			}
			if( retTuple ){
				PyTuple_SetItem( retTuple, i, ret );
			}
		}
		if( retTuple ){
			ret = retTuple;
		}
		xfree(result);
		ascanf_verbose = aV;
	}
	else{
		PyErr_NoMemory();
	}
	return ret;
}

static PyObject* python_source(PyObject *self, PyObject *args, PyObject *kw )
{ char *expr;
  char *kws[]= { "filename", NULL };
	CHECK_INTERRUPTED();
	if( !PyArg_ParseTupleAndKeywords(args, kw, "s:source", kws, &expr ) ){
		return( NULL );
	}
	return( Py_BuildValue( "i", Import_Python_File( expr, NULL, 0, False ) ) );
}


static PyObject* python_ad2str(PyObject *self, PyObject *args, PyObject *kw )
{ char *kws[]= { "value", NULL };
  PyObject *arg;
  double value;
	CHECK_INTERRUPTED();
	if( !PyArg_ParseTupleAndKeywords(args, kw, "O:Value2Str", kws, &arg ) ){
		return( NULL );
	}
	if( PyFloat_Check(arg) ){
		value= PyFloat_AsDouble(arg);
	}
	else if( PyInt_Check(arg) || PyLong_Check(arg) ){
		value= PyInt_AsLong(arg);
	}
	else if( PyBytes_Check(arg) || PyUnicode_Check(arg) ){
		Py_INCREF(arg);
		return(arg);
	}
	else{
 		PyErr_SetString( XG_PythonError, "value must be a scalar, int or float or string" );
// 		PyErr_SetString(  PyExc_TypeError, "value must be a scalar, int or float or string" );
		return(NULL);
	}
	{ char *c= ad2str( value, d3str_format, NULL);
		if( c ){
		  int len= strlen(c);
			if( c[0]== '`' && c[1]== '"' && c[len-1]== '"' ){
				c[len-1]= '\0';
				strcpy( c, &c[2] );
			}
		}
		return( PyString_FromString(c ) );
	}
}

#ifdef DEBUG

/* This is a test of how to create and return arrays: */
static PyObject* python_IdArray(PyObject *self, PyObject *args, PyObject *kw )
{ npy_intp dim[2]= {0,0}, ndim= 0;
  char *kws[]= { "rows", "columns", "name", NULL }, *pname=NULL;
  PyObject *parray= NULL;
  double *array;
	CHECK_INTERRUPTED();
	if( !PyArg_ParseTupleAndKeywords(args, kw, "|IIs:IdArray", kws, &dim[0], &dim[1], &pname ) ){
		return NULL;
	}
	if( dim[0]< 0 ){
		dim[0]= 0;
	}
	if( dim[1]< 0 ){
		dim[1]= 0;
	}
	if( dim[0]== 0 && dim[1]== 0 ){
		ndim= 0;
	}
	else if( dim[0] && dim[1] ){
		ndim= 2;
	}
	else{
		if( dim[1] ){
			dim[0]= dim[1];
			dim[1]= 1;
		}
		else{
			dim[1]= 1;
		}
		ndim= 1;
	}
	if( ndim ){
		if( (array= (double*) PyMem_New( double, dim[0] * dim[1] )) ){
		  npy_intp i;
		  	for( i= 0; i< dim[0]*dim[1]; i++ ){
				array[i]= i;
			}
/* 			parray= PyArray_FromDimsAndData( ndim, dim, PyArray_DOUBLE, (char*) array);	*/
			parray= PyArray_SimpleNewFromData( ndim, dim, PyArray_DOUBLE, (void*) array);
			PyArray_ENABLEFLAGS( (PyArrayObject*)parray, NPY_OWNDATA );
			if( pname ){
				Py_XINCREF(parray);
				PyModule_AddObject( AscanfPythonModule, pname, parray );
			}
		}
		else{
			PyErr_NoMemory();
			return(NULL);
		}
	}
	else{
 		PyErr_SetString( XG_PythonError, "at least one positive dimension should be set!" );
// 		PyErr_SetString(  PyExc_ValueError, "at least one positive dimension should be set!" );
	}
	return( parray );
}
#endif

static PyObject* python_WaitEvent(PyObject *self, PyObject *args, PyObject *kw )
{ int type= 0;
  char *kws[]= { "type", "message", NULL };
  char *typeText, *msg= NULL;
  PyObject *ret;
	CHECK_INTERRUPTED();
	if( !PyArg_ParseTupleAndKeywords(args, kw, "|zz:WaitEvent", kws, &typeText, &msg ) ){
		return NULL;
	}
	if( typeText ){
		if( strncasecmp( typeText, "key", 3 )== 0 ){
			type= KeyPress;
		}
		else{
 			PyErr_SetString( XG_PythonError, "unknown or invalid event specification ignored" );
// 			PyErr_SetString(  PyExc_SyntaxError, "unknown or invalid event specification ignored" );
		}
	}
	ret= Py_BuildValue( "d", ascanf_WaitForEvent_h( type, msg, "python_WaitEvent" ) );
	CHECK_INTERRUPTED();
	return(ret);
}

static PyObject* python_NoOp(PyObject *self, PyObject *args )
{
	CHECK_INTERRUPTED();
	Py_RETURN_NONE;
}

static PyObject* python_CheckEvent(PyObject *self, PyObject *args, PyObject *kw )
{ char *kws[]= { "interval", "python_interval", NULL };
  PyObject *ret;
// 	CHECK_INTERRUPTED();
	if( !PyArg_ParseTupleAndKeywords(args, kw, "|ii:CheckEvent", kws, &(ascanf_check_int), &python_check_int ) ){
		return( NULL );
	}
	ret= Py_BuildValue( "i", ascanf_check_event( "python_CheckEvent" ) );
	CHECK_INTERRUPTED();
	return(ret);
}

static PyObject* python_RedrawNow(PyObject *self, PyObject *args, PyObject *kw )
{ char *kws[]= { "silenced", "all", "update", NULL };
  int silenced= 0, all= 0, update= 0;
// 	CHECK_INTERRUPTED();
	if( !PyArg_ParseTupleAndKeywords(args, kw, "|iii:RedrawNow", kws, &silenced, &all, &update ) ){
		return( NULL );
	}
	if( !ActiveWin ){
		fprintf( StdErr, "## No active window defined, select one by redrawing it through the GUI!\n" );
	}
	_ascanf_RedrawNow( !silenced, all, update );
	CHECK_INTERRUPTED();
	Py_RETURN_NONE;
}

static PyObject* python_Python_grl_HandleEvents(PyObject *self, PyObject *arg, PyObject *kws )
{
// no interrupt checking here...
// 	CHECK_INTERRUPTED();
	(void) Python_grl_HandleEvents();
	CHECK_INTERRUPTED();
	Py_RETURN_NONE;
}

static PyObject* python_HandleEvents(PyObject *self, PyObject *args, PyObject *kw )
{ char *kws[]= { "caller", NULL };
  char *caller= "python_HandleEvents";
  int ret;
// no interrupt checking here...
// 	CHECK_INTERRUPTED();
	if( !PyArg_ParseTupleAndKeywords(args, kw, "|s:HandleEvents", kws, &caller ) ){
		return( NULL );
	}
	ret= Handle_An_Events( -1, 1, caller, 0, 0 );
	CHECK_INTERRUPTED();
	return( Py_BuildValue( "i", ret ) );
}

static PyObject* python_RedrawSet(PyObject *self, PyObject *args, PyObject *kw )
{ char *kws[]= { "set", "immediate", NULL };
  int set= 0, immediate= 0, ret;
	CHECK_INTERRUPTED();
	if( !PyArg_ParseTupleAndKeywords(args, kw, "i|i:RedrawSet", kws, &set, &immediate ) ){
		return( NULL );
	}
	if( set< 0 || set>= setNumber ){
 		PyErr_SetString( XG_PythonError, "invalid setNumber" );
// 		PyErr_SetString(  PyExc_ValueError, "invalid setNumber" );
		return(NULL);
	}
	ret= RedrawSet( set, immediate );
	CHECK_INTERRUPTED();
	return( Py_BuildValue("i", ret) );
}

static PyObject* python_gtkcons(PyObject *self, PyObject *args, PyObject *kw )
{ static char active= 0;
	CHECK_INTERRUPTED();
	if( !active && init_gtkcons() ){
		active= 1;
		Run_Python_Expr(
			"try:\n"
				"\txgraph.gtkconsole()\n"
			"except:\n"
				"\tpass\n"
		);
		active= 0;
	}
	else{
		PyErr_Warn( PyExc_Warning, "Could not initialise Gtk shell!" );
	}
	CHECK_INTERRUPTED();
	Py_RETURN_NONE;
}

static PyObject* python_tkcons(PyObject *self, PyObject *args, PyObject *kw )
{ static char active= 0;
	CHECK_INTERRUPTED();
	if( !active && init_tkcons() ){
		active= 1;
		Run_Python_Expr(
			"try:\n"
				"\txgraph.tkconsole()\n"
			"except:\n"
				"\tpass\n"
		);
		active= 0;
	}
	else{
		PyErr_Warn( PyExc_Warning, "Could not initialise Tk shell!" );
	}
	CHECK_INTERRUPTED();
	Py_RETURN_NONE;
}

static PyObject* python_interact(PyObject *self, PyObject *args, PyObject *kw )
{ static char active= 0;
	CHECK_INTERRUPTED();
	if( !active && init_interact() ){
		active= 1;
		Run_Python_Expr( "xgraph.interact()" );
		active= 0;
	}
	else{
		PyErr_Warn( PyExc_Warning, "Could not initialise embedded code.interact shell!" );
	}
	CHECK_INTERRUPTED();
	Py_RETURN_NONE;
}

static PyObject* python_ipshell(PyObject *self, PyObject *args, PyObject *kw )
{ static char active= 0;
	CHECK_INTERRUPTED();
// 	if( isatty(fileno(stdin)) && isatty(fileno(stdout)) ){
		if( !active && init_ipshell() ){
			active= 1;
			Run_Python_Expr( "xgraph.ipshell()" );
			active= 0;
		}
		else{
			PyErr_Warn( PyExc_Warning, "Could not initialise embedded IPython shell!" );
		}
// 	}
// 	else{
// 		PyErr_Warn( PyExc_Warning, "Not on an interactive terminal, not starting any embedded IPython shells!" );
// 	}
	CHECK_INTERRUPTED();
	Py_RETURN_NONE;
}

ascanf_Function *Py_getNamedAscanfVariable( char *name )
{ ascanf_Function *af;
  int aSC = ascanf_SyntaxCheck;
	// 20101025: the code below might result in internal callbacks being called. There is no real need for that for
	// what we want, and if ever it happens, there should be no side-effects. So, we pretend we're compiling and only
	// need to do a syntax-check ... which is quite correct in fact.
	ascanf_SyntaxCheck = 1;
	if( !(af= get_VariableWithName(name, False)) ){
#if 1
	  double dum;
		  // 20100610: I'd forgotten about find_ascanf_function(), which uses the same method as the hack
		  // below, but without all the overhead.
		if( (af = find_ascanf_function( name, &dum, NULL, "Py_getNamedAscanfVariable()" )) ){
			register_VariableName(af);
		}
#else
	  /* Aweful hack. I am not sure that all variables are properly registered in the Name->Variable hash_map (see ascanfcMap.cc).
	   \ Hence, we evaluate the expression
	   \ "&name"
	   \ when we don't get an immediate hit. If the variable exists, it will then be registered.
	   */
	  char *addr= concat( "&", name, NULL );
		if( addr ){
		  int n= 1;
		  double result= 0;
		  int aAVWCm= ascanf_AutoVarWouldCreate_msg;
			  /* Prevent "compiler would create" warnings when <name> doesn't exist: */
			ascanf_AutoVarWouldCreate_msg= False;
			fascanf2( &n, addr, &result, ',' );
			if( !(af= get_VariableWithName(name)) && n ){
				af= parse_ascanf_address( result, 0, "Py_getNamedAscanfVariable", (int) ascanf_verbose, NULL );
				  // if the evaluation of the "&name" expression above has no effect, that means that the address
				  // of that variable had already been determined once, before that action included registering
				  // variable names. Thus, we register the variable's name, such that a next time around we need to
				  // do less work:
				register_VariableName(af);
			}
			xfree(addr);
			ascanf_AutoVarWouldCreate_msg= aAVWCm;
		}
#endif
	}
	ascanf_SyntaxCheck = aSC;
	return(af);
}

#ifdef USE_COBJ
static void PCO_destructor( void *ptr, void *desc )
{ ascanf_Function *af= ptr;
	if( !af || af->function!= desc ){
		fprintf( StdErr, "PCO_destructor(%p,%p) called with something that's not an ascanf reference!\n",
			ptr, desc
		);
	}
}
#endif

int Py_ImportVariable_Copies= True;

PyObject *Py_ImportVariableFromAscanf( ascanf_Function **Af, char **Name, int Ndims, npy_intp *dim, int deref, int force_address )
{ int is_usage= False, is_address= force_address;
  npy_intp Dim[2];
  PyObject *ret= NULL;
  ascanf_Function *af, laf;
  int PIVC= Py_ImportVariable_Copies;
  char *name, quickImport= False;

	// This is for safety!!
	Py_ImportVariable_Copies= True;

	af= (Af)? *Af : NULL;
	name= (Name)? *Name : NULL;

	if( !af ){
		if( name ){
			do{
				switch(*name){
					case '&':
						name++;
						is_address= True;
						break;
					case '`':
						is_usage= True;
						name++;
						break;
				}
			} while( *name && (*name== '&' || *name== '`') );
			*Name= name;

			af= Py_getNamedAscanfVariable(name);
		}
		else{
			af= NULL;
		}
	}
	else if( name && af->name == name ){
	  /* 20081204: we don't import from a name only, we already have the ascanf object that
	   \ simply needs to be converted to a Python representation.
	   */
		quickImport= True;
	}

	if( af ){

		if( Af ){
			*Af= af;
		}

		if( af->own_address ){
		  /* parse_ascanf_address doesn't just parse an external representation of the Variable's address. It also ensures
		   \ that specific type(s) of variables are uptodate. Hence, parse af->own_address
		   */
		  int auaa= AlwaysUpdateAutoArrays;
			AlwaysUpdateAutoArrays= True;
			parse_ascanf_address( af->own_address, 0, "Py_ImportVariableFromAscanf", (int) ascanf_verbose, NULL );
			AlwaysUpdateAutoArrays= auaa;
		}
		if( is_usage ){
			if( af->usage ){
				ret= PyString_FromString( af->usage );
			}
			else{
				goto force_string;
			}
		}
		else if( is_address ){
#ifdef PRE_AOBJECT
			if( deref ){
			  /* Make a local copy so that we can store our own address without modifying the external representation */
				if( !(af->type== _ascanf_function || af->type== _ascanf_procedure
						|| af->type== NOT_EOF || af->type== NOT_EOF_OR_RETURN)
				){
					PyErr_Warn( PyExc_Warning, "forced dereferencing of the address of a variable should return itself" );
					laf= *af;
					laf.value= (af->own_address)? af->own_address : take_ascanf_address(af);
					af= &laf;
					goto deref_value;
				}
				else{
#ifdef USE_COBJ
					ret= PyCObject_FromVoidPtrAndDesc(af, af->function, PCO_destructor);
#else
					ret= PyAscanfObject_FromAscanfFunction(af);
#endif
					ret= Py_BuildValue( "O", ret );
				}
			}
			else{
				ret= Py_BuildValue( "d", (af->own_address)? af->own_address : take_ascanf_address(af) );
			}
#else
			if( deref ){
			  /* Make a local copy so that we can store our own address without modifying the external representation */
				if( !(af->type== _ascanf_function || af->type== _ascanf_procedure
						|| af->type== NOT_EOF || af->type== NOT_EOF_OR_RETURN)
				){
					PyErr_Warn( PyExc_Warning, "forced dereferencing of the address of a variable should return itself" );
					laf= *af;
					laf.value= (af->own_address)? af->own_address : take_ascanf_address(af);
					af= &laf;
					goto deref_value;
				}
			}
			ret= PyAscanfObject_FromAscanfFunction(af);
			ret= Py_BuildValue( "O", ret );
#endif
		}
		else{
deref_value:;
			  /* suppose user did DCL[foo, {0,bar,3}]; af->value will have a pointer value when importing foo.
			   \ We have no way of knowning if the pointer value should be exported to python, or the value pointed to.
			   \ However, if declared DCL[ &foo, {0,bar,3}], then dereference the pointer value. Warn if impossible,
			   \ and return the pointer value.
			   */
			if( !quickImport && ((af->is_address && !af->is_usage) || deref) ){
			  ascanf_Function *aaf;
			  int auaa= AlwaysUpdateAutoArrays;
				AlwaysUpdateAutoArrays= True;
				if( (aaf= parse_ascanf_address( af->value, 0, "Py_ImportVariableFromAscanf", (int) ascanf_verbose, NULL )) ){
					af= aaf;
				}
				else{
					PyErr_Warn( PyExc_Warning, "error resolving ascanf pointer variable: returning its value" );
				}
				AlwaysUpdateAutoArrays= auaa;
			}
			if( af->fp ){
//				PyErr_Warn( PyExc_Warning, "open-mode of ascanf files is unknown" );
				if( af->usage ){
					ret= PyFile_FromFile( af->fp, af->usage, (af->fp_mode)? af->fp_mode : "?", Python_No_fclosing );
				}
				else{
					ret= PyFile_FromFile( af->fp, "<open ascanf file>", (af->fp_mode)? af->fp_mode : "?", Python_No_fclosing );
				}
			}
			else if( af->is_usage )
/* 			else if( af->take_usage )	*/
			{
force_string:;
			  ascanf_Function *string= parse_ascanf_address( af->value, 0, "Py_ImportVariableFromAscanf", 0, NULL );
				if( string && string->usage ){
					ret= PyString_FromString( string->usage );
				}
				else{
					ret= PyString_FromString( af->usage );
				}
			}
			else switch( af->type ){
				case _ascanf_array:{
					if( !dim ){
						dim= Dim;
						dim[0]= af->N;
						dim[1]= 1;
						Ndims= 1;
					}
					else{
					  int i, size;
						size= dim[0];
						for( i= 1; i< Ndims; i++ ){
							size*= dim[i];
						}
						if( size< af->N ){
							if( ascanf_verbose ){
								fprintf( StdErr, " (warning: dimensionlist gives size=%d which does not use all %d elements in %s) ",
									size, af->N, af->name
								);
							}
						}
						else if( size> af->N ){
						  char emsg[256];
							snprintf( emsg, sizeof(emsg)/sizeof(char),
								"ignoring size %dx%d which would give more than %d elements in %s - reshape yourself!",
								dim[0], dim[1], af->N, af->name
							);
							if( (PyErr_Warn( PyExc_Warning, emsg )) ){
								return(NULL);
							}
							dim= Dim;
							dim[0]= af->N;
							dim[1]= 1;
							Ndims= 1;
						}
					}
					if( af->iarray ){
					  int *array= (PIVC)? NULL : af->iarray;
						if( array || (array= (int*) PyMem_New( int, af->N )) ){
							memcpy( array, af->iarray, af->N*sizeof(int) );
/* 							ret= PyArray_FromDimsAndData( Ndims, dim, PyArray_INT, (char*) array);	*/
							ret= PyArray_SimpleNewFromData( Ndims, dim, PyArray_INT, (void*) array);
							if( PIVC ){
								PyArray_ENABLEFLAGS( (PyArrayObject*)ret, NPY_OWNDATA );
							}
						}
						else{
							PyErr_NoMemory();
							return(NULL);
						}
					}
					else{
					  double *array= (PIVC)? NULL : af->array;
					  PyObject **strings;
						if( (strings= (PyObject **) PyMem_New( PyObject*, af->N )) ){
						  int i, tu;
						  ascanf_Function *str;
							for( i= 0; strings && i< af->N; i++ ){
								str= parse_ascanf_address( af->array[i], 0, "Py_ImportVariableFromAscanf", 0, &tu );
								if( str && tu ){
									strings[i]= PyString_FromString( /*strdup*/(str->usage) );
								}
								else{
								  int j=i-1;
									for( ; j>= 0 ; j-- ){
										Py_XDECREF(strings[j]);
									}
									PyMem_Free(strings);
									strings= NULL;
								}
							}
							if( strings ){
								ret= PyArray_SimpleNewFromData( Ndims, dim, PyArray_OBJECT, (void*) strings);
								PyArray_ENABLEFLAGS( (PyArrayObject*)ret, NPY_OWNDATA );
							}
						}
						else{
							PyErr_NoMemory();
							return(NULL);
						}
						if( !ret ){
							if( array || (array= (double*) PyMem_New( double, af->N )) ){
								memcpy( array, af->array, af->N*sizeof(double) );
/* 								ret= PyArray_FromDimsAndData( Ndims, dim, PyArray_DOUBLE, (char*) array);	*/
								ret= PyArray_SimpleNewFromData( Ndims, dim, PyArray_DOUBLE, (void*) array);
								if( PIVC ){
									PyArray_ENABLEFLAGS( (PyArrayObject*)ret, NPY_OWNDATA );
								}
							}
							else{
								PyErr_NoMemory();
								return(NULL);
							}
						}
					}
					break;
				}
				case _ascanf_python_object:
					ret= af->PyObject;
					Py_XINCREF(ret);
					break;
				default:
					ret= Py_BuildValue( "d", af->value );
					break;
			}
		}
	}
	else{
		PyErr_SetString( XG_PythonError, "ascanf variable not found/defined" );
// 		PyErr_SetString( PyExc_LookupError, "ascanf variable not found/defined" );
	}
	return( ret );
}

static PyObject* python_ImportVariable(PyObject *self, PyObject *args, PyObject *kw )
{ int argc, Ndims= 0, deref=0, Nok= 0;
  npy_intp *dim = NULL;
  char *kws[]= { "variable", "dimensions", "dereference", NULL }, *name= NULL;
  PyObject *var= NULL, *dims= NULL, *ret= NULL;
  ascanf_Function *af= NULL;
#ifdef IS_PY3K
  PyObject *bytes = NULL;
#endif
	CHECK_INTERRUPTED();
	argc= PyTuple_Size(args);
	if( !PyArg_ParseTupleAndKeywords(args, kw, "O|Oi:ImportVariable", kws, &var, &dims, &deref ) ){
		return NULL;
	}

	PyErr_Clear();
	if( dims ){
	  int i;
		if( PyList_Check(dims) ){
			if( !(dims= PyList_AsTuple(dims)) ){
 				PyErr_SetString( XG_PythonError, "unexpected failure converting dimensions list to tuple" );
// 				PyErr_SetString(  PyExc_RuntimeError, "unexpected failure converting dimensions list to tuple" );
				return(NULL);
			}
		}
		if( PyTuple_Check(dims) ){
			Ndims= PyTuple_Size(dims);
			if( !(dim= (npy_intp*) calloc( Ndims, sizeof(npy_intp) )) ){
				PyErr_NoMemory();
				return(NULL);
			}
		}
		for( i= 0; i< Ndims; i++ ){
		  PyObject *arg= PyTuple_GetItem(dims, i);
			if( PyFloat_Check(arg) ){
				dim[i]= (npy_intp) PyFloat_AsDouble(arg);
			}
			else if( PyInt_Check(arg) || PyLong_Check(arg) ){
				dim[i]= (npy_intp) PyInt_AsLong(arg);
			}
			else{
 				PyErr_SetString( XG_PythonError, "dimension sizes should be integer" );
// 				PyErr_SetString(  PyExc_TypeError, "dimension sizes should be integer" );
				goto PIV_ESCAPE;
			}
		}
	}

	{ int i, N, multi;
	  PyObject *arg= NULL;
		if( PyList_Check(var) ){
			if( !(var= PyList_AsTuple(var)) ){
 				PyErr_SetString( XG_PythonError, "unexpected failure converting list to tuple" );
// 				PyErr_SetString(  PyExc_RuntimeError, "unexpected failure converting list to tuple" );
				goto PIV_ESCAPE;
			}
		}
		if( PyTuple_Check(var) ){
			N= PyTuple_Size(var);
			multi= 1;
			ret= PyTuple_New(N);
		}
		else{
			arg= var;
			N= 1;
			multi= 0;
		}
		for( i= 0; i< N; i++ ){
			if( multi ){
				arg= PyTuple_GetItem(var, i);
			}
			af= NULL;
#ifdef USE_COBJ
			if( PyCObject_Check(arg) ){
				if( !(af= PyCObject_AsVoidPtr(arg)) || (PyCObject_GetDesc(arg)!= af->function) ){
					af= NULL;
				}
			}
#else
			if( PyAscanfObject_Check(arg) ){
				af= PyAscanfObject_AsAscanfFunction(arg);
			}
#endif
#ifdef IS_PY3K
			else if( PyUnicode_Check(arg) ){
				PYUNIC_TOSTRING( arg, bytes, name );
				if( PyErr_Occurred() ){
					PyErr_Print();
					Py_XDECREF(bytes);
					return(NULL);
				}
			}
#endif
			else if( PyBytes_Check(arg) ){
				name= PyBytes_AsString(arg);
			}
			else{
			  double d= PyFloat_AsDouble(arg);
				if( PyErr_Occurred() ){
					PyErr_Print();
					return(NULL);
				}
				af= parse_ascanf_address( d, 0, "python_ImportVariable", (int) ascanf_verbose, NULL );
			}
			if( af || name ){
				if( multi ){
					PyTuple_SetItem(ret, i, Py_ImportVariableFromAscanf( &af, &name, Ndims, dim, deref, 0 ) );
				}
				else{
					ret= Py_ImportVariableFromAscanf( &af, &name, Ndims, dim, deref, 0 );
				}
				Nok+= 1;
			}
			if( ret ){
				if( PyErr_Occurred() ){
					PyErr_Print();
				}
				PyErr_Clear();
			}
		}
	}
PIV_ESCAPE:;
	if( !Nok ){
		Py_INCREF(var);
		ret= var;
	}
#ifdef IS_PY3K
	if( bytes ){
		Py_XDECREF(bytes);
	}
#endif
	xfree(dim);
	return( ret );
}

static PyObject* python_ImportVariableToModule(PyObject *self, PyObject *args, PyObject *kw )
{ int argc, Ndims= 0, deref=0, Nok= 0, is_module= False;
  npy_intp *dim = NULL;
  char *kws[]= { "module", "variable", "dimensions", "dereference", NULL }, *name= NULL, *mname=NULL;
  PyObject *module= NULL, *var= NULL, *dims= NULL, *ret= NULL;
  ascanf_Function *af= NULL;
#ifdef IS_PY3K
  struct PyModuleDef *moduleDef = NULL;
  PyObject *bytes = NULL;
#endif
	CHECK_INTERRUPTED();
	argc= PyTuple_Size(args);
	if( !PyArg_ParseTupleAndKeywords(args, kw, "OO|Oi:ImportVariable", kws, &module, &var, &dims, &deref ) ){
		return NULL;
	}

	PyErr_Clear();

	if( PyModule_Check(module) ){
		is_module= True;
	}
	else if( PyUnicode_Check(module) ){
#if 0
		if( !(module= PyBytes_AsDecodedObject( module, "utf-8", "ignore" )) ){
 			PyErr_SetString( XG_PythonError, "unexpected failure converting Unicode string from (inferred) utf-8" );
// 			PyErr_SetString(  PyExc_RuntimeError, "unexpected failure converting Unicode string from (inferred) utf-8" );
			return(NULL);
		}
#else
		if( !(module= _PyUnicode_AsDefaultEncodedString( module, NULL )) ){
 			PyErr_SetString( XG_PythonError, "unexpected failure converting Unicode string from default encoding" );
// 			PyErr_SetString(  PyExc_RuntimeError, "unexpected failure converting Unicode string from default encoding" );
			return(NULL);
		}
#endif
	}
	if( !is_module ){
		if( PyBytes_Check(module) ){
			mname= strdup(PyBytes_AsString(module));
		}
#ifdef IS_PY3K
		else if( PyUnicode_Check(module) ){
			PYUNIC_TOSTRING( module, bytes, mname );
		}
#endif
		if( mname ){
			module= NULL;
#ifdef IS_PY3K
			if( !(moduleDef = calloc(1, sizeof(struct PyModuleDef))) ){
				PyErr_SetString( XG_PythonError, "cannot allocate PyModuleDef" );
				PyErr_NoMemory();
				if( bytes ){
					Py_XDECREF(bytes);
				}
				return(NULL);
			}
#endif
		}
		else{
 			PyErr_SetString( XG_PythonError, "module argument should be a string or module object" );
// 			PyErr_SetString(  PyExc_TypeError, "module argument should be a string or module object" );
#ifdef IS_PY3K
			if( bytes ){
				Py_XDECREF(bytes);
			}
#endif
			return(NULL);
		}
	}

	if( dims ){
	  int i;
		if( PyList_Check(dims) ){
			if( !(dims= PyList_AsTuple(dims)) ){
 				PyErr_SetString( XG_PythonError, "unexpected failure converting dimensions list to tuple" );
// 				PyErr_SetString(  PyExc_RuntimeError, "unexpected failure converting dimensions list to tuple" );
#ifdef IS_PY3K
				if( bytes ){
					Py_XDECREF(bytes);
				}
#endif
				return(NULL);
			}
		}
		if( PyTuple_Check(dims) ){
			Ndims= PyTuple_Size(dims);
			if( !(dim= (npy_intp*) calloc( Ndims, sizeof(npy_intp) )) ){
				PyErr_NoMemory();
#ifdef IS_PY3K
				if( bytes ){
					Py_XDECREF(bytes);
				}
#endif
				return(NULL);
			}
		}
		for( i= 0; i< Ndims; i++ ){
		  PyObject *arg= PyTuple_GetItem(dims, i);
			if( PyFloat_Check(arg) ){
				dim[i]= (npy_intp) PyFloat_AsDouble(arg);
			}
			else if( PyInt_Check(arg) || PyLong_Check(arg) ){
				dim[i]= (npy_intp) PyInt_AsLong(arg);
			}
			else{
 				PyErr_SetString( XG_PythonError, "dimension sizes should be integer" );
// 				PyErr_SetString(  PyExc_TypeError, "dimension sizes should be integer" );
				goto PIVM_ESCAPE;
			}
		}
	}

	{ int i, N, multi;
	  PyObject *arg= NULL;
		if( PyList_Check(var) ){
			if( !(var= PyList_AsTuple(var)) ){
 				PyErr_SetString( XG_PythonError, "unexpected failure converting list to tuple" );
// 				PyErr_SetString(  PyExc_RuntimeError, "unexpected failure converting list to tuple" );
				goto PIVM_ESCAPE;
			}
		}
		if( PyTuple_Check(var) ){
			N= PyTuple_Size(var);
			multi= 1;
		}
		else{
			arg= var;
			N= 1;
			multi= 0;
		}
		for( i= 0; i< N; i++ ){
#ifdef IS_PY3K
		  PyObject *bytes2 = NULL;
#endif
			if( multi ){
				arg= PyTuple_GetItem(var, i);
			}
			af= NULL;
#ifdef USE_COBJ
			if( PyCObject_Check(arg) ){
				if( !(af= PyCObject_AsVoidPtr(arg)) || (PyCObject_GetDesc(arg)!= af->function) ){
					af= NULL;
				}
			}
#else
			if( PyAscanfObject_Check(arg) ){
				af= PyAscanfObject_AsAscanfFunction(arg);
			}
#endif
#ifdef IS_PY3K
			else if( PyUnicode_Check(arg) ){
				name = NULL;
				PYUNIC_TOSTRING( arg, bytes2, name );
			}
#endif
			else if( PyBytes_Check(arg) ){
				name= PyBytes_AsString(arg);
			}
			else{
			  double d= PyFloat_AsDouble(arg);
				if( PyErr_Occurred() ){
					PyErr_Print();
				}
				else{
					af= parse_ascanf_address( d, 0, "python_ImportVariableToModule", (int) ascanf_verbose, NULL );
				}
			}
			if( af || name ){
				if( (ret= Py_ImportVariableFromAscanf( &af, &name, Ndims, dim, deref, 0 )) && ret!= Py_None ){
				  char *nname= NULL;
					if( !module &&
#ifdef IS_PY3K
						!(module= Py3_InitModule( moduleDef, mname, NULL ))
#else
						!(module= Py_InitModule( mname, NULL ))
#endif
					){
					  char *err;
						if( PyErr_Occurred() ){
							PyErr_Print();
						}
						if( (err= concat( "cannot create module ", mname, NULL )) ){
 							PyErr_SetString( XG_PythonError, err );
// 							PyErr_SetString(  PyExc_RuntimeError, err );
							xfree(err);
						}
						goto PIVM_ESCAPE;
					}
					Py_XINCREF(ret);
					  // replace a number of special meaning characters from the ascanf names that are invalid in Python:
					nname= (af && af->name)? af->name : name;
					switch( nname[0] ){
						case '$':
							nname= concat( "D_", &nname[1], NULL );
							break;
						case '%':
							nname= concat( "i_", &nname[1], NULL );
							break;
						default:
							nname= NULL;
							break;
					}
					if( af && af->name ){
						PyModule_AddObject( module, (nname)? nname : af->name, ret );
					}
					else{
						PyModule_AddObject( module, (nname)? nname : name, ret );
					}
					xfree(nname);
					Nok+= 1;
				}
				else{
					if( PyErr_Occurred() ){
						PyErr_Print();
					}
					PyErr_Clear();
				}
			}
#ifdef IS_PY3K
			if( bytes2 ){
				Py_XDECREF(bytes2);
			}
#endif
		}
	}
PIVM_ESCAPE:;
	xfree(dim);
#ifdef IS_PY3K
	if( bytes ){
		Py_XDECREF(bytes);
	}
#endif
	if( Nok ){
		if( !is_module ){
		  char *cmd= concat( "import ", mname, NULL );
		  int pA= pythonActive;
			if( cmd ){
				PyErr_Clear();
				pythonActive+= 1;
				PyRun_SimpleString( cmd );
				pythonActive= pA;
			}
			xfree(cmd);
		}
		xfree(mname);
		return( module );
	}
	else{
		xfree(mname);
		Py_RETURN_NONE;
	}
}

// RJVB 20080924: returns a pointer to an allocated string containing the name of the specified Python object
// or NULL
char *PyObject_Name( PyObject *var )
{ PyObject *nobj = PyObject_GetAttrString(var, "__name__");
  char *c= NULL;
  int pA= pythonActive;
	if( PyErr_Occurred() ){
	  /* we are not really interested in failures of getting a "__name__" attribute */
		PyErr_Clear();
	}
	if( nobj ){
	  PyObject *pname= PyObject_Str(nobj);
		Py_XDECREF(nobj);
		if( pname ){
#ifdef IS_PY3K
		  PyObject *bytes = NULL;
		  char *c = NULL;
			if( PyUnicode_Check(pname) ){
				PYUNIC_TOSTRING( pname, bytes, c );
				if( c ){
					c = XGstrdup(c);
				}
				Py_XDECREF(bytes);
			}
			else
#endif
			if( PyBytes_Check(pname) ){
				c= XGstrdup( PyBytes_AsString(pname) );
			}
		}
	}
	 // 20081218: do NOT overwrite previously found name???
	if( !c ){
		pythonActive+= 1;
		if( (c= (char*) PyEval_GetFuncName(var)) ){
			c= XGstrdup( c );
		}
		pythonActive= pA;
	}
	if( PyErr_Occurred() ){
	  /* we are still not really interested in failures of getting a "__name__" attribute */
		PyErr_Clear();
	}
	return(c);
}

ascanf_Function *make_ascanf_python_object( ascanf_Function *af, PyObject *var, char *caller )
{
	if( af ){
		if( var ){
		    /* 20080916: the number of arguments of a function can be found via the co_argcount
			\ attribute of the function's func_code attribute...
			*/
#ifdef IS_PY3K
		  PyObject *cobj = PyObject_GetAttrString(var, "__code__");
#else
		  PyObject *cobj = PyObject_GetAttrString(var, "func_code");
#endif
		  PyObject *aobj = (cobj)? PyObject_GetAttrString(cobj, "co_argcount") : NULL;
			if( PyErr_Occurred() ){
			  /* we are not really interested in failures of getting these attributes */
				PyErr_Clear();
			}
			af->type= _ascanf_python_object;
			af->PyObject= var;
			Py_XINCREF(var);
			xfree(af->PyObject_Name);
			af->PyObject_Name= PyObject_Name(var);
			af->PythonHasReturnVar = 0;
			if( aobj ){
				af->Nargs= PyInt_AsLong( aobj );
			}
			else{
				if( ascanf_verbose ){
					fprintf( StdErr, " (can't determine nr. of arguments to python function \"%s\", allowing the current ascanf maximum for \"%s\") ",
						af->PyObject_Name, af->name
					);
				}
				af->Nargs= AMAXARGS;
			}
		}
		else{
			af->type= _ascanf_variable;
			Py_XDECREF( ((PyObject*)af->PyObject) );
			af->PyObject= NULL;
			xfree(af->PyObject_Name);
			af->Nargs= 0;
		}
		af->value= af->own_address= take_ascanf_address(af);
		if( af->accessHandler ){
		  int level= -1;
			AccessHandler( af, af->name, &level, NULL, (caller)? caller : "make_ascanf_python_object", &af->value );
		}
	}
	return( af );
}

static void EVTA_Delete_Variable(ascanf_Function *af)
{ char *aname;
	aname= XGstrdup(af->name);
	 /* 20090113: here, we only "delete" a variable to get rid of unwanted settings. So we cannot
	  \ call Delete_Internal_Variable( NULL, af ) which would remove the variable alltogether...
	  */
	Delete_Variable(af);
	af->type= _ascanf_novariable;
	  /* ascanf_address depends on type, so make sure it's re-calculated! */
	af->own_address= 0;
	  /* but do make sure we can get back the same entry in the list when re-creating the variable! */
	if( !af->name ){
		af->name= aname;
	}
	else if( !af->name[0] ){
		xfree(af->name);
		af->name= aname;
	}
	else{
		xfree(aname);
	}
}

#ifdef IS_PY3K
// Call PyObject_AsFileDescriptor and clear any Python errors without however returning
// a valid file descriptor if an error was raised.
int PyObject_AsFileDescriptor_Clear(PyObject *var)
{ int py_fd;
	PyErr_Clear();
	py_fd = PyObject_AsFileDescriptor(var);
	if( PyErr_Occurred() ){
		py_fd = -1;
		PyErr_Clear();
	}
	return py_fd;
}
#endif

int ExportVariableToAscanf( PyAscanfObject *pao, char* name, PyObject *var, int force, int IDict, int as_pobj, ascanf_Function **ret )
{ int take_usage= False, take_address= False, local_af= False;
  ascanf_Function *af;
  char *usg= NULL;
#ifdef IS_PY3K
  int py_fd = -1;
  FILE *py_fp;
#endif

	if( pao && PyAscanfObject_Check(pao) ){
		af= pao->af;
		name= af->name;
	}
	else if( ret && *ret ){
		af= *ret;
		name= af->name;
		take_address= af->is_address;
		take_usage= af->take_usage;
		  // 20100503: clearly the ret argument was foreseen as one allowing to pass in a local variable
		  // ascanf_Function or something of the sort, but we are currently never called with a *ret
		  // that points to a local variable. Instead, we can get called with a non-null *ret that is supposed
		  // to receive the return value of a Python call from Ascanf; that ascanf_Function should thus not be converted
		  // to an AscanfPythonObject!
		// local_af= True;
	}
	else{
		do{
			switch(*name){
				case '&':
					name++;
					take_address= True;
					break;
				case '`':
					take_usage= True;
					name++;
					break;
			}
		} while( *name && (*name== '&' || *name== '`') );

		af= Py_getNamedAscanfVariable(name);
		if( af && af->type== _ascanf_novariable && !force ){
		  /* better handle this as a completely new variable... */
			af= NULL;
		}
	}

	if( as_pobj ){
PEV_as_pobj:;
		if( af && af->type!= _ascanf_python_object && force && !local_af){
			usg= XGstrdup(af->usage);
			EVTA_Delete_Variable(af);
			af= NULL;
		}
		if( !af ){
		  char *expr= concat( (IDict)? "IDict[" : "",
		  		"DCL[", name, "]", (IDict)? "]" : "", NULL );
			if( expr ){
			  int n= 1;
			  double result;
				if( ascanf_verbose ){
					fprintf( StdErr, " #%s# ", expr ); fflush(StdErr);
				}
				fascanf2( &n, expr, &result, ',' );
				if( !(af= Py_getNamedAscanfVariable(name)) ){
 					PyErr_SetString( XG_PythonError, "unexpected failure creating new ascanf variable" );
// 					PyErr_SetString(  PyExc_RuntimeError, "unexpected failure creating new ascanf variable" );
					return(False);
				}
				af->type= _ascanf_python_object;
				af->usage= usg;
			}
			else{
				PyErr_NoMemory();
				return(False);
			}
		}
		else if( local_af ){
			af->type= _ascanf_python_object;
		}
		if( af->type== _ascanf_python_object ){
			af= make_ascanf_python_object( af, var, "python_ExportVariable" );
			  /* ahem */
			goto PEV_finish;
		}
		else{
			if( ascanf_verbose ){
				fprintf( StdErr, " (%s \"%s\" already exists) ", AscanfTypeName(af->type), af->name );
			}
 			PyErr_SetString( XG_PythonError, "a non-PyObject variable of that name already exists" );
// 			PyErr_SetString(  PyExc_NameError, "a non-PyObject variable of that name already exists" );
			return(False);
		}
	}

	if( PyUnicode_Check(var) ){
#if 0
		if( !(var= PyBytes_AsDecodedObject( var, "utf-8", "ignore" )) ){
 			PyErr_SetString( XG_PythonError, "unexpected failure converting Unicode string from (inferred) utf-8" );
// 			PyErr_SetString(  PyExc_RuntimeError, "unexpected failure converting Unicode string from (inferred) utf-8" );
			return(False);
		}
#else
		if( !(var= _PyUnicode_AsDefaultEncodedString( var, NULL )) ){
 			PyErr_SetString( XG_PythonError, "unexpected failure converting Unicode string from default encoding" );
// 			PyErr_SetString(  PyExc_RuntimeError, "unexpected failure converting Unicode string from default encoding" );
			return(False);
		}
#endif
	}

	if( PyList_Check(var) ){
		if( !(var= PyList_AsTuple(var)) ){
 			PyErr_SetString( XG_PythonError, "unexpected failure converting list to tuple" );
// 			PyErr_SetString(  PyExc_RuntimeError, "unexpected failure converting list to tuple" );
			return(False);
		}
	}

	  /* scalars and strings: */
	if( PyInt_Check(var) || PyLong_Check(var) || PyFloat_Check(var)
#ifndef IS_PY3K
		|| PyFile_Check(var)
#else
		|| (py_fd = PyObject_AsFileDescriptor_Clear(var)) >= 0
#endif
	){
	  double value;
		if( af && af->type!= _ascanf_variable && force && !local_af){
			usg= XGstrdup(af->usage);
			EVTA_Delete_Variable(af);
			af= NULL;
		}
		if( !af ){
		  char *expr= concat( (IDict)? "IDict[" : "",
		  		"DCL[", name, "]", (IDict)? "]" : "", NULL );
			if( expr ){
			  int n= 1;
			  double result;
				if( ascanf_verbose ){
					fprintf( StdErr, " #%s# ", expr ); fflush(StdErr);
				}
				fascanf2( &n, expr, &result, ',' );
				if( !(af= Py_getNamedAscanfVariable(name)) ){
 					PyErr_SetString( XG_PythonError, "unexpected failure creating new ascanf variable" );
// 					PyErr_SetString(  PyExc_RuntimeError, "unexpected failure creating new ascanf variable" );
					return(False);
				}
				af->usage= usg;
			}
			else{
				PyErr_NoMemory();
				return(False);
			}
		}
		else if( local_af ){
			af= make_ascanf_python_object( af, NULL, "python_ExportVariable" );
			af->type= _ascanf_variable;
		}
		if( af->type== _ascanf_variable ){
			if(
#ifndef IS_PY3K
				PyFile_Check(var)
#else
				py_fd >= 0 && !PyLong_Check(var)
#endif
			){
#ifdef IS_PY3K
				if( !(py_fp = get_FILEForDescriptor(py_fd)) ){
					PyErr_SetString( XG_PythonError, "Cannot export io objects without a known associated FILE (i.e. opened via Python)" );
					return False;
				}
#endif
				set_NaN(af->value);
				xfree(af->usage);
#ifndef IS_PY3K
				af->usage= XGstrdup( PyBytes_AsString(PyFile_Name(var)) );
				af->fp= PyFile_AsFile(var);
#else
				af->fp = py_fp;
#endif
				PyErr_Warn( PyExc_Warning, "assuming that this is an open file, not pipe\0" );
				af->fp_is_pipe= False;
			}
			else{
				value= PyFloat_AsDouble(var);
				af->value= value;
				af->is_address= af->is_usage= 0;
			}
			if( af->accessHandler ){
			  int level= -1;
				AccessHandler( af, af->name, &level, NULL, "python_ExportVariable", &value );
			}
		}
		else{
 			PyErr_SetString( XG_PythonError, "type clash: scalar object can't be assigned to non-scalar" );
// 			PyErr_SetString(  PyExc_TypeError, "type clash: scalar object can't be assigned to non-scalar" );
			return(False);
		}
	}
	else if( PyBytes_Check(var)
#ifdef IS_PY3K
		|| PyUnicode_Check(var)
#endif
	){
#ifdef IS_PY3K
	  PyObject *bytes = NULL;
#endif
		if( af && af->type!= _ascanf_variable && force && !local_af ){
			EVTA_Delete_Variable(af);
			af= NULL;
		}
		else if( local_af ){
			af= make_ascanf_python_object( af, NULL, "python_ExportVariable" );
			af->type= _ascanf_variable;
		}
		if( !af || af->type== _ascanf_variable ){
		  char *c= NULL, *expr= NULL;
#ifdef IS_PY3K
			if( PyUnicode_Check(var) ){
				PYUNIC_TOSTRING( var, bytes, c );
				if( c ){
					c = parse_codes(c);
				}
			}
			else
#endif
			{
				c = parse_codes(PyBytes_AsString(var));
			}
			if( af ){
				xfree(af->usage);
				af->usage= strdup(c);
			}
			else{
				expr= concat( (IDict)? "IDict[" : "", "DCL[`", name, ",\"", c, "\"]", (IDict)? "]" : "", NULL );
			}
#ifdef IS_PY3K
			if( bytes ){
				Py_XDECREF(bytes);
			}
#endif
			if( expr ){
			  int n= 1;
			  double result;
				  // 20090413: replace the toplevel call into fascanf2() by call to Create_Internal_ascanfString()
				  // or similar which *ought* to be more efficient...!
				if( ascanf_verbose ){
					fprintf( StdErr, " #%s# ", expr ); fflush(StdErr);
				}
				fascanf2( &n, expr, &result, ',' );
				xfree(expr);
				if( n< 1 ){
 					PyErr_SetString( XG_PythonError, "unexpected failure creating new ascanf variable" );
// 					PyErr_SetString(  PyExc_RuntimeError, "unexpected failure creating new ascanf variable" );
					return(False);
				}
				if( !(af= Py_getNamedAscanfVariable(name)) ){
 					PyErr_SetString( XG_PythonError, "unexpected failure creating new ascanf variable" );
// 					PyErr_SetString(  PyExc_RuntimeError, "unexpected failure creating new ascanf variable" );
					return(False);
				}
			}
			else{
				PyErr_NoMemory();
				return(False);
			}
			af->is_usage= af->take_usage= 1;
			xfree(expr);
		}
		else{
 			PyErr_SetString( XG_PythonError, "type clash: string object can't be assigned to non-scalar" );
// 			PyErr_SetString(  PyExc_TypeError, "type clash: string object can't be assigned to non-scalar" );
			return(False);
		}
	}
	  /* non-scalars: */
	else if( PyTuple_Check(var) || PyComplex_Check(var) || PyArray_Check(var) ){
PEV_tuple:;
	  int i, N, type;
		if( PyTuple_Check(var) ){
			N= PyTuple_Size(var);
			for( i= 0; i< N; i++ ){
			  PyObject *el= PyTuple_GetItem(var, i);
				if( !(el && (PyInt_Check(el) || PyLong_Check(el) || PyFloat_Check(el))) ){
 					PyErr_SetString( XG_PythonError, "type clash: only tuples with scalar, numeric elements are supported" );
// 					PyErr_SetString(  PyExc_TypeError, "type clash: only tuples with scalar, numeric elements are supported" );
					return(False);
				}
			}
			type= 0;
		}
		else if( PyComplex_Check(var) ){
			N= 2;
			type= 1;
		}
		else if( PyArray_Check(var) ){
			N= PyArray_Size(var);
			type= 2;
		}
		if( af && af->type!= _ascanf_array && force && !local_af ){
			usg= XGstrdup(af->usage);
			EVTA_Delete_Variable(af);
			af= NULL;
		}
		if( !af ){
		  char Nitems[64], *expr;
			snprintf( Nitems, 64, "%d", N );
			expr= concat( (IDict)? "IDict[" : "",
		  		"DCL[", name, ",", Nitems, ",0]", (IDict)? "]" : "", NULL );
			if( expr ){
			  int n= 1;
			  double result;
				if( ascanf_verbose ){
					fprintf( StdErr, " #%s# ", expr );
				}
				fascanf2( &n, expr, &result, ',' );
				if( !(af= Py_getNamedAscanfVariable(name)) ){
 					PyErr_SetString( XG_PythonError, "unexpected failure creating new ascanf array" );
// 					PyErr_SetString(  PyExc_RuntimeError, "unexpected failure creating new ascanf array" );
					return(False);
				}
				af->usage= usg;
			}
			else{
				PyErr_NoMemory();
				return(False);
			}
		}
		else if( local_af ){
			af= make_ascanf_python_object( af, NULL, "python_ExportVariable" );
			if( !af->array && !af->iarray ){
				  /* let Resize_Ascanf_Array take care of initialising an _ascanf_array according to the rules: */
				af->type= _ascanf_variable;
				af->N= 0;
			}
			else{
				af->type= _ascanf_array;
			}
		}
		if( af->type== _ascanf_array || af->type== _ascanf_variable ){
		  double value;
		  PyArrayObject* xd= NULL;
		  double *PyArrayBuf= NULL;
		  PyArrayIterObject *it;
			if( type== 2 ){
				  /* 20061022: used to use PyArray_CopyFromObject, which appears to be up to 37% slower */
				if( (xd = (PyArrayObject*) PyArray_ContiguousFromObject( (PyObject*) var, PyArray_DOUBLE, 0, 0 )) ){
					PyArrayBuf= (double*)PyArray_DATA(xd); /* size would be N*sizeof(double) */
				}
				else{
					PyErr_Clear();
					it= (PyArrayIterObject*) PyArray_IterNew(var);
				}
			}
			if( af->N!= N ){
				Resize_ascanf_Array( af, N, &value );
			}
			if( PyArrayBuf && type == 2 ){
				// 20131004: removed this case to outside the big loop-with-tests to allow it
				// to be optimised
				if( af->array ){
					if( af->array != PyArrayBuf ){
						memcpy( af->array, PyArrayBuf, N * sizeof(double) );
					}
				}
				else{
					for( i= 0; i< N; i++ ){
						af->iarray[i]= (int) PyArrayBuf[i];
					}
				}
			}
			else{
				for( i= 0; i< N; i++ ){
					switch( type ){
						case 0:
							value= PyFloat_AsDouble( PyTuple_GetItem(var,i) );
							break;
						case 1:
							value= (i)? PyComplex_ImagAsDouble(var) : PyComplex_RealAsDouble(var);
							break;
						case 2:{
						  PyArrayObject *parray= (PyArrayObject*) var;
							if( it->index < it->size ){
							  PyObject *elem= PyArray_DESCR(parray)->f->getitem( it->dataptr, var);
								if( PyInt_Check(elem) || PyLong_Check(elem) || PyFloat_Check(elem) ){
									value= PyFloat_AsDouble(elem);
								}
								else if( PyUnicode_Check(elem) ){
									if( !(elem= _PyUnicode_AsDefaultEncodedString( elem, NULL )) ){
										PyErr_SetString( XG_PythonError, "unexpected failure converting Unicode string from default encoding" );
										PyErr_Print();
									}
									else{
										goto store_aarray_string_elem;
									}
								}
								else if( PyBytes_Check(elem)
#ifdef IS_PY3K
									|| PyUnicode_Check(elem)
#endif
								){
store_aarray_string_elem:;
								  int level= -2;
								  ascanf_Function *ef= NULL;
#ifdef IS_PY3K
									if( PyUnicode_Check(elem) ){
									  PyObject *bytes = NULL;
									  char *str = NULL;
										PYUNIC_TOSTRING( elem, bytes, str );
										if( str ){
											ef = Create_Internal_ascanfString( str, &level );
										}
										if( bytes ){
											Py_XDECREF(bytes);
										}
									}
									else
#endif
									{
										ef = Create_Internal_ascanfString( PyBytes_AsString(elem), &level );
									}
									if( ef ){
										ef->links+= 1;
										ef->is_usage= True;
										ef->take_usage= True;
										value= take_ascanf_address(ef);
									}
								}
								else{
									PyErr_Warn( PyExc_Warning, "type clash: only arrays with scalar, numeric elements are supported" );
									PyErr_Print();
									set_NaN(value);
								}
								PyArray_ITER_NEXT(it);
							}
							else{
								set_NaN(value);
							}
							break;
						}
						default:
							break;
					}
					if( af->iarray ){
						af->iarray[i]= (int) value;
					}
					else{
						af->array[i]= value;
					}
				}
			}
			if( type==2 ){
				if( xd ){
					Py_XDECREF(xd);
				}
				else{
					Py_DECREF(it);
				}
			}
			af->is_address= af->is_usage= 0;
			af->last_index= 0;
			af->value= ASCANF_ARRAY_ELEM(af,0);
			if( af->accessHandler ){
			  int level= -1;
				AccessHandler( af, af->name, &level, NULL, "python_ExportVariable", &value );
			}
		}
		else{
 			PyErr_SetString( XG_PythonError, "type clash: non-scalar object can't be assigned to scalar" );
// 			PyErr_SetString(  PyExc_TypeError, "type clash: non-scalar object can't be assigned to scalar" );
			return(False);
		}
	}
	else if( PySequence_Check(var) ){
		if( !(var= PySequence_Fast(var, "attempt to convert non-sequence object to a tuple")) ){
 			PyErr_SetString( XG_PythonError, "unexpected failure converting sequence to tuple" );
// 			PyErr_SetString(  PyExc_RuntimeError, "unexpected failure converting sequence to tuple" );
			return(False);
		}
		if( PyList_Check(var) ){
			if( !(var= PyList_AsTuple(var)) ){
 				PyErr_SetString( XG_PythonError, "unexpected failure converting list to tuple" );
// 				PyErr_SetString(  PyExc_RuntimeError, "unexpected failure converting list to tuple" );
				return(False);
			}
		}
		if( PyTuple_Check(var) ){
			goto PEV_tuple;
		}
		else{
			goto PEV_escape;
		}
	}

	else{
		PyErr_Warn( PyExc_Warning, "type which cannot be \"translated\" into a corresponding ascanf type, exporting as Python object" );
		goto PEV_as_pobj;
PEV_escape:;
// kludgy: we never come here other than through a jump:
 		PyErr_SetString( XG_PythonError, "unsupported object type" );
// 		PyErr_SetString(  PyExc_TypeError, "unsupported object type" );
		return(False);
	}
PEV_finish:;
	if( ret ){
		if( af && !af->own_address ){
			take_ascanf_address(af);
		}
		*ret= af;
	}
	return( True );
}

PyObject *Py_ExportVariableToAscanf( PyAscanfObject *pao, char* name, PyObject *var, int force, int IDict, int as_pobj, char *label, int retVar )
{ ascanf_Function *af= NULL;
  char *curLabel;
	if( ascanf_VarLabel ){
		curLabel = ascanf_VarLabel->usage;
		if( label ){
			ascanf_VarLabel->usage = strdup(label);
			label = ascanf_VarLabel->usage;
		}
	}
	if( ExportVariableToAscanf( pao, name, var, force, IDict, as_pobj, &af ) ){
		if( ascanf_VarLabel ){
			if( ascanf_VarLabel->usage == label ){
				// 20110327: clever ... we didn't initialise curLabel for the case label==NULL...
				xfree(ascanf_VarLabel->usage);
				ascanf_VarLabel->usage = curLabel;
			}
			else{
				// something unforeseen happened: ascanf_VarLabel changed during the call to ExportVariableToAscanf().
				// Therefore, we assume that the <label> copy has been deallocated, and we'll have to deallocate
				// the cached pointer to the previous string:
				xfree(curLabel);
				label = NULL;
			}
		}
		if( af  ){
			af->PythonHasReturnVar = retVar;
// 20120413: why?!
// 			if( af->Nargs >= 0 ){
// 				af->Nargs += 1;
// 			}
#if 0
// specifying a return variable while exporting a Python object is already something, but not flexible enough...
			if( PyAscanfObject_Check(retVar) ){
				if( (af->PyObject_ReturnVar = ((PyAscanfObject*)retVar)->af) ){
					af->PythonHasReturnVar = True;
				}
			}
			else if( PyUnicode_Check(retVar) ){
				if( !(retVar= _PyUnicode_AsDefaultEncodedString( retVar, NULL )) ){
					PyErr_SetString( XG_PythonError, "unexpected failure converting Unicode string to default encoding" );
				}
			}
			if( PyBytes_Check(retVar)
#ifdef IS_PY3K
				|| PyUnicode_Check(retVar)
#endif
			){
#ifdef IS_PY3K
				if( PyUnicode_Check(retVar) ){
				  PyObject *bytes = NULL;
				  char *str = NULL;
					PYUNIC_TOSTRING( retVar, bytes, str );
					if( str ){
						af->PyObject_ReturnVar = Py_getNamedAscanfVariable(str);
					}
					else{
						af->PyObject_ReturnVar = NULL;
					}
					if( bytes ){
						Py_XDECREF(bytes);
					}
				}
				else
#endif
				{
					af->PyObject_ReturnVar = Py_getNamedAscanfVariable( PyBytes_AsString(retVar) );
				}
				if( af->PyObject_ReturnVar ){
					af->PythonHasReturnVar = True;
				}
			}
			else{
				PyErr_SetString( XG_PythonError, "returnVar argument must be a PyAscanfObject or string" );
				af->PyObject_ReturnVar = NULL;
				af->PythonHasReturnVar = False;
			}
#endif
		}
		if( pao ){
		  /* Return the object we were called with. The af it points to will have been modified. */
			return( (PyObject*) pao);
		}
		else{
			if( af ){
/* 				return( Py_BuildValue( "d", af->own_address ) );	*/
				return( Py_BuildValue( "O", PyAscanfObject_FromAscanfFunction(af) ) );
			}
			else{
				Py_RETURN_NONE;
			}
		}
	}
	else{
		if( ascanf_VarLabel ){
			if( ascanf_VarLabel->usage==label ){
				xfree(ascanf_VarLabel->usage);
				ascanf_VarLabel->usage = curLabel;
			}
			else{
				// something unforeseen happened: ascanf_VarLabel changed during the call to ExportVariableToAscanf().
				// Therefore, we assume that the <label> copy has been deallocated, and we'll have to deallocate
				// the cached pointer to the previous string:
				xfree(curLabel);
				label = NULL;
			}
		}
		return( NULL );
	}
}

static PyObject* python_ExportVariable(PyObject *self, PyObject *args, PyObject *kw )
{ int IDict= -1, force= False, as_po= False, retVar= False;
  char *kws[]= { "name", "variable", "replace", "IDict", "as_PObj", "label", "returnVar", NULL };
  PyObject *pname, *pvar, *plabel=NULL;
  char *label = NULL, *name = NULL;
  PyObject *ret = NULL;
#ifdef IS_PY3K
  PyObject *nBytes = NULL, *lBytes = NULL;
#endif
	CHECK_INTERRUPTED();

	if( !PyArg_ParseTupleAndKeywords(args, kw, "OO|iiiOi:ExportVariable", kws, &pname, &pvar, &force, &IDict, &as_po, &plabel, &retVar ) ){
		return NULL;
	}

	if( plabel ){
		if( PyUnicode_Check(plabel) ){
#ifdef IS_PY3K
			PYUNIC_TOSTRING( plabel, lBytes, label );
#else
			if( !(plabel= _PyUnicode_AsDefaultEncodedString( plabel, NULL )) ){
				PyErr_Warn( PyExc_Warning, "unexpected failure converting Unicode label string to default encoding" );
				plabel = NULL;
			}
#endif
		}
		if( plabel && PyBytes_Check(plabel) ){
			label = PyBytes_AsString(plabel);
		}
		else{
			plabel = NULL;
		}
	}
	if( PyUnicode_Check(pname) ){
#ifdef IS_PY3K
		PYUNIC_TOSTRING( pname, nBytes, name );
#else
		if( !(pname= _PyUnicode_AsDefaultEncodedString( pname, NULL )) ){
 			PyErr_SetString( XG_PythonError, "unexpected failure converting Unicode string to default encoding" );
// 			PyErr_SetString(  PyExc_RuntimeError, "unexpected failure converting Unicode string to default encoding" );
			Py_RETURN_NONE;
		}
#endif
	}
	if( name || (pname && PyBytes_Check(pname)) ){
		if( IDict== -1 ){
			IDict= False;
		}
		if( name ){
			ret = Py_ExportVariableToAscanf( NULL, name, pvar, force, IDict, as_po, label, retVar );
		}
		else{
			ret = Py_ExportVariableToAscanf( NULL, PyBytes_AsString(pname), pvar, force, IDict, as_po, label, retVar );
		}
	}
	else if( PyAscanfObject_Check(pname) ){
		if( IDict== -1 ){
			IDict= ((PyAscanfObject*)pname)->af->internal;
		}
		ret = Py_ExportVariableToAscanf( (PyAscanfObject*) pname, NULL, pvar, force, IDict, as_po, label, retVar );
	}
	else{
	  int i, N;
		if( IDict== -1 ){
			IDict= False;
		}
		if( PyList_Check(pname) ){
			if( !(pname= PyList_AsTuple(pname)) ){
 				PyErr_SetString( XG_PythonError, "unexpected failure converting names list to tuple" );
// 				PyErr_SetString( PyExc_RuntimeError, "unexpected failure converting names list to tuple" );
				Py_RETURN_NONE;
			}
		}
		if( PyList_Check(pvar) ){
			if( !(pvar= PyList_AsTuple(pvar)) ){
 				PyErr_SetString( XG_PythonError, "unexpected failure converting variables list to tuple" );
// 				PyErr_SetString( PyExc_RuntimeError, "unexpected failure converting variables list to tuple" );
				Py_RETURN_NONE;
			}
		}
		if( (N= PyTuple_Size(pname)) != PyTuple_Size(pvar) ){
 			PyErr_SetString( XG_PythonError, "names and variables lists (tuples) must be of equal length" );
// 			PyErr_SetString( PyExc_AttributeError, "names and variables lists (tuples) must be of equal length" );
			PyErr_Print();
			Py_RETURN_NONE;
		}
		if( !(ret= PyTuple_New(N)) ){
			if( PyErr_Occurred() ){
				PyErr_Print();
			}
			Py_RETURN_NONE;
		}
		for( i= 0; i< N; i++ ){
		  PyObject *ppname= PyTuple_GetItem(pname, i);
			if( PyUnicode_Check(ppname) ){
#if 0
				if( !(ppname= PyBytes_AsDecodedObject( ppname, "utf-8", "ignore" )) ){
 					PyErr_SetString( XG_PythonError, "unexpected failure converting Unicode name string from (inferred) utf-8" );
// 					PyErr_SetString( PyExc_RuntimeError, "unexpected failure converting Unicode name string from (inferred) utf-8" );
					PyErr_Print();
				}
#else
				if( !(ppname= _PyUnicode_AsDefaultEncodedString( ppname, NULL )) ){
					PyErr_SetString( XG_PythonError, "unexpected failure converting Unicode string from default encoding" );
// 					PyErr_SetString(  PyExc_RuntimeError, "unexpected failure converting Unicode string from default encoding" );
					PyErr_Print();
				}
#endif
			}
			if( PyBytes_Check(ppname) ){
				PyTuple_SetItem( ret, i,
					Py_ExportVariableToAscanf( NULL, PyBytes_AsString(ppname), PyTuple_GetItem(pvar,i), force, IDict, as_po, label, retVar )
				);
			}
			else{
				PyErr_Warn( PyExc_Warning, "names list may only contain strings - ignoring invalid name/variable items" );
			}
		}
	}
#ifdef IS_PY3K
	if( lBytes ){
		Py_XDECREF(lBytes);
	}
	if( nBytes ){
		Py_XDECREF(nBytes);
	}
#endif
	return ret;
}

static PyObject* python_TBARprogress ( PyObject *self, PyObject *args, PyObject *kw )
{ int argc;
  char *kws[]= { "current", "final", "step", NULL };
  double current, final, step= 10;
	CHECK_INTERRUPTED();

	argc= PyTuple_Size(args);
	if( !PyArg_ParseTupleAndKeywords(args, kw, "dd|d:TBARprogress", kws, &current, &final, &step ) ){
		return NULL;
	}

	{ int len= 0;
	  char *h= TBARprogress_header, *h2= TBARprogress_header2, *sep;
	  static char *TBbuf= NULL;
	  static int n= -1, tbuf_len= 0;
	  static double pc, pf, ps;
	  double perc;
		CLIP_EXPR( step, step, 0.1, 100 );
		perc= current* 100/ final;
		if( n== -1 || pf!= final || current< pc || step!= ps ){
			n= 0;
		}
		if( !TBARprogress_header ){
			h= "";
			sep= (h2)? ": " : "";
		}
		else{
			sep= ": ";
		}
		len= 130+ strlen(h)+ ((h2)? strlen(h2) : 0)+ 2* strlen(sep);
		if( len> tbuf_len ){
			TBbuf= XGrealloc( TBbuf, (tbuf_len= len)* sizeof(char) );
		}
		if( TBbuf ){
			if( perc>= n* step ){
				n+= 1;
				len= sprintf( TBbuf, "%s%s%s%s%s%% (%s of %s)",
					h, sep, (h2)? h2 : "", (h2)? sep : "",
					ad2str( perc, d3str_format, NULL),
					ad2str( current, d3str_format, NULL),
					ad2str( final, d3str_format, NULL)
				);
				StringCheck( TBbuf, tbuf_len, __FILE__, __LINE__ );
				if( ascanf_window ){
					XStoreName( disp, ascanf_window, TBbuf );
					if( !RemoteConnection ){
						XFlush( disp );
					}
				}
				else{
					fprintf( StdErr, "\033]0;%s\007", TBbuf );
					fflush( StdErr );
				}
				if( pragma_unlikely(ascanf_verbose) ){
					fprintf( StdErr, "\"%s\" == ", TBbuf );
				}
			}
		}
		else{
			fprintf( StdErr, "TBAR_process[%s,%s..]: can't get memory (%s)\n",
				ad2str( current, d3str_format, NULL),
				ad2str( final, d3str_format, NULL), serror()
			);
			tbuf_len= 0;
			set_NaN(perc);
			PyErr_NoMemory();
			return(NULL);
		}
		pc= current;
		pf= final;
		ps= step;
		return( Py_BuildValue("d", perc) );
	}
}

PyObject *python_SetNumber ( PyObject *self, PyObject *args, PyObject *kw )
{ char *kws[]= { "current", NULL };
  int current= 0;
	CHECK_INTERRUPTED();

	if( !PyArg_ParseTupleAndKeywords(args, kw, "|i:SetNumber", kws, &current ) ){
		return NULL;
	}

	if( current ){
		return( Py_BuildValue( "i", *ascanf_setNumber ) );
	}
	return( Py_BuildValue( "i", setNumber ) );
}

PyObject *python_NumPoints ( PyObject *self, PyObject *args, PyObject *kw )
{ int argc;
  char *kws[]= { "set", "numPoints", NULL };
  int idx= 0, N= -1, result;
	CHECK_INTERRUPTED();

	argc= PyTuple_Size(args);
	if( !PyArg_ParseTupleAndKeywords(args, kw, "|ii:NumPoints", kws, &idx, &N ) ){
		return NULL;
	}

	if( !argc ){
	  int i;
		result= 0;
		  /* 20030929: I don't think user's ascanf code will be very interested in maxitems, but rather in this: */
		for( i= 0; i< setNumber; i++ ){
			if( AllSets[i].numPoints> result ){
				result= AllSets[i].numPoints;
			}
		}
		return( Py_BuildValue( "i", result ) );
	}
	else{
		if( idx>= 0 && idx< MaxSets && AllSets ){
			if( argc> 1 ){
				if( N>= 0 && N<= MAXINT ){
					if( N!= AllSets[idx].numPoints ){
					  int np= AllSets[idx].numPoints, i, j;
						if( (AllSets[idx].numPoints= N) ){
							realloc_points( &AllSets[idx], N, False );
						}
						else{
							  /* 20010901: RJVB: set the skipOnce flag for this set when we set
							   \ numPoints to 0. The *DATA_...* processing chain for the current
							   \ point ($Counter) will be finished, but everything afterwards
							   \ in the data-processing loop (in DrawData()) should not be done
							   \ (especially not setting array elements array[this_set->numPoints-1] ...
							   */
							AllSets[idx].skipOnce= True;
						}
						for( i= np; i< AllSets[idx].numPoints; i++ ){
							for( j= 0; j< AllSets[idx].ncols; j++ ){
								set_NaN( AllSets[idx].columns[j][i] );
							}
							if( ActiveWin && ActiveWin!= StubWindow_ptr ){
								if( ActiveWin->curve_len ){
									ActiveWin->curve_len[idx][i]= ActiveWin->curve_len[idx][np];
								}
								if( ActiveWin->error_len ){
									ActiveWin->error_len[idx][i]= ActiveWin->error_len[idx][np];
								}
							}
						}
						if( AllSets[idx].numPoints> maxitems ){
							maxitems= AllSets[idx].numPoints;
							realloc_Xsegments();
						}
						if( !ActiveWin || !ActiveWin->drawing ){
							RedrawSet( idx, False );
						}
					}
				}
			}
			return( Py_BuildValue( "i", AllSets[idx].numPoints ) );
		}
		else{
 			PyErr_SetString( XG_PythonError, "invalid setNumber" );
// 			PyErr_SetString( PyExc_ValueError, "invalid setNumber" );
			return(NULL);
		}
	}
}

PyObject *python_ncols ( PyObject *self, PyObject *args, PyObject *kw )
{ int argc;
  char *kws[]= { "set", "columns", NULL };
  int idx= (int) *ascanf_setNumber, N= -1;
	CHECK_INTERRUPTED();

	argc= PyTuple_Size(args);
	if( !PyArg_ParseTupleAndKeywords(args, kw, "|ii:ncols", kws, &idx, &N ) ){
		return NULL;
	}

	if( idx>= 0 && idx<= setNumber && AllSets ){
		if( argc>= 2 ){
			if( N> 0 && AllSets[idx].ncols!= N ){
				if( AllSets[idx].numPoints ){
					AllSets[idx].columns= realloc_columns( &AllSets[idx], N );
					Check_Columns( &AllSets[idx] );
				}
				else{
					AllSets[idx].ncols= N;
				}
			}
		}
		N= AllSets[idx].ncols;
	}
	else{
 		PyErr_SetString( XG_PythonError, "invalid setNumber" );
// 		PyErr_SetString( PyExc_ValueError, "invalid setNumber" );
		return(NULL);
	}
	return( Py_BuildValue( "i", N ) );
}

#define PYTHON_SETCOLUMN(self,args,kw,COLUMN) { int argc; \
  char *kws[]= { "set", "column", NULL }; \
  int idx= (int) *ascanf_setNumber, N= -1; \
	CHECK_INTERRUPTED(); \
 \
	argc= PyTuple_Size(args); \
	if( !PyArg_ParseTupleAndKeywords(args, kw, "|ii:"STRING(COLUMN), kws, &idx, &N ) ){ \
		return NULL; \
	} \
 \
	if( ActiveWin && ActiveWin!= StubWindow_ptr ){ \
		if( idx>= 0 && idx< setNumber && AllSets ){ \
			if( argc>= 2 ){ \
				CLIP_EXPR( ActiveWin->COLUMN[idx], N, 0, AllSets[idx].ncols-1); \
				if( AllSets[idx].COLUMN!= ActiveWin->COLUMN[idx] ){ \
					AllSets[idx].COLUMN= ActiveWin->COLUMN[idx]; \
					AllSets[idx].init_pass= True; \
				} \
			} \
			N= ActiveWin->COLUMN[idx]; \
		} \
		else{ \
			if( argc>= 2 && AllSets ){ \
				for( idx= 0; idx< setNumber; idx++ ){ \
					CLIP_EXPR( ActiveWin->COLUMN[idx], N, 0, AllSets[idx].ncols-1); \
					if( AllSets[idx].COLUMN!= ActiveWin->COLUMN[idx] ){ \
						AllSets[idx].COLUMN= ActiveWin->COLUMN[idx]; \
						AllSets[idx].init_pass= True; \
					} \
				} \
			} \
			else{ \
				N= 0; \
			} \
		} \
	} \
	else{ \
		if( idx>= 0 && idx< setNumber && AllSets ){ \
			if( argc>= 2 ){ \
			  int nc; \
				CLIP_EXPR( nc, N, 0, AllSets[idx].ncols-1); \
				if( AllSets[idx].COLUMN!= nc ){ \
					AllSets[idx].COLUMN= nc; \
					AllSets[idx].init_pass= True; \
				} \
			} \
			N= AllSets[idx].COLUMN; \
		} \
		else{ \
			PyErr_SetString( XG_PythonError /*PyExc_ValueError*/, "invalid setNumber" ); \
			return(NULL); \
		} \
	} \
	return( Py_BuildValue( "i", N ) ); \
}

PyObject *python_xcol ( PyObject *self, PyObject *args, PyObject *kw )
{
	PYTHON_SETCOLUMN( self, args, kw, xcol );
}

PyObject *python_ycol ( PyObject *self, PyObject *args, PyObject *kw )
{
	PYTHON_SETCOLUMN( self, args, kw, ycol );
}

PyObject *python_ecol ( PyObject *self, PyObject *args, PyObject *kw )
{
	PYTHON_SETCOLUMN( self, args, kw, ecol );
}

PyObject *python_lcol ( PyObject *self, PyObject *args, PyObject *kw )
{
	PYTHON_SETCOLUMN( self, args, kw, lcol );
}

PyObject *python_Ncol ( PyObject *self, PyObject *args, PyObject *kw )
{ int argc;
  char *kws[]= { "set", "column", NULL };
  int idx= (int) *ascanf_setNumber, N= -1;
	CHECK_INTERRUPTED();

	argc= PyTuple_Size(args);
	if( !PyArg_ParseTupleAndKeywords(args, kw, "|ii:Ncol", kws, &idx, &N ) ){
		return NULL;
	}

	if( idx>= 0 && idx< setNumber && AllSets ){
		if( argc>= 2 ){
		  int nc;
			CLIP_EXPR( nc, N, 0, AllSets[idx].ncols-1);
			if( AllSets[idx].Ncol!= nc ){
				AllSets[idx].Ncol= nc;
				AllSets[idx].init_pass= True;
			}
		}
		N= AllSets[idx].Ncol;
	}
	else{
 		PyErr_SetString( XG_PythonError, "invalid setNumber" );
// 		PyErr_SetString( PyExc_ValueError, "invalid setNumber" );
		return(NULL);
	}
	return( Py_BuildValue( "i", N ) );
}

PyObject *python_DataVal ( PyObject *self, PyObject *args, PyObject *kw )
{ int argc, set, col, idx;
  char *kws[]= { "set", "column", "index", "value", NULL };
  double nvalue;
  PyObject *ret;
	CHECK_INTERRUPTED();

	argc= PyTuple_Size(args);
	if( !PyArg_ParseTupleAndKeywords(args, kw, "iii|d:DataVal", kws, &set, &col, &idx, &nvalue ) ){
		return NULL;
	}

	if( AllSets ){
		CLIP_EXPR(set, set, 0, setNumber- 1 );
		CLIP_EXPR(col, col, 0, AllSets[set].ncols- 1 );
		CLIP_EXPR(idx, idx, 0, AllSets[set].numPoints- 1 );
		if( argc> 3 ){
			ret= Py_BuildValue( "d", (AllSets[set].columns[col][idx]= nvalue) );
		}
		else{
			ret= Py_BuildValue( "d", AllSets[set].columns[col][idx] );
		}
	}
	else{
		PyErr_SetString( XG_PythonError, "no (more) DataSets" );
		ret= NULL;
	}
	return( ret );
}

PyObject *python_SetTitle ( PyObject *self, PyObject *args, PyObject *kw )
{ int argc;
  char *kws[]= { "set", "Title", "parse", NULL };
  int idx= (int) *ascanf_setNumber, parse= False;
  char *SetTitle= NULL;
  PyObject *ret= NULL;

	CHECK_INTERRUPTED();

	argc= PyTuple_Size(args);
	if( !PyArg_ParseTupleAndKeywords(args, kw, "|izi:SetTitle", kws, &idx, &SetTitle, &parse ) ){
		return NULL;
	}

	if( argc ){
		if( !(idx>= 0 && idx< setNumber) ){
 			PyErr_SetString( XG_PythonError, "setNumber out of range" );
// 			PyErr_SetString( PyExc_ValueError, "setNumber out of range" );
			return(NULL);
		}
	}
	if( AllSets && SetTitle ){
		xfree( AllSets[idx].titleText );
		AllSets[idx].titleText= strdup( SetTitle );
	}
	if( AllSets && AllSets[idx].titleText ){
		if( parse ){
		  char *ntt, *parsed_end= NULL;
			if( (ntt= ParseTitlestringOpcodes( ActiveWin, idx, AllSets[idx].titleText, &parsed_end )) ){
				ret= PyString_FromString( ntt );
			}
			else{
				ret= PyString_FromString( /*strdup*/(AllSets[idx].titleText) );
			}
		}
		else{
			ret= PyString_FromString( /*strdup*/( AllSets[idx].titleText) );
		}
	}
	else{
		ret= PyString_FromString( /*strdup*/("") );
	}
	return( ret );
}

PyObject *python_SetName ( PyObject *self, PyObject *args, PyObject *kw )
{ int argc;
  char *kws[]= { "set", "Name", "parse", NULL };
  int idx= (int) *ascanf_setNumber, parse= False;
  char *SetName= NULL;
  PyObject *ret;

	CHECK_INTERRUPTED();

	argc= PyTuple_Size(args);
	if( !PyArg_ParseTupleAndKeywords(args, kw, "|izi:SetName", kws, &idx, &SetName, &parse ) ){
		return NULL;
	}

	if( argc ){
		if( !(idx>= 0 && idx< setNumber) ){
 			PyErr_SetString( XG_PythonError, "setNumber out of range" );
// 			PyErr_SetString( PyExc_ValueError, "setNumber out of range" );
			return(NULL);
		}
	}
	if( AllSets && SetName ){
		xfree( AllSets[idx].setName );
		AllSets[idx].setName= strdup( SetName );
	}
	if( AllSets && AllSets[idx].setName ){
		if( parse ){
		  char *ntt, *parsed_end= NULL;
			if( (ntt= ParseTitlestringOpcodes( ActiveWin, idx, AllSets[idx].setName, &parsed_end )) ){
				ret= PyString_FromString( ntt );
			}
			else{
				ret= PyString_FromString( /*strdup*/(AllSets[idx].setName) );
			}
		}
		else{
			ret= PyString_FromString( /*strdup*/( AllSets[idx].setName) );
		}
	}
	else{
		ret= PyString_FromString( /*strdup*/("") );
	}
	return( ret );
}

PyObject *python_SetInfo ( PyObject *self, PyObject *args, PyObject *kw )
{ int argc;
  char *kws[]= { "set", "Info", NULL };
  int idx= (int) *ascanf_setNumber;
  char *SetInfo= NULL;
  PyObject *ret;

	CHECK_INTERRUPTED();

	argc= PyTuple_Size(args);
	if( !PyArg_ParseTupleAndKeywords(args, kw, "|iz:SetInfo", kws, &idx, &SetInfo ) ){
		return NULL;
	}

	if( argc ){
		if( !(idx>= 0 && idx< setNumber) ){
 			PyErr_SetString( XG_PythonError, "setNumber out of range" );
// 			PyErr_SetString( PyExc_ValueError, "setNumber out of range" );
			return(NULL);
		}
	}
	if( AllSets && AllSets[idx].set_info ){
		ret= PyString_FromString( /*strdup*/( AllSets[idx].set_info) );
	}
	if( AllSets && SetInfo ){
		xfree( AllSets[idx].set_info );
		AllSets[idx].set_info= strdup( SetInfo );
	}
	else{
		ret= PyString_FromString( /*strdup*/("") );
	}
	return( ret );
}

static PyObject *DataColumn2Array( PyObject *self, int argc, long idx, int cooked, long col, PyObject *startObj, long end, long offset, long pad, double pad_low, double pad_high )
{ long i, j, visN= 0, N;
  long *visarray= NULL, targN=-1;
  long start=0, pl_set= 0, ph_set= 0, use_set_visible= 0;
  double *targ= NULL;
  DataSet *set;
  double *column, *Ncolumn= NULL;
  PyObject *ret, *visible= NULL;

	if( startObj ){
		if( PyInt_Check(startObj) || PyLong_Check(startObj) ){
			start= PyInt_AsLong(startObj);
		}
		else if( PyList_Check(startObj) ){
			if( !(visible= PyList_AsTuple(startObj)) ){
 				PyErr_SetString( XG_PythonError, "unexpected failure converting list to tuple" );
// 				PyErr_SetString( PyExc_RuntimeError, "unexpected failure converting list to tuple" );
				return(NULL);
			}
			visN= PyList_Size(startObj);
		}
		else if( PyTuple_Check(startObj) ){
			visible= startObj;
			visN= PyTuple_Size(startObj);
		}
		else if( PyArray_Check(startObj) ){
			visible= startObj;
			visN= PyArray_Size(startObj);
		}
		else{
			if( end<= 0 ){
 				PyErr_SetString( XG_PythonError, "start argument must be integer or a tuple/list/Numpy-array" );
// 				PyErr_SetString( PyExc_TypeError, "start argument must be integer or a tuple/list/Numpy-array" );
				goto DC2A_ESCAPE;
			}
			else{
			  /* visible argument is actually ignored in this case! */
				visible= startObj;
			}
		}
	}

	if( idx< 0 || idx>= setNumber || !AllSets || AllSets[(int)idx].numPoints<= 0 ){
 		PyErr_SetString( XG_PythonError, "invalid setnumber" );
// 		PyErr_SetString( PyExc_ValueError, "invalid setnumber" );
		goto DC2A_ESCAPE;
	}
	set= &AllSets[idx];
	if( col< 0 || col>= set->ncols ){
 		PyErr_SetString( XG_PythonError, "column number out of range" );
// 		PyErr_SetString( PyExc_ValueError, "column number out of range" );
		goto DC2A_ESCAPE;
	}
	if( cooked ){
		if( col== set->xcol ){
			column= set->xvec;
		}
		else if( col== set->ycol ){
			column= set->yvec;
		}
		else if( col== set->ecol ){
			column= set->errvec;
		}
		else if( col== set->lcol ){
			column= set->lvec;
		}
		else if( col== set->Ncol ){
#if ADVANCED_STATS==1
			if( (Ncolumn= (double*) calloc(set->numPoints, sizeof(double))) ){
				for( i= 0; i< set->numPoints; i++ ){
					Ncolumn[i]= NVAL(set,i);
				}
			}
			else{
				PyErr_NoMemory();
				goto DC2A_ESCAPE;
			}
			column= Ncolumn
#elif ADVANCED_STATS==2
			column= set->columns[set->Ncol];
#endif
		}
		else{
			column= set->columns[col];
		}
	}
	else{
		column= set->columns[col];
	}
	CLIP_EXPR( start, start, 0, set->numPoints-1 );
	CLIP_EXPR( offset, offset, 0, MAXINT );
	if( offset< 0 ){
		offset= 0;
	}
	if( !visible ){
		if( start< 0 ){
			start= 0;
		}
		CLIP_EXPR( end, end, -1, set->numPoints-1 );
		if( end< 0 ){
			end= set->numPoints-1;
		}
		if( pad>= 0 ){
			CLIP_EXPR( pad, pad, 0, MAXINT );
		}
		else{
			pad= 0;
		}
		if( pad ){
			if( NaN(pad_low) ){
				pad_low= column[start];
				pl_set= 1;
			}
			if( NaN(pad_high) ){
				pad_high= column[end];
				ph_set= 1;
			}
		}
	}
	else{
		offset= 0;
		start= 0;
		if( end>0 ){
			use_set_visible= 1;
		}
		else if( end< 0 ){
			use_set_visible= -1;
#if 0
		  int vN=0;
			if( PyTuple_Check(visible) ){
				for( i= 0; i< visN; i++ ){
					if( ASCANF_TRUE(PyFloat_AsDouble( PyTuple_GetItem(visible,i) )) ){
						vN+= 1;
					}
				}
			}
			else if( PyArray_Check(visible) ){
			  PyArrayIterObject *it;
			  PyArrayObject *parray= (PyArrayObject*) visible;
				it= (PyArrayIterObject*) PyArray_IterNew(visible);
				for( i= 0; i< visN; i++ ){
					if( it->index < it->size ){
						if( ASCANF_TRUE( PyFloat_AsDouble( PyArray_DESCR(parray)->f->getitem( it->dataptr, visible) )) ){
							vN+= 1;
						}
						PyArray_ITER_NEXT(it);
					}
				}
				Py_DECREF(it);
			}
#endif
		}
		end= set->numPoints- 1;
		if( argc> 4 && ascanf_verbose ){
			fprintf( StdErr, " (<offset> and subsequent arguments ignored)== " );
		}
	}
	if( visible ){
		switch( use_set_visible ){
			case 1:
				N= 0;
				if( ActiveWin && ActiveWin!= StubWindow_ptr ){
					for( i= 0; i<= end; i++ ){
						N+= ActiveWin->pointVisible[(int) idx][i];
					}
					if( ActiveWin->numVisible[idx]!= N ){
						if( pragma_unlikely(ascanf_verbose) ){
							fprintf( StdErr, " (DataColumn2Array[%d]: %d points visible according to numVisible, pointVisible says %d)== ",
								(int) idx, ActiveWin->numVisible[(int) idx], N
							);
						}
					}
				}
				else if( pragma_unlikely(ascanf_verbose) ){
					fprintf( StdErr, " (Warning: no window active, so 0 points are taken to be visible)== " );
				}
				break;
			case 0:
			case -1:{
			  long nons= 0;
				N= visN;
				if( !(visarray= (long*) calloc( visN, sizeof(long) )) ){
					PyErr_NoMemory();
					goto DC2A_ESCAPE;
				}
				if( PyTuple_Check(visible) ){
					for( i= 0; i< visN; i++ ){
					  long vidx;
						if( use_set_visible==0 ){
							vidx= PyInt_AsLong( PyTuple_GetItem(visible,i) );
						}
						else{
						  double vx= PyFloat_AsDouble( PyTuple_GetItem(visible,i) );
							if( ASCANF_TRUE(vx) ){
								vidx= i;
							}
							else{
								vidx= -1;
								nons+= 1;
							}
						}
						if( vidx< 0 || vidx>= set->numPoints ){
							N-= 1;
						}
						visarray[i]= vidx;
					}
				}
				else if( PyArray_Check(visible) ){
				  PyArrayIterObject *it;
				  PyArrayObject *parray= (PyArrayObject*) visible;
				  PyArrayObject* xd= NULL;
				  double *PyArrayBuf= NULL;
				  long *PyArrayBufLong= NULL;
					if( use_set_visible== 0 ){
						xd = (PyArrayObject*) PyArray_ContiguousFromObject( (PyObject*) visible, PyArray_LONG, 0, 0 );
					}
					else{
						xd = (PyArrayObject*) PyArray_ContiguousFromObject( (PyObject*) visible, PyArray_DOUBLE, 0, 0 );
					}
					if( xd ){
						if( use_set_visible== 0 ){
							PyArrayBufLong= (long*)PyArray_DATA(xd);
						}
						else{
							PyArrayBuf= (double*)PyArray_DATA(xd); /* size would be N*sizeof(double) */
						}
					}
					else{
						it= (PyArrayIterObject*) PyArray_IterNew(visible);
					}
					if( xd ){
						if( use_set_visible== 0 ){
// 							for( i= 0; i< visN; i++ ){
// 								visarray[i] = PyArrayBufLong[i];
// 							}
							if( visarray != PyArrayBufLong ){
								memcpy( visarray, PyArrayBufLong, visN * sizeof(long) );
							}
						}
						else{
							for( i= 0; i< visN; i++ ){
								if( ASCANF_TRUE(PyArrayBuf[i]) ){
									visarray[i] = i;
								}
								else{
									visarray[i] = -1;
									nons+= 1;
								}
							}
						}
					}
					else{
					  long vidx;
						for( i= 0; i< visN; i++ ){
							if( it->index < it->size ){
								if( use_set_visible== 0 ){
									vidx= PyInt_AsLong( PyArray_DESCR(parray)->f->getitem( it->dataptr, visible) );
								}
								else{
								  double vx= PyFloat_AsDouble( PyArray_DESCR(parray)->f->getitem( it->dataptr, visible) );
									if( ASCANF_TRUE(vx) ){
										vidx= i;
									}
									else{
										vidx= -1;
										nons+= 1;
									}
								}
								PyArray_ITER_NEXT(it);
							}
							else{
								vidx= -1;
							}
						}
						if( vidx< 0 || vidx>= set->numPoints ){
							N-= 1;
						}
						visarray[i]= vidx;
					}
					if( xd ){
						Py_XDECREF(xd);
					}
					else{
						Py_DECREF(it);
					}
				}
				if( pragma_unlikely(ascanf_verbose) && (N+nons)!= visN ){
					fprintf( StdErr, " (Warning: <visible> object references %d invalid datapoints) ", visN - N - nons );
				}
				break;
			}
		}
	}
	else{
		N= offset+ end- start+ 1+ 2*pad;
	}
	targN= MAX( N, 0 );
#if 0
	if( visarray && use_set_visible<= 0 && visN ){
		fprintf( StdErr, "visarray=%d", visarray[0] );
		for( i= 1; i< visN; i++ ){
			fprintf( StdErr, ",%d", visarray[i] );
		}
		fputc( '\n', StdErr );
	}
#endif
	if( !(targ= (double*) PyMem_New( double, (targN+1) )) ){
		PyErr_NoMemory();
		goto DC2A_ESCAPE;
	}
	if( visible && use_set_visible>0 ){
		if( !(visarray= (long*) PyMem_New( long, (targN+1) )) ){
			PyErr_NoMemory();
			goto DC2A_ESCAPE;
		}
	}
	if( visarray && N && use_set_visible>0 ){
	  signed char *pointVisible= (ActiveWin && ActiveWin!= StubWindow_ptr)? ActiveWin->pointVisible[(int) idx] : NULL;
		for( j= 0, i= 0; i<= end && j< N; i++ ){
			if( pointVisible[i] ){
				visarray[j++]= i;
			}
		}
	}
	if( visarray ){
	  long NN= (use_set_visible>0)? N : visN;
		if( NN ){
		  int j, vidx;
			if( use_set_visible> 0 ){
				start= visarray[0];
				end= visarray[N-1];
			}
			else{
				start= -1;
				end= 0;
			}
			if( !use_set_visible>0 ){
				CLIP( start, 0, set->numPoints-1 );
				CLIP( end, 0, set->numPoints-1 );
			}
			for( j= i= 0; i< NN; i++ ){
				vidx= visarray[i];
				if( use_set_visible>0 || (vidx>=0 && vidx<set->numPoints) ){
					targ[j]= column[vidx];
					j++;
					if( use_set_visible<= 0 ){
						if( start== -1 || vidx< start ){
							start= vidx;
						}
						if( vidx> end ){
							end= vidx;
						}
					}
				}
			}
		}
		j= N;
	}
	else{
	  int ii= offset;
		for( j= 0; j< pad; j++, ii++ ){
			targ[ii]= pad_low;
		}
		for( j= 0, i= start; i<= end; j++, i++, ii++ ){
			targ[ii]= column[i];
		}
		for( j= 0; j< pad; j++, ii++ ){
			targ[ii]= pad_high;
		}
	}
	if( pragma_unlikely(ascanf_verbose) ){
		fprintf( StdErr, " (copied %d(%d-%d) %selements from set#%d column %d to <target>[%d]",
			j, start, end, (visible)? "visible " : "", set->set_nr, (int) col, offset
		);
		if( pad ){
			fprintf( StdErr, " padded with %dx%s (low) and %dx%s (high)",
				pad, ad2str(pad_low, d3str_format,0), pad, ad2str( pad_high, d3str_format,0)
			);
		}
		fprintf( StdErr, " (%d elements))== ", targN );
	}
	{ npy_intp dim[1]= {targN};
/* 		ret= PyArray_FromDimsAndData( 1, dim, PyArray_DOUBLE, (char*) targ );	*/
		ret= PyArray_SimpleNewFromData( 1, dim, PyArray_DOUBLE, (void*) targ );
		PyArray_ENABLEFLAGS( (PyArrayObject*)ret, NPY_OWNDATA);
DC2A_ESCAPE:;
		xfree(Ncolumn);
		if( ret ){
			if( visarray && use_set_visible>0 ){
/* 			  PyObject *ret2= PyArray_FromDimsAndData( 1, dim, PyArray_LONG, (char*) visarray );	*/
			  PyObject *ret2= PyArray_SimpleNewFromData( 1, dim, PyArray_LONG, (void*) visarray );
				PyArray_ENABLEFLAGS( (PyArrayObject*)ret2, NPY_OWNDATA );
				return( Py_BuildValue( "(OO)", ret, ret2 ) );
			}
			else{
				return( Py_BuildValue( "O", ret ) );
			}
		}
		return(NULL);
	}
}

PyObject *python_DataColumn2Array ( PyObject *self, PyObject *args, PyObject *kw )
{ int argc;
  char *kws[]= { "set", "column", "start", "end", "offset", "pad", "padlow", "padhigh", NULL };
  long col, end=-1, offset= 0, idx= 0;
  PyObject *startObj= NULL;
  double pad_low, pad_high;
  long pad= 0;

	CHECK_INTERRUPTED();

	set_NaN(pad_low);
	set_NaN(pad_high);
	argc= PyTuple_Size(args);
	if( !PyArg_ParseTupleAndKeywords(args, kw, "ll|Ollldd:DataColumn2Array", kws,
			&idx, &col, &startObj, &end, &offset, &pad, &pad_low, &pad_high)
	){
		return NULL;
	}

	return( DataColumn2Array( self, argc, idx, False, col, startObj, end, offset, pad, pad_low, pad_high ) );
}

int ndArray_ShapeOK( PyArrayObject *parray, int check )
{ int ret= False;
	switch( check ){
		case 0:
		default:
			  // 20091125: check if this is in fact a 1-dimensional array. It *might* be necessary
			  // to accept only a >1 size for the 1st dimension if we want to be sure to be able
			  // to convert the arena to a contiguous vector with the dedicated routine.
			if( PyArray_NDIM(parray) == 1
				|| (PyArray_NDIM(parray) == 2 && (PyArray_DIMS(parray)[0] <= 1 || PyArray_DIMS(parray)[1] <= 1))
			){
				ret = True;
			}
			break;
	}
	return( ret );
}

static PyObject *Array2DataColumn( PyObject *self, int argc, long idx, long col, PyObject *data, long start, long end, long offset )
{ long i, j, N;
  PyArrayObject *parray= NULL;
  double *column;
  DataSet *set;

	if( idx< 0 || idx> setNumber || !AllSets || AllSets[idx].numPoints< 0 ){
 		PyErr_SetString( XG_PythonError, "setnumber out of range" );
// 		PyErr_SetString( PyExc_ValueError, "setnumber out of range" );
		return(NULL);
	}
	set= &AllSets[idx];
	if( col< 0 || col>= set->ncols ){
 		PyErr_SetString( XG_PythonError, "column number out of range" );
// 		PyErr_SetString( PyExc_ValueError, "column number out of range" );
		return(NULL);
	}
	if( PyList_Check(data) ){
		if( !(data= PyList_AsTuple(data)) ){
 			PyErr_SetString( XG_PythonError, "unexpected failure converting list to tuple" );
// 			PyErr_SetString( PyExc_RuntimeError, "unexpected failure converting list to tuple" );
			return(NULL);
		}
	}
	if( PyTuple_Check(data) ){
		N= PyTuple_Size(data);
	}
	else if( PyArray_Check(data) ){
		parray= (PyArrayObject*) data;
		if( !ndArray_ShapeOK(parray, 0) ){
  			PyErr_SetString( XG_PythonError, "data argument cannot be a multi-dimensional Numpy array currently" );
// 			PyErr_SetString( PyExc_TypeError, "data argument cannot be a multi-dimensional Numpy array currently" );
 			return(NULL);
//  			PyErr_Warn( PyExc_Warning, "multi-dimensional Numpy array aspect of data argument currently ignored!" );
		}
		N= PyArray_Size(data);
	}
	else{
 		PyErr_SetString( XG_PythonError, "data argument must be a tuple/list/Numpy-array" );
// 		PyErr_SetString( PyExc_TypeError, "data argument must be a tuple/list/Numpy-array" );
		return(NULL);
	}
	CLIP(start, 0, N-1 );
	CLIP(end, -1, N-1 );
	if( end== -1 ){
		end= N-1;
	}
	CLIP(offset, 0, MAXINT );
	if( set->numPoints< (offset+ end- start+ 1) ){
	  int n= offset+ end- start+ 1;
	  int old_n= set->numPoints;
		set->numPoints= n;
		realloc_points( set, n, False );
		if( n> maxitems ){
			maxitems= n;
			realloc_Xsegments();
		}
		for( j= 0; j< set->ncols; j++ ){
			for( i= old_n; i< n && set->columns[j]; i++ ){
				set->columns[j][i]= 0;
			}
		}
	}
	column= set->columns[col];
	if( parray ){
	  PyArrayObject* xd= NULL;
	  double *PyArrayBuf= NULL;
	  PyArrayIterObject *it;
		if( (xd = (PyArrayObject*) PyArray_ContiguousFromObject( (PyObject*) data, PyArray_DOUBLE, 0, 0 )) ){
			PyArrayBuf= (double*)PyArray_DATA(xd); /* size would be N*sizeof(double) */
		}
		else{
			it= (PyArrayIterObject*) PyArray_IterNew(data);
			for( i= 0; i< start; i++ ){
				if( it->index < it->size ){
					PyArray_ITER_NEXT(it);
				}
			}
		}
		if( PyArrayBuf ){
			for( j= 0, i= start; i<= end; j++, i++ ){
				column[offset+j]= PyArrayBuf[i];
			}
			Py_XDECREF(xd);
		}
		else{
			for( j= 0, i= start; i<= end; j++, i++ ){
				if( it->index < it->size ){
					column[offset+j]= PyFloat_AsDouble( PyArray_DESCR(parray)->f->getitem( it->dataptr, data) );
					PyArray_ITER_NEXT(it);
				}
			}
			Py_DECREF(it);
		}
	}
	else{
		for( j= 0, i= start; i<= end; j++, i++ ){
			column[offset+j]= PyFloat_AsDouble( PyTuple_GetItem(data,i) );
		}
	}
	if( pragma_unlikely(ascanf_verbose) ){
		fprintf( StdErr, " (copied %d elements from <data>[%d-%d] (%d elements) to set#%d column %d[%d])== ",
			j, start, end, N, set->set_nr, col, offset
		);
	}
	return( Py_BuildValue( "l", j ) );
}

PyObject *python_Array2DataColumn ( PyObject *self, PyObject *args, PyObject *kw )
{ int argc;
  char *kws[]= { "set", "column", "data", "start", "end", "offset", NULL };
  long idx, col;
  long start= 0, end= -1, offset= 0;
  PyObject *data;

	CHECK_INTERRUPTED();

	argc= PyTuple_Size(args);
#if 0
	if( argc== 3 ){
	  PyObject *arg1, *arg2, *arg3;
	  int ok;
		arg1= PyTuple_GetItem(args, 0);
		ok= PyLong_Check(arg1);
		idx= PyInt_AsLong(arg1);
		arg2= PyTuple_GetItem(args, 1);
		ok= PyLong_Check(arg2);
		col= PyInt_AsLong(arg2);
		arg3= PyTuple_GetItem(args, 2);
		ok= PyArray_Check(arg3);
	}
#endif
	if(
		!PyArg_ParseTupleAndKeywords(args, kw, "llO|lll:Array2DataColumn", kws, &idx, &col, &data, &start, &end, &offset )
/* 		!PyArg_ParseTuple(args, "llO|lll:Array2DataColumn", &idx, &col, &data, &start, &end, &offset )	*/
	){
		return NULL;
	}

	return( Array2DataColumn( self, argc, idx, col, data, start, end, offset ) );
}

PyObject *python_Set2Arrays ( PyObject *self, PyObject *args, PyObject *kw )
{ int argc;
  char *kws[]= { "set", "raw", "columns", "start", "end", "offset", "pad", "padlow", "padhigh", NULL };
  long end=-1, offset= 0, idx= 0;
  int raw= True, N= 0, Nresults;
  PyObject *columns= NULL, *startObj= NULL, *result= NULL;
  PyArrayObject *parray= NULL;
  PyArrayIterObject *it;
  double pad_low, pad_high;
  long pad= 0, i;

	CHECK_INTERRUPTED();

	set_NaN(pad_low);
	set_NaN(pad_high);
	argc= PyTuple_Size(args);
	if( !PyArg_ParseTupleAndKeywords(args, kw, "llO|Ollldd:Set2Arrays", kws,
			&idx, &raw, &columns, &startObj, &end, &offset, &pad, &pad_low, &pad_high)
	){
		return NULL;
	}

	if( PyList_Check(columns) ){
		if( !(columns= PyList_AsTuple(columns)) ){
 			PyErr_SetString( XG_PythonError, "unexpected failure converting list to tuple" );
// 			PyErr_SetString( PyExc_RuntimeError, "unexpected failure converting list to tuple" );
			return(NULL);
		}
	}
	if( PyTuple_Check(columns) ){
		N= PyTuple_Size(columns);
	}
	else if( PyArray_Check(columns) && ndArray_ShapeOK( (parray=(PyArrayObject*)columns), 0) ){
		N= PyArray_Size(columns);
	}
	else{
 		PyErr_SetString( XG_PythonError, "columns argument must be a tuple/list/1d-Numpy-array" );
// 		PyErr_SetString( PyExc_TypeError, "columns argument must be a tuple/list/1d-Numpy-array" );
		return(NULL);
	}

/* 	Nresults= N;	*/
	Nresults= 0;

/* 	if( !(result= PyTuple_New( Nresults)) )	*/
	if( !(result= PyList_New(Nresults)) )
	{
		PyErr_NoMemory();
		goto S2A_ESCAPE;
	}

	if( parray ){
		it= (PyArrayIterObject*) PyArray_IterNew(columns);
	}
	for( i= 0; i< N; i++ ){
	  PyObject *array;
		if( parray ){
			array= DataColumn2Array( self, argc, idx, !raw,
				PyInt_AsLong( PyArray_DESCR(parray)->f->getitem( it->dataptr, columns)),
				startObj, end, offset, pad, pad_low, pad_high );
			PyArray_ITER_NEXT(it);
		}
		else{
			array= DataColumn2Array( self, argc, idx, !raw,
				PyInt_AsLong( PyTuple_GetItem(columns,i)), startObj, end, offset, pad, pad_low, pad_high );
		}
		if( !array ){
			goto S2A_ESCAPE;
		}
		if( PyTuple_Check(array) ){
		  int ii, n= PyTuple_Size(array);
			for( ii= 0; ii< n; ii++ ){
			  PyObject *elem= PyTuple_GetItem(array, ii);
				if( PyList_Append( result, elem ) ){
					result= NULL;
					goto S2A_ESCAPE;
				}
				Nresults+= 1;
			}
		}
		else{
			if( PyList_Append( result, array ) ){
				result= NULL;
				goto S2A_ESCAPE;
			}
			Nresults+= 1;
		}
		Py_DECREF(array);
	}
	if( parray ){
		Py_DECREF(it);
	}

	if( result ){
		if( !(result= PyList_AsTuple(result)) ){
 			PyErr_SetString( XG_PythonError, "unexpected failure converting result list to tuple" );
// 			PyErr_SetString( PyExc_RuntimeError, "unexpected failure converting result list to tuple" );
		}
	}
S2A_ESCAPE:;
	return(result);
}

PyObject *python_Arrays2Set ( PyObject *self, PyObject *args, PyObject *kw )
{ int argc;
  char *kws[]= { "set", "columns", "data", "start", "end", "offset", NULL };
  long start= 0, end=-1, offset= 0, idx= 0, i;
  int N= 0, N2= 0, Nresults;
  PyObject *columns= NULL, *data= NULL, *result= NULL;
  PyArrayObject *parray= NULL, *pdarray= NULL;
  PyArrayIterObject *it, *itd;

	CHECK_INTERRUPTED();

	argc= PyTuple_Size(args);
	if( !PyArg_ParseTupleAndKeywords(args, kw, "lOO|lll:Arrays2Set", kws,
			&idx, &columns, &data, &start, &end, &offset)
	){
		return NULL;
	}

	if( PyList_Check(columns) ){
		if( !(columns= PyList_AsTuple(columns)) ){
 			PyErr_SetString( XG_PythonError, "unexpected failure converting column list to tuple" );
// 			PyErr_SetString( PyExc_RuntimeError, "unexpected failure converting column list to tuple" );
			return(NULL);
		}
	}
	if( PyTuple_Check(columns) ){
		N= PyTuple_Size(columns);
	}
	else if( PyArray_Check(columns) && ndArray_ShapeOK( (parray=(PyArrayObject*)columns), 0) ){
		N= PyArray_Size(columns);
	}
	else{
 		PyErr_SetString( XG_PythonError, "columns argument must be a tuple/list/1d-Numpy-array" );
// 		PyErr_SetString( PyExc_TypeError, "columns argument must be a tuple/list/1d-Numpy-array" );
		return(NULL);
	}
	if( PyList_Check(data) ){
		if( !(data= PyList_AsTuple(data)) ){
 			PyErr_SetString( XG_PythonError, "unexpected failure converting data list to tuple" );
// 			PyErr_SetString( PyExc_RuntimeError, "unexpected failure converting data list to tuple" );
			return(NULL);
		}
	}
	if( PyTuple_Check(data) ){
		N2= PyTuple_Size(data);
	}
	else if( PyArray_Check(data) && ndArray_ShapeOK( (pdarray=(PyArrayObject*)data), 0) ){
		N2= PyArray_Size(data);
	}
	else{
 		PyErr_SetString( XG_PythonError, "data argument must be a tuple/list/1d-Numpy-array" );
// 		PyErr_SetString( PyExc_TypeError, "data argument must be a tuple/list/1d-Numpy-array" );
		return(NULL);
	}
	if( N!= N2 ){
 		PyErr_SetString( XG_PythonError, "columns and data length must be equal!" );
// 		PyErr_SetString( PyExc_ValueError, "columns and data length must be equal!" );
		return(NULL);
	}

	Nresults= 0;

	if( !(result= PyList_New(Nresults)) )
	{
		PyErr_NoMemory();
		goto A2S_ESCAPE;
	}

	if( parray ){
		it= (PyArrayIterObject*) PyArray_IterNew(columns);
	}
	if( pdarray ){
		itd= (PyArrayIterObject*) PyArray_IterNew(data);
	}

	{ DataSet *this_set;
		if( idx>= 0 && idx<= setNumber && AllSets ){
		  int maxCol;
		  long column;
			this_set= &AllSets[idx];
			maxCol= this_set->ncols;
			for( i= 0; i< N; i++ ){
				if( parray ){
					column= PyInt_AsLong( PyArray_DESCR(parray)->f->getitem( it->dataptr, columns));
					PyArray_ITER_NEXT(it);
				}
				else{
					column= PyInt_AsLong( PyTuple_GetItem(columns,i));
				}
				 // must accomodate columns[column], so:
				column+= 1;
				if( column> maxCol ){
					maxCol = column;
				}
			}
			if( this_set->ncols != maxCol ){
				if( this_set->numPoints> 0 ){
					this_set->columns= realloc_columns( this_set, maxCol );
				}
				else{
					this_set->ncols= maxCol;
				}
			}
		}
	}

	  // 20090917: rewind the column array iterator...
	if( parray ){
		PyArray_ITER_RESET(it);
	}

	for( i= 0; i< N; i++ ){
	  PyObject *array, *ret;
	  long column;
		if( parray ){
			column= PyInt_AsLong( PyArray_DESCR(parray)->f->getitem( it->dataptr, columns));
			PyArray_ITER_NEXT(it);
		}
		else{
			column= PyInt_AsLong( PyTuple_GetItem(columns,i));
		}
		if( pdarray ){
		  /* would be logical to take 1D 'slices' of a 2D array, for instance... */
			array= PyArray_DESCR(pdarray)->f->getitem( itd->dataptr, data);
			PyArray_ITER_NEXT(itd);
		}
		else{
			array= PyTuple_GetItem(data, i);
		}
		if( !(ret= Array2DataColumn( self, argc, idx, column, array, start, end, offset )) ){
			goto A2S_ESCAPE;
		}
		if( PyTuple_Check(ret) ){
		  int ii, n= PyTuple_Size(ret);
			for( ii= 0; ii< n; ii++ ){
			  PyObject *elem= PyTuple_GetItem(ret, ii);
				if( PyList_Append( result, elem ) ){
					result= NULL;
					goto A2S_ESCAPE;
				}
				Nresults+= 1;
			}
		}
		else{
			if( PyList_Append( result, ret ) ){
				result= NULL;
				goto A2S_ESCAPE;
			}
			Nresults+= 1;
		}
		Py_DECREF(ret);
	}
	if( parray ){
		Py_DECREF(it);
	}
	if( pdarray ){
		Py_DECREF(itd);
	}

	if( result ){
		if( !(result= PyList_AsTuple(result)) ){
 			PyErr_SetString( XG_PythonError, "unexpected failure converting result list to tuple" );
// 			PyErr_SetString( PyExc_RuntimeError, "unexpected failure converting result list to tuple" );
		}
	}
A2S_ESCAPE:;
	return(result);
}

PyObject *python_SetAssociation ( PyObject *self, PyObject *args, PyObject *kw )
{ int argc;
  char *kws[]= { "set", "values", NULL };
  PyObject *values= NULL, *result= NULL;
  PyArrayObject *parray= NULL;
  PyArrayIterObject *it;
  PyArrayObject* xd= NULL;
  double *PyArrayBuf= NULL;
  DataSet *set;
  long i, idx, N= 0;
  double *assoc= NULL;

	CHECK_INTERRUPTED();

	argc= PyTuple_Size(args);
	if( !PyArg_ParseTupleAndKeywords(args, kw, "l|O:SetAssociation", kws,
			&idx, &values)
	){
		return NULL;
	}

	if( idx< 0 || idx> setNumber || !AllSets || AllSets[idx].numPoints< 0 ){
 		PyErr_SetString( XG_PythonError, "setnumber out of range" );
// 		PyErr_SetString( PyExc_ValueError, "setnumber out of range" );
		return(NULL);
	}
	set= &AllSets[idx];

	if( values ){
		if( PyList_Check(values) ){
			if( !(values= PyList_AsTuple(values)) ){
 				PyErr_SetString( XG_PythonError, "unexpected failure converting list to tuple" );
// 				PyErr_SetString( PyExc_RuntimeError, "unexpected failure converting list to tuple" );
				return(NULL);
			}
		}
		if( PyTuple_Check(values) ){
			N= PyTuple_Size(values);
		}
		else if( PyArray_Check(values) && ndArray_ShapeOK( (parray=(PyArrayObject*)values), 0) ){
			N= PyArray_Size(values);
		}
		else{
 			PyErr_SetString( XG_PythonError, "values argument must be a tuple/list/1d-Numpy-array" );
// 			PyErr_SetString( PyExc_TypeError, "values argument must be a tuple/list/1d-Numpy-array" );
			return(NULL);
		}
		if( set->allocAssociations!= N ){
			if( !(set->Associations= (double*) realloc( set->Associations, N*sizeof(double))) ){
				PyErr_NoMemory();
				goto SA_ESCAPE;
			}
			set->allocAssociations= N;
			if( set->allocAssociations> ASCANF_MAX_ARGS ){
				Ascanf_AllocMem( set->allocAssociations );
			}
		}
		if( parray ){
			if( (xd = (PyArrayObject*) PyArray_ContiguousFromObject( (PyObject*) values, PyArray_DOUBLE, 0, 0 )) ){
				PyArrayBuf= (double*)PyArray_DATA(xd); /* size would be N*sizeof(double) */
			}
			else{
				it= (PyArrayIterObject*) PyArray_IterNew(values);
			}
		}
		if( parray && PyArrayBuf ){
			if( set->Associations != PyArrayBuf ){
				memcpy( set->Associations, PyArrayBuf, N * sizeof(double) );
			}
		}
		else{
			for( i= 0; i< N; i++ ){
				if( parray ){
					if( it->index < it->size ){
						set->Associations[i]= PyFloat_AsDouble( PyArray_DESCR(parray)->f->getitem( it->dataptr, values) );
						PyArray_ITER_NEXT(it);
					}
					else{
						set_NaN( set->Associations[i] );
					}
				}
				else{
					set->Associations[i]= PyFloat_AsDouble( PyTuple_GetItem(values,i) );
				}
			}
		}
		set->numAssociations= N;
		if( parray ){
			if( xd ){
				Py_XDECREF(xd);
			}
			else{
				Py_DECREF(it);
			}
		}
		result= values;
		Py_INCREF(values);
	}
	else{
		if( !(assoc= (double*) PyMem_New( double, set->numAssociations )) ){
			PyErr_NoMemory();
			goto SA_ESCAPE;
		}
		else{
		  npy_intp dim[1];
			for( i= 0; i< set->numAssociations; i++ ){
				assoc[i]= set->Associations[i];
			}
			dim[0]= set->numAssociations;
/* 			result= PyArray_FromDimsAndData( 1, dim, PyArray_DOUBLE, (char*) assoc);	*/
			result= PyArray_SimpleNewFromData( 1, dim, PyArray_DOUBLE, (void*) assoc);
			PyArray_ENABLEFLAGS( (PyArrayObject*)result, NPY_OWNDATA );
		}
	}

SA_ESCAPE:;
	return(result);
}

static void ImportModuleVars( char *moduleName )
{ PyObject *module;
  char *mname;
#ifdef IS_PY3K
  struct PyModuleDef *moduleDef = NULL;
#endif

	if( !moduleName ){
		return;
	}
	mname = concat( "ascanf.", moduleName, NULL );
#ifdef IS_PY3K
	if( !(moduleDef = calloc(1, sizeof(struct PyModuleDef))) ){
		PyErr_SetString( XG_PythonError, "cannot allocate PyModuleDef" );
		PyErr_NoMemory();
		return;
	}
	module= Py3_InitModule3( moduleDef, mname, NULL,
		"ascanf variables exported by the DM module of the same name\n"
	);
#else
	module= Py_InitModule3( mname, NULL,
		"ascanf variables exported by the DM module of the same name\n"
	);
#endif
	xfree(mname);
	if( PyErr_Occurred() ){
		PyErr_Print();
	}
	{ ascanf_Function *af;
	  int i= 0, count= 0;
	  // 20101103: import using deref=True!
	  int deref = 0;
		af= &vars_ascanf_Functions[0];
		while( af ){
			if( af->name
			   && af->type != NOT_EOF && af->type != NOT_EOF_OR_RETURN && af->type != _ascanf_function
			   && af->type != _ascanf_procedure && af->type != _ascanf_novariable
			){
			  char *name;
			  PyObject *ret;
				if( af->dymod && strcmp(af->dymod->name, moduleName) == 0 ){
					if( af->name[0] == '$' ){
						name = strdup(&af->name[1]);
					}
					else{
						name = strdup(af->name);
					}
					{ char *c = name;
						while( *c ){
							switch( *c ){
								case '-':
									*c = '_';
									break;
							}
							c++;
						}
					}
					ret= Py_ImportVariableFromAscanf( &af, &af->name, 0, NULL, deref, 1 );
					if( ret!= Py_None ){
						Py_XINCREF(ret);
						PyModule_AddObject( module, name, ret );
						if( PyErr_Occurred() ){
							PyErr_Print();
						}
						count+= 1;
					}
					xfree(name);
				}
			}
			if( af->cdr ){
				af= af->cdr;
			}
			else if( i< ascanf_Functions-1 ){
				i+= 1;
				af= &vars_ascanf_Functions[i];
			}
			else{
				af= NULL;
			}
		}
		if( count ){
			Py_XINCREF(module);
			PyModule_AddObject( AscanfPythonModule, moduleName, module );
			if( PyErr_Occurred() ){
				PyErr_Print();
			}
		}
	}
}

static PyObject* python_LoadDyMod( PyObject *self, PyObject *args, PyObject *kw )
{ int argc;
  char *kws[]= { "name", "flags", NULL };
  char *moduleName = NULL, *loadFlags = NULL;

	ascanf_arg_error= 0;
	CHECK_INTERRUPTED();

	argc= PyTuple_Size(args);
	if( !PyArg_ParseTupleAndKeywords(args, kw, "s|s:LoadModule", kws,
			&moduleName, &loadFlags)
	){
		return NULL;
	}

	if( moduleName ){
	  int flags= RTLD_LAZY, auto_unload= False, no_dump= False, autolist= False;
	  DyModAutoLoadTables new;
	  char *c;
		if( loadFlags ){
			if( (c= strcasestr( loadFlags, "auto-load"))== 0 && isspace(c[9]) ){
				autolist= True;
			}
			if( (c= strcasestr( loadFlags, "export"))== 0 && isspace(c[6]) ){
				flags|= RTLD_GLOBAL;
			}
			if( (c= strcasestr( loadFlags, "auto" ))== 0 && isspace(c[4]) ){
				auto_unload= True;
			}
			if( (c= strcasestr( loadFlags, "nodump" ))== 0 && isspace(c[4]) ){
				no_dump= True;
			}
		}
		if( moduleName[0]!= '\n' ){
			if( *moduleName ){
				if( moduleName[ strlen(moduleName)-1 ]== '\n' ){
					moduleName[ strlen(moduleName)-1 ]= '\0';
				}
				if( autolist ){
				  char *c;
				  int str= False;
					if( (c= ascanf_index(moduleName, ascanf_separator, &str)) ){
						*c= '\0';
						memset( &new, 0, sizeof(new) );
						new.functionName= moduleName;
						new.DyModName= &c[1];
						new.flags= flags;
						AutoLoadTable= Add_LoadDyMod( AutoLoadTable, &AutoLoads, &new, 1 );
					}
					else{
						PyErr_Warn( PyExc_Warning, "invalid auto-load specification" );
					}
				}
				else{
					if( !LoadDyMod( moduleName, flags, no_dump, auto_unload ) ){
						ascanf_exit= True;
						ascanf_arg_error= 1;
					}
					else{
						ImportModuleVars( moduleName );
					}
				}
			}
		}
	}
	else{
		ascanf_arg_error= 1;
	}
	return PyBool_FromLong((long)!ascanf_arg_error);
}

static PyObject* python_UnReloadDyMod( PyObject *self, PyObject *args, PyObject *kw, int reload, char *parseSpec )
{ int argc;
  char *kws[]= { "name", "flags", "force", NULL };
  char *moduleName = NULL, *loadFlags = NULL;
  int force= 0, all= 0;

	ascanf_arg_error= 0;
	CHECK_INTERRUPTED();

	argc= PyTuple_Size(args);
	if( !PyArg_ParseTupleAndKeywords(args, kw, parseSpec /*"s|si:LoadModule"*/, kws,
			&moduleName, &loadFlags, &force )
	){
		return NULL;
	}

	if( moduleName ){
	  int flags= RTLD_LAZY, auto_unload= False, no_dump= False;
	  char *c;

		if( loadFlags ){
			if( reload ){
				if( (c= strcasestr( loadFlags, "export" )) && isspace( c[6]) ){
					flags|= RTLD_GLOBAL;
				}
				if( (c= strcasestr( loadFlags, "auto" )) && isspace( c[4]) ){
					auto_unload= True;
				}
				if( (c= strcasestr( loadFlags, "nodump" )) && isspace( c[6]) ){
					no_dump= True;
				}
			}
			if( (c= strcasestr( loadFlags, "all")) && isspace( c[3]) ){
				if( !reload ){
					all= True;
				}
			}
		}
		if( all ){
		  int n;
		  int N= DyModsLoaded, i= 0;
			while( DyModList && i<= N /* DyModsLoaded */ ){
				UnloadDyMod( DyModList->name, &n, force );
				i+= 1;
			}
			if( DyModsLoaded ){
				fprintf( StdErr, "Warning: %d modules are still loaded!\n", DyModsLoaded );
			}
		}
		if( moduleName[0]!= '\n' ){
			moduleName= parse_codes(moduleName);
			if( !all && *moduleName ){
			  int n, c;
				if( moduleName[ strlen(moduleName)-1 ]== '\n' ){
					moduleName[ strlen(moduleName)-1 ]= '\0';
				}
				n= UnloadDyMod( moduleName, &c, force );
				if( n== c ){
					if( reload ){
						  /* NB: we *could* reset Unloaded_Used_Modules here to the value it had
						   \ before the above call to UnloadDyMod(). However, as we're not
						   \ sure that the reload will have put all symbols back at they're original
						   \ addresses, we won't, since that would make a coredump in CleanUp
						   \ possible.
						   */
						if( !LoadDyMod( moduleName, flags, no_dump, auto_unload ) ){
							ascanf_exit= True;
							ascanf_arg_error= 1;
						}
						else{
							ImportModuleVars( moduleName );
						}
					}
				}
			}
		}
	}
	else{
		ascanf_arg_error= 1;
	}
	return PyBool_FromLong((long)!ascanf_arg_error);
}

static PyObject* python_ReloadDyMod ( PyObject *self, PyObject *args, PyObject *kw )
{
	return python_UnReloadDyMod( self, args, kw, True, "s|si:ReloadModule" );
}

static PyObject* python_UnloadDyMod ( PyObject *self, PyObject *args, PyObject *kw )
{
	return python_UnReloadDyMod( self, args, kw, False, "s|si:UnloadModule" );
}

static PyObject* python_IOImport_Data ( PyObject *self, PyObject *args, PyObject *kw )
{ int argc, ret;
  char *importLibName = NULL, *fileName = NULL;
  char *kws[]= { "importLibName", "fileName", NULL };

	ascanf_arg_error= 0;
	CHECK_INTERRUPTED();

	argc= PyTuple_Size(args);
	if( !PyArg_ParseTupleAndKeywords(args, kw, "ss:IOImportData", kws,
			&importLibName, &fileName )
	){
		return NULL;
	}
	ret = IOImport_Data( importLibName, fileName );
	if( !ret ){
		ImportModuleVars( importLibName );
	}
	return Py_BuildValue( "i", ret );
}

extern PyObject* python_AscanfCall ( PyObject *self, PyObject *args, PyObject *kw );
extern PyObject* python_AscanfCall2 ( PyObject *self, PyObject *args, PyObject *kw );

#pragma mark ----AscanfPythonMethods----
static PyMethodDef AscanfPythonMethods[] = {
#if DEBUG
	{ "ListVariables", (PyCFunction) python_ListVars, METH_VARARGS,
		"List all current ascanf variables."
	},
#endif
	{ "Eval", (PyCFunction) python_AscanfEval, METH_VARARGS|METH_KEYWORDS,
		"Eval(expression[,N=1][,dereference][,verbose=0]): Evaluate an ascanf expression.\n"
		" The <N> argument allows to indicate the number of elements to expect in the expression;\n"
		" Eval returns either a single scalar result, or a tuple containing the evaluation results.\n"
		" If <verbose> is 1, ascanf verbose mode is active for the duration of the evaluation (and if >1,\n"
		" also during the derefencing step).\n"
	},
	{ "ImportVariable", (PyCFunction) python_ImportVariable, METH_VARARGS|METH_KEYWORDS,
		"ImportVariable(variable[,(rows,columns)[,derefence]]): imports an ascanf variable\n"
		" When <variable> is an ascanf array, it is possible to specify a dimensions layout via a tuple/list.\n"
		" Set derefence=True when an attempt should be made to interpret the variable's value\n"
		" as an address and derefence it (this is automatic if the variable was declared as an\n"
		" address e.g. DCL[ &foo, <value>]).\n"
		" <variable> can also be a double floating point value: if a valid pointer to an ascanf object,\n"
		" that object will be imported, else the value will be returned 'as is'.\n"
		" Instead of a single <variable>, a list or tuple can be given as the 1st argument; ImportVariable\n"
		" will then import all the elements (applying the other arguments in identical fashion to all)\n"
		" and return a tuple containing the Python objects.\n"
	},
	{ "ImportVariableToModule", (PyCFunction) python_ImportVariableToModule, METH_VARARGS|METH_KEYWORDS,
		"ImportVariableToModule(module,variable[,(rows,columns)[,derefence]]): imports ascanf variable(s)\n"
		" as ImportVariable, except that reference(s) are created in the specified module. <module> can be\n"
		" either a string, in which case a new module will be created, or an existing module object.\n"
		" Upon success, the (newly created) module is returned. This function will not create empty new modules.\n"
		" NB: ascanf allows and uses certain characters in variable names that are invalid in Python. Leading $ symbols\n"
		" are replaced with \"D_\", leading % symbols with \"i_\".\n"
	},
	{ "ExportVariable", (PyCFunction) python_ExportVariable, METH_VARARGS|METH_KEYWORDS,
		"ExportVariable(name,variable[,replace][,IDict][,as_PObj][,label][,returnVar]): exports a Python <variable> to the\n"
		" ascanf variable <name>. If IDict, store in the internal dictionary.\n"
		" The <replace> argument allows to force the 'replacing' of existing variables as\n"
		" variables of the required type, if necessary.\n"
		" If <as_PObj> is true, export an ascanf object containing a reference to the python <variable> (e.g. for use with Python-Call)\n"
		" (Objects that have no current \"native\" representation in ascanf are exported in the same way regardless as_PObj).\n"
		" The <label> argument sets the ascanf variable $VariableLabel for the scope of the call\n"
/* 		" The function returns None upon failure, but the ascanf address of the\n"	*/
		" The function returns None upon failure, but a PyAscanfObject referencing the\n"
		" (newly created) variable upon success.\n"
		" As with ImportVariable, <name> and <variable> can be lists or tuples of identical length, which\n"
		" is equivalent to calling ExportVariable with each of the name,variable couples.\n"
		" <returnVar> applies to exported Python callable objects. If set, the 1st argument passed in should be a pointer\n"
		" to an ascanf object that will/can receive the callable's return value. This is handy when a Python function returns\n"
		" an array, as it avoids having to do a CopyArray[] into the destination variable.\n"
	},
#ifdef DEBUG
	{ "IdArray", (PyCFunction) python_IdArray, METH_VARARGS|METH_KEYWORDS,
		"IdArray(rows,columns): return a 1d or 2d matrix with all elements initialised in C allocation order"
	},
#endif
	{ "CheckEvent", (PyCFunction) python_CheckEvent, METH_VARARGS|METH_KEYWORDS,
		"CheckEvent([interval[,python_interval]]): check and handle event and/or change checking interval.\n"
		" The python_interval controls at what interval of calling ascanf and or xgraph python methods\n"
		" an additional event check is performed (for code that's mostly Python).\n"
	},
	{ "idle", (PyCFunction) python_NoOp, METH_VARARGS,
		"idle(): a function that does nothing but the event checking at the CheckEvent interval and returns None"
	},
	{ "Value2Str", (PyCFunction) python_ad2str, METH_VARARGS|METH_KEYWORDS,
		"Value2str(value): return a string representation of value as the ascanf printf routine would print it.\n"
		" This uses the numeric format as set by *DPRINTF* and prints ascanf pointer canonically.\n"
	},
	{ "call", (PyCFunction) python_AscanfCall, METH_VARARGS|METH_KEYWORDS,
		"call(method[,arguments][,repeat][,[as_array][,dereference][,verbose]): invoke the ascanf function or procedure 'method', passing it the optional remaining\n"
		" arguments (in a list or tuple) as its arguments. Returns the returned value, or None upon failure\n"
		" The optional <repeat> argument allows to repeat the exact call that many times, returning all return\n"
		" values in a tuple. This saves on overhead.\n"
		" The <as_array> argument controls whether multiple results (when repeating) are returned as a numpy array, or a tuple.\n"
		" The <dereference> and <verbose> arguments are treated as described for Eval().\n"
		" This version is not reentrant, i.e. it cannot reliably be called recursively, but it is quite fast.\n"
		" NB: if the ascanf method (function) returns values through pointer arguments, import it as a PyAscanfObject,\n"
		" and use its returnArgs mechanism.\n"
	},
	{ "callr", (PyCFunction) python_AscanfCall2, METH_VARARGS|METH_KEYWORDS,
		"callr(method[,arguments][,repeat][,[as_array][,dereference][,verbose]): invoke the ascanf function or procedure 'method', passing it the optional remaining\n"
		" arguments (in a list or tuple) as its arguments. Returns the returned value, or None upon failure\n"
		" The optional <repeat> argument allows to repeat the exact call that many times, returning all return\n"
		" values in a tuple. This saves on overhead.\n"
		" The <as_array> argument controls whether multiple results (when repeating) are returned as a numpy array, or a tuple.\n"
		" The <dereference> and <verbose> arguments are treated as described for Eval().\n"
		" This version is reentrant, i.e. it CAN be called recursively, but it is slower.\n"
		" NB this only applies to the wrapper that calls the ascanf function - which can still be unfit for recursive calling\n"
		" (e.g. many functions return a pointer to an internal static buffer).\n"
		" This is the default for calling a PyAscanfObject object as a function (cf. the PyAscanfObject 'reentrant' method).\n"
	},
	{NULL, NULL, 0, NULL}
};


extern PyObject* python_DataSet ( PyObject *self, PyObject *args, PyObject *kw );
extern PyObject* python_GetULabels ( PyObject *self, PyObject *args, PyObject *kw );

static PyMethodDef XGraphPythonMethods[] = {
	{ "TBARprogress", (PyCFunction) python_TBARprogress, METH_VARARGS|METH_KEYWORDS,
		"TBARprogress(current,final[,step]): shows progress in attached window's WM title bar"
	},
	{ "setNumber", (PyCFunction) python_SetNumber, METH_VARARGS|METH_KEYWORDS,
		"setNumber or setNumber(current): the current setNumber or else the total number of sets\n"
		" NOTE that this is backwards from the homologous ascanf function!\n"
	},
	{ "numPoints", (PyCFunction) python_NumPoints, METH_VARARGS|METH_KEYWORDS,
		"numPoints or numPoints(set[,numPoints]): maximum number of points or number of points in set <set>\n"
		" It <numPoints> is given, set the number of points in set <set> to <numPoints>\n"
	},
	{ "ncols", (PyCFunction) python_ncols, METH_VARARGS|METH_KEYWORDS,
		"ncols([set[,columns]): returns or sets the <set> set's number of columns"
	},
	{ "xcol", (PyCFunction) python_xcol, METH_VARARGS|METH_KEYWORDS,
		"xcol([set[,column]): returns or sets the <set> set's current X column (for the current window)"
	},
	{ "ycol", (PyCFunction) python_ycol, METH_VARARGS|METH_KEYWORDS,
		"ycol([set[,column]): returns or sets the <set> set's current Y column (for the current window)"
	},
	{ "ecol", (PyCFunction) python_ecol, METH_VARARGS|METH_KEYWORDS,
		"ecol([set[,column]): returns or sets the <set> set's current E (error) column (for the current window)"
	},
	{ "lcol", (PyCFunction) python_lcol, METH_VARARGS|METH_KEYWORDS,
		"lcol([set[,column]): returns or sets the <set> set's current Length column (for the current window)"
	},
	{ "Ncol", (PyCFunction) python_Ncol, METH_VARARGS|METH_KEYWORDS,
		"Ncol([set[,column]): returns or sets the <set> set's current Nobs column (for the current window)"
	},
	{ "DataVal", (PyCFunction) python_DataVal, METH_VARARGS|METH_KEYWORDS,
		"DataVal(set,column,index[,value]): (new) value #<index> in column <col> of set <set>"
	},
	{ "SetTitle", (PyCFunction) python_SetTitle, METH_VARARGS|METH_KEYWORDS,
		"SetTitle([set[,`Title[,parse]]): return and/or set the title for the current or set'th set\n"
		" If parse is given and True, the returned string is parsed for opcodes.\n"
	},
	{ "SetName", (PyCFunction) python_SetName, METH_VARARGS|METH_KEYWORDS,
		"SetName([index[,`Name[,parse]]]): return and/or set the legend-entry (name) for the current or idx'th set\n"
		" If `SetName is given and a valid string variable, the setName is updated accordingly.\n"
		" If parse is given and True, the returned string is parsed for opcodes.\n"
	},
	{ "SetInfo", (PyCFunction) python_SetInfo, METH_VARARGS|METH_KEYWORDS,
		"SetInfo([index[,`Info]]): return the current set_info for the current or idx'th set\n"
		" If `Info is given, store the string it contains in the set's set_info.\n"
		" NB: this function returns the old info, *not* the new if it is changed!\n"
	},
	{ "DataColumn2Array", (PyCFunction) python_DataColumn2Array, METH_VARARGS|METH_KEYWORDS,
		"DataColumn2Array(set,column[,start[,end[,offset[,pad=0[,padlow[,padhigh]]]]]]): return set's <set> column <column> in a Numpy array\n"
		" <start> and <end> specify (inclusive) source start and end of copying (end==-1\n"
		" to copy until last); <offset> specifies starting point in <dest_p> which will be expanded/shrunk to the correct size\n"
		" pad,padlow,padhigh: pad begin and/or end according to the SavGolayInit conventions* (pad==-1 is of course undefined).\n"
		" Padding starts at <offset>, so the 1st copied datapoint is at offset+pad; default padding values are those at\n"
		" <start> and <end>.\n"
		" <start>,<end> may also be given as <Visible>[,getVisible], with Visible an array/tuple. This means that only points will\n"
		" be returned that are visible in the currently active window (no active window => no visible points!).\n"
		" If getVisible==0, then the points currently referenced in Visible (as point numbers) will be retrieved.\n"
		" If getVisible==-1, then the points i will be retrieved where Visible[i]==True.\n"
		" The function returns either the requested datacolumn in an array, or a tuple containing that array plus the Visible array.\n"
		" * see the fourconv.so module.\n"
	},
	{ "Array2DataColumn", (PyCFunction) python_Array2DataColumn, METH_VARARGS|METH_KEYWORDS,
		"Array2DataColumn(set,column,data[,start[,end[,offset]]]): copy into set's <set> column <column> from <data> which\n"
		" must be a Numpy array, tuple or list. <start> and <end> specify (inclusive) source start and end of copying (end==-1\n"
		" to copy until last); <offset> specifies starting point in <set> which will be expanded as necessary\n"
	},
	{ "Set2Arrays", (PyCFunction) python_Set2Arrays, METH_VARARGS|METH_KEYWORDS,
		"Set2Arrays(set,raw,columns[,start[,end[,offset[,pad=0[,padlow[,padhigh]]]]]]):\n"
		" Return the <set>'s columns (a tuple/list/1d Numpy array) into a tuple of Numpy arrays.\n"
		" <start> and <end> specify (inclusive) source start and end of copying (end==-1\n"
		" to copy until last); <offset> specifies the starting point in the target arrays which will be expanded/shrunk\n"
		" to the correct size.\n"
		" pad,padlow,padhigh: pad begin and/or end according to the SavGolayInit conventions (pad==-1 is of course undefined).\n"
		" Padding starts at <offset>, so the 1st copied datapoint is at offset+pad; default padding values are those at\n"
		" <start> and <end>.\n"
		" If <raw> is True, uses the raw, unprocessed values; otherwise, processed values are used.\n"
	},
	{ "Arrays2Set", (PyCFunction) python_Arrays2Set, METH_VARARGS|METH_KEYWORDS,
		"Arrays2Set(set,columns,data[,start[,end[,offset]]]): reverse of Set2Arrays()"
	},
	{ "SetAssociation", (PyCFunction) python_SetAssociation, METH_VARARGS|METH_KEYWORDS,
		"SetAssociation(set[,values]): read or set DataSet association(s) for DataSet <set>.\n"
		" If only <set> is specified, returns the associations in a 1D Numpy array.\n"
		" Values can be a list/tuple/Numpy array with new associations.\n"
	},
	{ "DataSet", (PyCFunction) python_DataSet, METH_VARARGS|METH_KEYWORDS,
		"DataSet([setnr]): return an internal representation of either the current dataset ($CurrentSet)\n"
		" or the specified set number.\n"
	},
	{ "UserLabels", (PyCFunction) python_GetULabels, METH_VARARGS|METH_KEYWORDS,
		"GetULabels(): return a structure containing all the information of the active window's User Labels.\n"
		" The information returned is a dict object, containing:\n"
		" * label: a list with the User Labels\n"
		" * index: another dict that maps a User Label text to an element entry in the label list\n"
		"          if a given string occurs multiple times, its index entry will be a list with the element entries\n"
		" * linked2: another dict that maps a set number to the list of User Labels linked to it\n"
		" * type: anoteher dict that maps User Label type to User Labels\n"
		" * count: the number of User Labels\n"
		" Thus, foo['label'][ foo['index']['bar'] ]['text'] will return 'bar'\n"
		" foo['label'][ foo['index']['bar'] ]['start'] will return the start co-ordinate pair\n"
		" array(UL['label'])[UL['linked2'][0]] returns an array of all User Labels linking to set 0\n"
		" array(UL['label'])[UL['index']['A']] returns an array of all User Labels with text \"A\"\n"
	},
	{ "RedrawNow", (PyCFunction) python_RedrawNow, METH_VARARGS|METH_KEYWORDS,
		"RedrawNow([silenced[,all]]): redraw immediately.\n"
		" If silenced is missing or is False, silenced mode is *deactivated* for this one redraw.\n"
		" <all> determines whether all currently open windows are redrawn.\n"
	},
	{ "RedrawSet", (PyCFunction) python_RedrawSet, METH_VARARGS|METH_KEYWORDS,
		" RedrawSet(set[,immediate]): cause a possibly immediate redraw of <set> in all windows displaying it.\n"
		" Returns the number of windows redrawn (<immediate>== True) or set to be redrawn\n"
		" RedrawSet(-1,1) causes an immediate redraw of all windows wanting one\n"
	},
	{ "WaitEvent", (PyCFunction) python_WaitEvent, METH_VARARGS|METH_KEYWORDS,
		"WaitEvent([type[,message]]): wait for an event to occur in the currently active window\n"
		" When type, a string pointer, is not specified, wait for any event, otherwise\n"
		" for an event of the specified type. Currently supported:\n"
		"   type='key': wait for a keypress.\n"
		" The optional 2nd argument can be a string to be displayed in the waiting window's titlebar.\n"
		" NB: other events are not discarded, but processed. Hence, call WaitEvent('key') from the\n"
		" interactive Python console (IPython) in order to perform any GUI-driven actions that do not involve\n"
		" hitting a key in the currently active window (e.g. via the Settings dialog).\n"
	},
	{ "__idle_input_handler__", (PyCFunction) python_Python_grl_HandleEvents, METH_VARARGS,
		"Invokes the internal handler that processes X11 events when Python is waiting for input"
	},
	{ "HandleQueuedEvents", (PyCFunction) python_HandleEvents, METH_VARARGS|METH_KEYWORDS,
		"HandleQueuedEvents([caller]): handles all queued X11 events.\n"
	},
	{ "tkconsole", (PyCFunction) python_tkcons, METH_VARARGS|METH_KEYWORDS,
		" Initialises and calls a Tk-based interactive console\n"
	},
	{ "gtkconsole", (PyCFunction) python_gtkcons, METH_VARARGS|METH_KEYWORDS,
		" Initialises and calls a Gtk-based interactive console\n"
	},
	{ "interact", (PyCFunction) python_interact, METH_VARARGS|METH_KEYWORDS,
		" Initialises and calls the code.interact embedded interactive console\n"
	},
	{ "ipshell", (PyCFunction) python_ipshell, METH_VARARGS|METH_KEYWORDS,
		" Initialises and calls the IPython embedded interactive console\n"
	},
	{ "source", (PyCFunction) python_source, METH_VARARGS|METH_KEYWORDS,
		"source(filename): source (import) a Python file."
	},
	{ "IOImportData", (PyCFunction) python_IOImport_Data, METH_VARARGS|METH_KEYWORDS,
		"IOImportData('IOImport module', 'filename'): import the data in 'filename'\n"
		" using the specified IOImport module. Cf. the *DM_IO_IMPORT* command.\n"
	},
	{ "LoadModule", (PyCFunction) python_LoadDyMod, METH_VARARGS|METH_KEYWORDS,
		"LoadModule(name[,flags]): load a 'DyMod' module; equivalent to\n"
		" *LOAD_MODULE* flags\n"
		" name\n"
		" and the Load_Module ascanf function\n"
		" Supported flags: 'auto-load', 'export', 'auto' and 'nodump'\n"
	},
	{ "ReloadModule", (PyCFunction) python_ReloadDyMod, METH_VARARGS|METH_KEYWORDS,
		"ReloadModule(name[,flags]): reload a 'DyMod' module; equivalent to\n"
		" the Reload_Module ascanf function\n"
		" Supported flags: 'export', 'auto' and 'nodump'\n"
	},
	{ "UnloadModule", (PyCFunction) python_UnloadDyMod, METH_VARARGS|METH_KEYWORDS,
		"UnloadModule(name[,flags[,force]]): unload a 'DyMod' module; equivalent to\n"
		" *UNLOAD_MODULE* flags\n"
		" name\n"
		" and the Unload_Module ascanf function\n"
		" Supported flags: 'all'\n"
	},
	{NULL, NULL, 0, NULL}
};

static PyMethodDef PenPythonMethods[] = {
	{NULL, NULL, 0, NULL}
};


/* 20061023: numpy's import_array() macro does a return upon failure. This should NOT cause us
 \ to return (without value!) from our dymod initialisation function. Therefore, we need a wrapper that
 \ allows us to check reliably whether or not numpy was imported with success.
 \\ NB!! This routine has to exist in each module separately...!!!
 */
#ifdef IS_PY3K
static void *call_import_array( int *success )
#else
static void call_import_array( int *success )
#endif
{
	*success= 0;
	import_array();
	*success= 1;
#ifdef IS_PY3K
	// if we're here we've had success, return something valid and (hopefully) non-null
	return success;
#endif
}

static void InitModules( int force )
{
	if( initialised || force ){
	  int pA= pythonActive;
		if( AscanfPythonModule ){
			Py_XDECREF(AscanfPythonModule);
		}
#ifdef IS_PY3K
		AscanfPythonModule= Py3_InitModule3( &AscanfModuleDef, "ascanf", AscanfPythonMethods,
			"Builtins for interaction with ascanf\n"
			" Most functions accept keyword arguments, see the definitions below for the accepted keywords.\n"
		);
		fprintf( StdErr, "A" ); fflush( StdErr );
#else
		AscanfPythonModule= Py_InitModule3( "ascanf", AscanfPythonMethods,
			"Builtins for interaction with ascanf\n"
			" Most functions accept keyword arguments, see the definitions below for the accepted keywords.\n"
		);
#endif
		if( PyErr_Occurred() ){
			PyErr_Print();
		}
		if( AscanfPythonDictionary ){
			Py_XDECREF(AscanfPythonDictionary);
		}
		AscanfPythonDictionary= PyModule_GetDict( AscanfPythonModule );

		if( XGraphPythonModule ){
			Py_XDECREF(XGraphPythonModule);
		}
#ifdef IS_PY3K
		XGraphPythonModule= Py3_InitModule3( &XGraphModuleDef, "xgraph", XGraphPythonMethods,
			"Builtins for interaction with XGraph\n"
			" Most functions accept keyword arguments, see the definitions below for the accepted keywords.\n"
		);
		fprintf( StdErr, "X" ); fflush( StdErr );
#else
		XGraphPythonModule= Py_InitModule3( "xgraph", XGraphPythonMethods,
			"Builtins for interaction with XGraph\n"
			" Most functions accept keyword arguments, see the definitions below for the accepted keywords.\n"
		);
#endif
		if( XGraphPythonDictionary ){
			Py_XDECREF(XGraphPythonDictionary);
		}
		XGraphPythonDictionary= PyModule_GetDict( XGraphPythonModule );
		if( XG_PythonError ){
			Py_XDECREF(XG_PythonError);
		}
		XG_PythonError= PyErr_NewException( "xgraph.Error", NULL, NULL );
		PyDict_SetItemString( XGraphPythonDictionary, "Error", XG_PythonError );
		Py_XINCREF(XG_PythonError);
		PyModule_AddObject( XGraphPythonModule, "Error", XG_PythonError );
		if( PyErr_Occurred() ){
			PyErr_Print();
		}

		if( PenPythonModule ){
			Py_XDECREF(PenPythonModule);
		}
#ifdef IS_PY3K
		PenPythonModule= Py3_InitModule3( &PenModuleDef, "xgraph.Pen", PenPythonMethods,
			"Builtins for controlling XGraph pens\n"
		);
#else
		PenPythonModule= Py_InitModule3( "xgraph.Pen", PenPythonMethods,
			"Builtins for controlling XGraph pens\n"
		);
#endif
		if( PyErr_Occurred() ){
			PyErr_Print();
		}
		 // populate the Pen module, and add a number of other 'goodies' to the XGraph module:
		{ ascanf_Function *af;
		  int i= 0, count= 0;
		  // 20101103: import using deref=True!
		  int deref = 1;
			af= &vars_ascanf_Functions[0];
			while( af ){
				if( af->name ){
				  int iSP= 0, iNP= 0, iPS= 0, pens= 0;
				  char *name;
				  PyObject *ret;
					if( (iPS= (strcmp( af->name, "PensShown?")== 0))
						|| (pens= (strncmp( af->name, "Pens", 4)== 0))
						|| strncmp( af->name, "Pen", 3)== 0
						|| (iSP= (strcmp( af->name, "SelectPen")== 0))
						|| (iNP= (strcmp( af->name, "NumPens")== 0))
					){
						if( iSP ){
							name= "Select";
						}
						else if( iNP ){
							name= "Count";
						}
						else if( iPS ){
							name= "AllShown";
						}
						else if( pens ){
							name= &af->name[4];
						}
						else{
							name= &af->name[3];
						}
						// 20101103:
					  	ret= Py_ImportVariableFromAscanf( &af, &name, 0, NULL, deref, 1 );
						if( ret!= Py_None ){
							Py_XINCREF(ret);
							PyModule_AddObject( PenPythonModule, name, ret );
							if( PyErr_Occurred() ){
								PyErr_Print();
							}
							count+= 1;
						}
					}
					 // 20100527: search for other handy functions to add to the xgraph module
					else if( strcmp( af->name, "ParseArguments") == 0 ){
						name = af->name;
					  	ret= Py_ImportVariableFromAscanf( &af, &name, 0, NULL, deref, 1 );
						if( ret!= Py_None ){
							Py_XINCREF(ret);
							PyModule_AddObject( XGraphPythonModule, name, ret );
							if( PyErr_Occurred() ){
								PyErr_Print();
							}
						}
					}
					else if( af->special_fun == SHelp_fun ){
						name = af->name;
					  	ret= Py_ImportVariableFromAscanf( &af, &name, 0, NULL, deref, 1 );
						if( ret!= Py_None ){
							Py_XINCREF(ret);
							PyModule_AddObject( AscanfPythonModule, name, ret );
							if( PyErr_Occurred() ){
								PyErr_Print();
							}
						}
					}
				}
				if( af->cdr ){
					af= af->cdr;
				}
				else if( i< ascanf_Functions-1 ){
					i+= 1;
					af= &vars_ascanf_Functions[i];
				}
				else{
					af= NULL;
				}
			}
			if( count ){
				Py_XINCREF(PenPythonModule);
				PyModule_AddObject( XGraphPythonModule, "Pen", PenPythonModule );
				if( PyErr_Occurred() ){
					PyErr_Print();
				}
			}
		}

		pythonActive+= 1;
		PyRun_SimpleString( "import ascanf, xgraph" );
		pythonActive= pA;
	}
}

#ifdef IS_PY3K
PyObject *PyInit_xgraph()
#else
void initxgraph()
#endif
{
	fprintf( StdErr, "<<Python importing xgraph module>>\n" );
#ifdef IS_PY3K
	return XGraphPythonModule;
#endif
}

#ifdef IS_PY3K
PyObject *PyInit_ascanf()
#else
void initascanf()
#endif
{
	fprintf( StdErr, "<<Python importing ascanf module>>\n" );
#ifdef IS_PY3K
	return AscanfPythonModule;
#endif
}

DyModTypes initDyMod( INIT_DYMOD_ARGUMENTS )
{ static int called= 0;

	if( !DMBase ){
	  DyModLists *current;
	  int *UsePythonVersion_ptr;
		  // 20080922: Python.so uses the other interface-initialisation method, which allows to load a DyMod
		  // safely if the DyMod_Interface structure was expanded AT ITS TAIL.
		if( !(DMBase= initialise(NULL)) ){
			fprintf( stderr, "%s: Error attaching to xgraph's main (programme) module\n", theDyMod->name );
			fprintf( stderr, "Now returning DM_Error to our caller!\n" );
			return( DM_Error );
		}
		if( !DyMod_API_Check2(DMBase) ){
		  const char *de= dlerror();
			fprintf( stderr, "DyMod API version mismatch: either this module or XGraph is newer than the other...\n" );
			if( de ){
				fprintf( stderr, "\tDyMod loading error \"%s\"\n", de );
			}
			fprintf( stderr, "Now returning DM_Error to our caller!\n" );
			return( DM_Error );
		}
		if( (current= DyModList) ){
			while( current ){
				if( current->type== DM_Python ){
					fprintf( StdErr, "%s: %s already loaded from %s; overloading is impossible; using the existing module.\n",
						theDyMod->name, current->name, current->path
					);
					theDyMod->already_loaded_version= current;
					return( DM_Error );
				}
				current= current->cdr;
			}
		}
		  /* The XGRAPH_FUNCTION macro can be used to easily initialise the additional variables we need.
		   \ In line with the bail out remark above, this macro returns DM_Error when anything goes wrong -
		   \ i.e. aborts initDyMod!
		   */
		XGRAPH_FUNCTION( Create_Internal_ascanfString_ptr, "Create_Internal_ascanfString");
		XGRAPH_FUNCTION( show_ascanf_functions_ptr, "show_ascanf_functions");
// 		XGRAPH_FUNCTION( new_param_now_ptr, "new_param_now");
		XGRAPH_FUNCTION( ascanf_call_method_ptr, "ascanf_call_method");
		XGRAPH_FUNCTION( ascanf_WaitForEvent_h_ptr, "ascanf_WaitForEvent_h");
		XGRAPH_FUNCTION( find_ascanf_function_ptr, "find_ascanf_function" );
		XGRAPH_FUNCTION( register_VariableNames_ptr, "register_VariableNames");
// 		XGRAPH_FUNCTION( get_VariableWithName_ptr, "get_VariableWithName");
		XGRAPH_FUNCTION( register_VariableName_ptr, "register_VariableName");
		XGRAPH_FUNCTION( Delete_Variable_ptr, "Delete_Variable");
		XGRAPH_FUNCTION( Delete_Internal_Variable_ptr, "Delete_Internal_Variable");
		XGRAPH_FUNCTION( realloc_Xsegments_ptr, "realloc_Xsegments");
		XGRAPH_FUNCTION( realloc_points_ptr, "realloc_points");
		XGRAPH_FUNCTION( realloc_columns_ptr, "realloc_columns");
		XGRAPH_FUNCTION( Check_Columns_ptr, "Check_Columns");
		XGRAPH_FUNCTION( _ascanf_RedrawNow_ptr, "_ascanf_RedrawNow");
		XGRAPH_FUNCTION( Handle_An_Events_ptr, "Handle_An_Events");
		XGRAPH_FUNCTION( AscanfTypeName_ptr, "AscanfTypeName");
		XGRAPH_FUNCTION( ULabel_pixelCName_ptr, "ULabel_pixelCName" );
		XGRAPH_FUNCTION( ColumnLabelsString_ptr, "ColumnLabelsString" );
		XGRAPH_FUNCTION( LinkSet2_ptr, "LinkSet2" );
		XGRAPH_FUNCTION( grl_HandleEvents_ptr, "grl_HandleEvents" );
		XGRAPH_FUNCTION( Create_Internal_ascanfString_ptr, "Create_Internal_ascanfString" );
		XGRAPH_FUNCTION( Create_ascanfString_ptr, "Create_ascanfString" );
		XGRAPH_FUNCTION( LoadDyMod_ptr, "LoadDyMod" );
		XGRAPH_FUNCTION( UnloadDyMod_ptr, "UnloadDyMod" );
		XGRAPH_FUNCTION( ascanf_index_ptr, "ascanf_index" );
		XGRAPH_FUNCTION( Add_LoadDyMod_ptr, "Add_LoadDyMod" );
		XGRAPH_FUNCTION( IOImport_Data_ptr, "IOImport_Data" );
		XGRAPH_FUNCTION( get_FILEForDescriptor_ptr, "get_FILEForDescriptor" );
		XGRAPH_FUNCTION( DBG_SHelp_ptr, "DBG_SHelp" );

		XGRAPH_VARIABLE( Argv_ptr, "Argv" );
		XGRAPH_VARIABLE( ascanf_check_int_ptr, "ascanf_check_int" );
		XGRAPH_VARIABLE( TBARprogress_header_ptr, "TBARprogress_header" );
		XGRAPH_VARIABLE( TBARprogress_header2_ptr, "TBARprogress_header2" );
		XGRAPH_VARIABLE( maxitems_ptr, "maxitems" );
		XGRAPH_VARIABLE( AlwaysUpdateAutoArrays_ptr, "AlwaysUpdateAutoArrays" );
		XGRAPH_VARIABLE( ascanf_AutoVarWouldCreate_msg_ptr, "ascanf_AutoVarWouldCreate_msg" );
		XGRAPH_VARIABLE( ULabelTypeNames_ptr, "ULabelTypeNames" );
		XGRAPH_VARIABLE( ascanf_Functions_ptr,  "ascanf_Functions" );
		XGRAPH_VARIABLE( vars_ascanf_Functions_ptr, "vars_ascanf_Functions" );
		XGRAPH_VARIABLE( grl_HandlingEvents_ptr, "grl_HandlingEvents" );
		XGRAPH_VARIABLE( AutoLoadTable_ptr, "AutoLoadTable" );
		XGRAPH_VARIABLE( AutoLoads_ptr, "AutoLoads" );
		XGRAPH_VARIABLE( DyModsLoaded_ptr, "DyModsLoaded" );
		XGRAPH_VARIABLE( UsePythonVersion_ptr, "UsePythonVersion" );
		if( UsePythonVersion_ptr && theDyMod->name ){
		  int n;
			if( (n = sscanf( theDyMod->name, "Python.%d", UsePythonVersion_ptr )) == 1 ){
				*UsePythonVersion_ptr = n;
			}
		}
	}

	  /* NB: XGRAPH_ATTACH() is the first thing we ought to do, but only when !intialised ! */
	fprintf( StdErr, "%s::initDyMod(): Initialising %s loaded from %s (build: %s), call %d\n", __FILE__, theDyMod->name, theDyMod->path, XG_IDENTIFY(), ++called );

	if( !initialised ){
	  extern int init_AscanfCall_module();

		RVN= register_VariableNames(1);

		af_initialise( theDyMod, theDyMod->name );
		add_ascanf_functions( Python_Function, Python_Functions, "Python::initDyMod()" );

		fprintf( StdErr, "%s loading Python ", __FILE__ ); fflush( StdErr );
#ifdef IS_PY3K
		{ size_t len = strlen(Argv[0]);
		  wchar_t *buffer = malloc( len * sizeof(wchar_t) );
		  	if( buffer ){
				mbstowcs( buffer, Argv[0], len );
				Py_SetProgramName(buffer);
				// buffer should not be freed!
			}
		}
#else
		Py_SetProgramName(Argv[0]);
#endif

		// 20120413: invoke PyImport_AppendInittab() here. This is necessary for Python >=3.2 to
		// acknowledge the existence of our modules. It does not appear to be an issue that
		// the modules are created only later (in InitModules), possibly because the actual
		// invocation on the PyInit_xx functions is done only after we call Py3_InitModule3
		// in InitModules(). However, we should probably expect this to change in the future ...
#ifdef IS_PY3K
		PyImport_AppendInittab( "xgraph", &PyInit_xgraph );
		PyImport_AppendInittab( "ascanf", &PyInit_ascanf );
#else
		PyImport_AppendInittab( "xgraph", &initxgraph );
		PyImport_AppendInittab( "ascanf", &initascanf );
#endif

		setlocale( LC_ALL, "en_US.ISO8859-1" );
		Py_Initialize();
		if( PyErr_Occurred() ){
			PyErr_Print();
		}
		fprintf( StdErr, "." ); fflush( StdErr );

		InitModules(1);
		fprintf( StdErr, "." ); fflush( StdErr );

		PyErr_Clear();
		pythonActive= True;
#ifdef IS_PY3K
		PyRun_SimpleString( "import locale\n"
			"try:\n"
			"\tlocale.setlocale(locale.LC_ALL, locale.locale_alias['c.iso88591'])\n"
			//"\tlocale.setlocale(1, locale.locale_alias['c.iso88591'])\n"
			//"\tlocale.setlocale(2, locale.locale_alias['c.iso88591'])\n"
			"except:\n"
			"\tpass\n"
		);
#endif
		PyRun_SimpleString( "from __future__ import division" );
#ifdef linux
		// for some reason, sitecustomize.py seems to be ignored on my Debian6 distro, so:
		PyRun_SimpleString( "try:\n"
							"\timport pyximport; pyximport.install()\n"
						"except:\n"
							"\tpass\n\n"
		);
#endif
		  /* formally make the Numeric and our own modules available to the Python programmer: */
		PyRun_SimpleString( "import sys, numpy\nfrom numpy import *" );
		pythonActive= False;
		if( PyErr_Occurred() ){
			PyErr_Print();
		}
		fprintf( StdErr, "." ); fflush( StdErr );

		  /* Do some importing of functionality necessary to use numpy's arrays: */
		PyErr_Clear();
		{ int ok;
			call_import_array(&ok);
			if( PyErr_Occurred() || !ok ){
				PyErr_Print();
				initialised= -1;
				return( DM_Error );
			}
		}
		fprintf( StdErr, "." ); fflush( StdErr );

		MainModule= PyImport_AddModule( "__main__" );
		MainDictionary= PyModule_GetDict( MainModule );
		SysModule= PyImport_AddModule( "sys" );
		SysDictionary= PyModule_GetDict( SysModule );
		if( PyErr_Occurred() ){
			PyErr_Print();
		}
		fprintf( StdErr, "." ); fflush( StdErr );

#ifdef IS_PY3K
		{ PyObject *Py_stdin= PyFile_FromFile(stdin, "stdin", "r", Python_No_fclosing );
			if( Py_stdin ){
				PyModule_AddObject( SysModule, "stdin", Py_stdin );
			}
		}
		{ PyObject *Py_stdout= PyFile_FromFile(stdout, "stdout", "w", Python_No_fclosing );
			if( Py_stdout ){
				PyModule_AddObject( SysModule, "stdout", Py_stdout );
			}
		}
		PyRun_SimpleString( "try:\n"
							"\tfrom py2file import file\n"
						"except:\n"
							"\tpass\n"
						"\n"
		);
#endif
		{ PyObject *Py_StdErr= PyFile_FromFile(StdErr, "StdErr", "w", Python_No_fclosing );
			if( Py_StdErr ){
				PyModule_AddObject( SysModule, "stderr", Py_StdErr );
			}
		}

		Python_SysArgv0( NULL, NULL );

		{ char *command= concat( "sys.path.append('", PrefsDir, "')", NULL );
			pythonActive= True;
#ifdef IS_PY3K
			PyRun_SimpleString( "sys.path.insert(0, './py3k')" );
#else
			PyRun_SimpleString( "sys.path.insert(0, './py2k')" );
#endif
#if (PY_MAJOR_VERSION >= 3) || (PY_MAJOR_VERSION == 2 && PY_MINOR_VERSION >= 7)
			fprintf( StdErr, "###<prepending current directory to search path as in Python <2.7!>" ); fflush(stderr);
			PyRun_SimpleString( "sys.path.insert(0, '.')" );
#endif
			PyRun_SimpleString(command);
			pythonActive= False;
		}

		if( getenv("XG_INIT_IPYTHONSHELL") ){
			init_ipshell();
			fprintf( StdErr, "." ); fflush( StdErr );
		}

		_DM_Python_Interface_.type= DM_Python;
		_DM_Python_Interface_.isAvailable= Check_External_Availability;
		_DM_Python_Interface_.Run_Python_Expr= Run_Python_Expr;
		_DM_Python_Interface_.Import_Python_File= Import_Python_File_wrapper;
		_DM_Python_Interface_.Get_Python_ReturnValue= Get_Python_ReturnValue;
		_DM_Python_Interface_.Evaluate_Python_Expr= Evaluate_Python_Expr;
		_DM_Python_Interface_.open_PythonShell= open_PythonShell;
		_DM_Python_Interface_.ascanf_PythonCall= _ascanf_PythonCall;
		_DM_Python_Interface_.Python_SysArgv= Py2Sys_SetArgv;
		_DM_Python_Interface_.Python_INCREF= (void*) Python_INCREF;
		_DM_Python_Interface_.Python_DECREF= (void*) Python_DECREF;
		_DM_Python_Interface_.Python_CheckSignals= (void*) Python_CheckSignals;
		_DM_Python_Interface_.Python_SetInterrupt= (void*) Python_SetInterrupt;
		_DM_Python_Interface_.pythonActive= &pythonActive;

		init_AscanfCall_module();
		init_DataSet_module();
		init_ULabel_module();

		{ int n= 1;
		  double result;
			fascanf2( &n, "IDict[ DCL[$Python-Call-Result,NaN,\"Result from the last call to Python-Call[]\"] ] @", &result, ',' );
		}
		fprintf( StdErr, "." ); fflush( StdErr );

		{ char *command =
#ifdef IS_PY3K
				"print( ' ', sys.version, file=sys.stderr )\n"
#else
				"print >> sys.stderr, ' ', sys.version\n"
#endif
				"sys.stderr.flush()\n";
			pythonActive= True;
			PyRun_SimpleString(command);
			pythonActive= False;
#ifdef IS_PY3K
			fprintf( StdErr, "\tDefault encoding: %s\n", Py_FileSystemDefaultEncoding );
#endif
		}

		ascanf_VarLabel = get_VariableWithName( "$VariableLabel", 1 );

		fflush(stdout);
		fflush(StdErr);
		initialised= True;
	}
#ifdef IS_PY3K
	else{
		// 20120413: this seems weird, but apparently we have to re-append our modules?
		// (but we never get here, then why did I have a case where the xgraph module 'disappeared'??)
#ifdef DEBUG
		fprintf( StdErr, "<< re-appending internal python modules >>\n" );
#endif
		PyImport_AppendInittab( "xgraph", &PyInit_xgraph );
		PyImport_AppendInittab( "ascanf", &PyInit_ascanf );
	}
#endif
	theDyMod->libHook= &_DM_Python_Interface_;
	theDyMod->libname= XGstrdup( "DM-Python" );
	theDyMod->buildstring= concat(XG_IDENTIFY(), " compiled with Python ",
		STRING(PY_MAJOR_VERSION), ".", STRING(PY_MINOR_VERSION), ".", STRING(PY_MICRO_VERSION), NULL );
	theDyMod->description= XGstrdup(
		" A dynamic module (library) that contains\n"
		" hooks to interface with the Python language.\n"
	);

	return( DM_Python );
}

/* The close handler. We can be called with the force flag set to True or False. True means force
 \ the unload, e.g. when exitting the programme. In that case, we are supposed not to care about
 \ whether or not there are ascanf entries still in use. In the alternative case, we *are* supposed
 \ to care, and thus we should heed remove_ascanf_function()'s return value. And not take any
 \ action when it indicates variables are in use (or any other error). Return DM_Unloaded when the
 \ module was de-initialised, DM_Error otherwise (in that case, the module will remain opened).
 */
int closeDyMod( DyModLists *target, int force )
{ static int called= 0;
  int i;
  DyModTypes ret= DM_Error;
  FILE *SE= (initialised)? StdErr : stderr;
	fprintf( SE, "%s::closeDyMod(%d): Closing %s loaded from %s, call %d", __FILE__,
		force, target->name, target->path, ++called
	);
	if( target->loaded4 ){
		fprintf( SE, "; auto-loaded because of \"%s\"", target->loaded4 );
	}

	if( initialised ){
	  int r= remove_ascanf_functions( Python_Function, Python_Functions, force );
		if( force || r== Python_Functions ){

			if( pythonActive ){
				fprintf( StdErr, "\n### Sending SIGINT to running Python\n" );
				PyErr_SetInterrupt();
// #ifdef IS_PY3K
// 				PyRun_SimpleString( "try:\n\tprint( '### Python receiving SIGINT', file=sys.stderr )\nexcept:\n\tpass\n" );
// #else
// 				PyRun_SimpleString( "try:\n\tprint >>sys.stderr, '### Python receiving SIGINT\nexcept:\n\tpass\n'" );
// #endif
 				PyErr_CheckSignals();
				PyErr_Clear();
			}

			initialised= -1;

			if( in_IPShell ){
			  int pA= pythonActive;
				in_IPShell= False;
				fprintf( StdErr, "\n### unloading Python while in an IPython embedded shell" );
//				if( grl_HandlingEvents_ref < 0 || grl_HandlingEvents == grl_HandlingEvents_ref )
				if( !inInteractiveShell() )
				{
					  // we'd be here only when the exit request emanates from a command entered through
					  // the Python console.
					fprintf( StdErr, " (may crash)\n" );
					pythonActive+= 1;
					PyRun_SimpleString( "try:\n\txgraph.ipshell.IP.magic_Exit()\nexcept:\n\tpass\n" );
					pythonActive= pA;
					Py_Finalize();
				}
				else{
					// 20100507: we'll let the exit() routine yank the ground from under the Python console;
					// that at least appears not to cause a crash.
					fprintf( StdErr, "\n" );
				}
			}
			else{
				Py_Finalize();
			}

			memset( &_DM_Python_Interface_, 0, sizeof(DM_Python_Interface) );

			for( i= 0; i< Python_Functions; i++ ){
				Python_Function[i].dymod= NULL;
			}
			xfree( target->libname );
			xfree( target->buildstring );
			xfree( target->description );
			target->libHook= NULL;
			initialised= False;
			ret= target->type= DM_Unloaded;
			if( r<= 0 || ascanf_emsg ){
				fprintf( SE, " -- warning: variables are in use (remove_ascanf_functions() returns %d,\"%s\")",
					r, (ascanf_emsg)? ascanf_emsg : "??"
				);
				Unloaded_Used_Modules+= 1;
				if( force ){
					ret= target->type= DM_FUnloaded;
				}
			}
			fputc( '\n', SE );

			  // 20080922: as the very last step, free the interface structure:
			xfree(DMBase);
		}
		else{
			fprintf( SE, " -- refused: variables are in use (remove_ascanf_functions() returns %d out of %d)\n",
				r, Python_Functions
			);
		}

		register_VariableNames(RVN);
	}
	else{
		ret= target->type= DM_Unloaded;
	}
	fputc( '\n', SE );
	return(ret);
}

void R_init_Python()
{
	wrong_dymod_loaded( "R_init_Python()", "R", "Python.so" );
}

