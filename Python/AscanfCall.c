#include "config.h"
IDENTIFY( "ascanf library module for interfacing with Python: ascanf.Call function" );

#ifndef XG_DYMOD_SUPPORT
#error You need to define XG_DYMOD_SUPPORT in config.h - and compile xgraph with it!!!
#endif

#include "Python/Python_headers.h"

#include <stdio.h>
#include <stdlib.h>

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

#include "dymod_interface.h"

#define __PYTHON_MODULE_SRC__
#include "Python/DM_Python.h"

typedef struct PAO_Options {
	int verbose, call_reentrant;
	int returnArgs, *returnArg;
	PyObject *p_returnArg;
} PAO_Options;

/* PyAscanfObject implementation, modified from Python 2.4.3 cobject.[ch] files by RJVB */

/* Wrap void* pointers to be passed between C modules */

PyObject *PyAscanfObject_FromAscanfFunction( ascanf_Function *af )
{
	PyAscanfObject *self= NULL;

	if( af ){
		if( (self = PyObject_NEW(PyAscanfObject, &PyAscanfObject_Type))
			&& (self->opts= PyMem_New( PAO_Options, 1))
		){
			self->af= af;
			// 20120417: just to be sure...
			take_ascanf_address(af);
			af->PyAOself.self = self;
			af->PyAOself.selfaf = &(self->af);
			memset( self->opts, 0, sizeof(PAO_Options) );
			self->opts->call_reentrant= 1;
		}
		else{
			if( self ){
				PyObject_DEL(self);
				self= NULL;
				PyErr_NoMemory();
			}
		}
	}
	else{
		PyErr_SetString(PyExc_TypeError,
			"PyAscanfObject_FromAscanfFunction called with NULL pointer" );
	}

	return (PyObject *)self;
}

ascanf_Function *PyAscanfObject_AsAscanfFunction(PyObject *self)
{
	if (self) {
		if (self->ob_type == &PyAscanfObject_Type)
			return ((PyAscanfObject *)self)->af;
		PyErr_SetString(PyExc_TypeError,
				"PyAscanfObject_AsAscanfFunction with non-Ascanf-object");
	}
	if (!PyErr_Occurred() ){
		PyErr_SetString(PyExc_TypeError,
				"PyAscanfObject_AsAscanfFunction called with null pointer");
	}
	return NULL;
}

int PyAscanfObject_SetAscanfFunction(PyObject *self, ascanf_Function *af )
{ PyAscanfObject* cself = (PyAscanfObject*)self;
	if( cself == NULL || !PyAscanfObject_Check(cself) ){
		PyErr_SetString(PyExc_TypeError,
			"Invalid call to PyAscanfObject_SetAscanfFunction");
		return 0;
	}
	cself->af = af;
	af->PyAOself.self = cself;
	af->PyAOself.selfaf = &(cself->af);
	return 1;
}

static void PyAscanfObject_dealloc(PyAscanfObject *self)
{
#if 0
	if (self->destructor) {
		if(self->desc)
			((destructor2)(self->destructor))(self->cobject, self->desc);
		else
			(self->destructor)(self->cobject);
	}
#endif
	if( self->af ){
		self->af->PyAOself.self = NULL;
		self->af->PyAOself.selfaf = NULL;
	}
	if( self->opts->returnArgs> 0 && self->opts->returnArg ){
		PyMem_Free(self->opts->returnArg);
		self->opts->returnArg= NULL;
		self->opts->returnArgs= 0;
	}
	PyMem_Free(self->opts);
	PyObject_DEL(self);
}


static PyObject *PyAscanfObject_Value( PyAscanfObject *self /* , PyObject *args, void *closure */ )
{ extern int Py_ImportVariable_Copies;
	Py_ImportVariable_Copies= False;
	if( self && self->af ){
		return( Py_ImportVariableFromAscanf( &self->af, &self->af->name, 0, NULL, 0, 0 ) );
	}
	else{
		PyErr_SetString( XG_PythonError, "unexpected internal condition in PyAscanfObject_Value()" );
		return(NULL);
	}
}

static char *ATN(ascanf_Function *af)
{
	if( af->type== NOT_EOF || af->type== NOT_EOF_OR_RETURN ){
		return( AscanfTypeName(_ascanf_function) );
	}
	else{
		return( AscanfTypeName(af->type) );
	}
}

static PyObject *AO_repr( PyAscanfObject *self )
{ char *repr= NULL;
  PyObject *ret= NULL;
	if( self ){
		if( self->af ){
			repr= concat( "<", ATN(self->af), " PyAscanfObject \"", self->af->name, "\"", NULL );
			if( self->af->type== _ascanf_array ){
			  char *type= (self->af->iarray)? "integer(s)" : "double(s)";
			  char count[256];
				snprintf( count, sizeof(count)/sizeof(char), "0x%lx[%d]",
					((self->af->iarray)? (void*) self->af->iarray : (void*) self->af->array), self->af->N
				);
				repr= concat2( repr, "\t: ", count, " ", type, NULL );
			}
			if( self->af->usage ){
				repr= concat2( repr, "\t:\t\"", self->af->usage, "\"", NULL );
			}
			repr= concat2( repr, ">", NULL );
		}
		else{
			repr= XGstrdup("(NULL pointer)");
		}
	}
// 	ret= PyBytes_FromString(repr);
	ret= PyString_FromString( repr );
	xfree(repr);
	return(ret);
}

static PyObject *AO_str( PyAscanfObject *self )
{ char *repr= NULL;
  PyObject *ret= NULL;
	if( self ){
		if( self->af ){
			repr= concat( "<", ATN(self->af), " PyAscanfObject \"", self->af->name, "\">", NULL );
		}
		else{
			repr= XGstrdup("(NULL pointer)");
		}
	}
// 	ret= PyBytes_FromString(repr);
	ret= PyString_FromString( repr );
	xfree(repr);
	return(ret);
}

static PyObject *AO_address( PyAscanfObject *self )
{ double aaddr= 0;
  PyObject *ret= NULL;
	if( self ){
		if( self->af ){
		  double old_address = self->af->own_address;
			aaddr= take_ascanf_address(self->af);
			if( self->af->own_address != old_address ){
				fprintf( StdErr, "### Warning: corrected address for object '%s'! ### ", self->af->name );
				fflush(StdErr);
			}
		}
	}
	ret= PyFloat_FromDouble(aaddr);
	return(ret);
}

static PyObject *AscanfCall( ascanf_Function *af, PyObject *arglist, long repeats, int asarray, int deref,
						PAO_Options *opts, char *caller )
{ int fargs= 0, aargc= 0, volatile_args= 0;
  double result= 0, *aresult=NULL;
  static double *AARGS= NULL;
  static char *ATYPE= NULL;
  static ascanf_Function *LAF= NULL;
  static size_t LAFN= 0;
  double *aargs= NULL;
  char *atype= NULL;
  ascanf_Function *laf= NULL, *af_array= NULL;
  size_t lafN= 0;
  PyObject *ret= NULL;
  static ascanf_Function *nDindex= NULL;
  int aV = ascanf_verbose;

	if( arglist ){
		if( PyList_Check(arglist) ){
			if( !(arglist= PyList_AsTuple(arglist)) ){
 				PyErr_SetString( XG_PythonError, "unexpected failure converting argument list to tuple" );
// 				PyErr_SetString( PyExc_RuntimeError, "unexpected failure converting argument list to tuple" );
				return(NULL);
			}
		}
		if( !PyTuple_Check(arglist) ){
			PyErr_SetString(
 				XG_PythonError,
// 				PyExc_SyntaxError,
				"arguments to the ascanf method should be passed as a tuple or list\n"
				" NB: a 1-element tuple is specified as (value , ) !!\n"
			);
			return(NULL);
		}

		aargc= PyTuple_Size(arglist);
	}
	else{
		aargc= 0;
	}
	if( !af ){
		goto PAC_ESCAPE;
	}
	if( af->type!= _ascanf_procedure && af->Nargs> 0 ){
	  /* procedures can have as many arguments as MaxArguments, which is probably too much to allocate here.
	   \ However, we know how many arguments a function can get (if all's well...), and we can assure that
	   \ it will have space for those arguments
	   \ 20061015: unless it also has MaxArguments, i.e. Nargs<0 ...
	   */
		fargs= af->Nargs;
	}
	{ long n= (aargc+fargs+1)*2;
		if( opts->call_reentrant ){
			lafN= n;
			aargs= (double*) calloc( lafN, sizeof(double) );
			atype= (char*)  calloc( lafN, sizeof(char) );
			if( !aargs || !atype || !(laf= (ascanf_Function*) calloc( lafN, sizeof(ascanf_Function) )) ){
				PyErr_NoMemory();
				return(NULL);
			}
		}
		else{
			if( !LAF ){
				LAFN= n;
				AARGS= (double*) calloc( LAFN, sizeof(double) );
				ATYPE= (char*) calloc( LAFN, sizeof(char) );
				if( !AARGS || !ATYPE || !(LAF= (ascanf_Function*) calloc( LAFN, sizeof(ascanf_Function) )) ){
					PyErr_NoMemory();
					return(NULL);
				}
			}
			else if( n> LAFN ){
				AARGS= (double*) realloc( AARGS, n * sizeof(double) );
				ATYPE= (char*) realloc( ATYPE, n * sizeof(char) );
				if( !AARGS || !ATYPE || !(LAF= (ascanf_Function*) realloc( LAF, n * sizeof(ascanf_Function) )) ){
					PyErr_NoMemory();
					return(NULL);
				}
				else{
					for( ; LAFN< n; LAFN++ ){
						AARGS[LAFN]= 0;
						memset( &LAF[LAFN], 0, sizeof(ascanf_Function) );
					}
				}
				LAFN= n;
			}
			aargs= AARGS;
			atype= ATYPE;
			laf= LAF;
			lafN= LAFN;
		}
	}

	{ int a= 0, i;
		if( opts->verbose > 1 ){
			ascanf_verbose = 1;
		}
		if( af->type== _ascanf_array ){
			if( !nDindex ){
				nDindex= Py_getNamedAscanfVariable("nDindex");
			}
			if( nDindex ){
				af_array= af;
				aargs[a]= (af->own_address)? af->own_address : take_ascanf_address(af);
				af= nDindex;
				a+= 1;
			}
		}
		for( i= 0; i< aargc; i++, a++ ){
		  PyObject *arg= PyTuple_GetItem(arglist, i);
		  ascanf_Function *aaf;

			if( PyFloat_Check(arg) ){
				aargs[a]= PyFloat_AsDouble(arg);
				atype[a]= 1;
			}
#ifdef USE_COBJ
			else if( PyCObject_Check(arg) ){
				if( (aaf= PyCObject_AsVoidPtr(arg)) && (PyCObject_GetDesc(arg)== aaf->function) ){
					aargs[a]= (aaf->own_address)? aaf->own_address : take_ascanf_address(aaf);
					atype[a]= 2;
				}
				else{
 					PyErr_SetString( XG_PythonError, "unsupported PyCObject type does not contain ascanf pointer" );
// 					PyErr_SetString( PyExc_TypeError, "unsupported PyCObject type does not contain ascanf pointer" );
					goto PAC_ESCAPE;
				}
			}
#else
			else if( PyAscanfObject_Check(arg) ){
				if( (aaf= PyAscanfObject_AsAscanfFunction(arg)) ){
					aargs[a]= (aaf->own_address)? aaf->own_address : take_ascanf_address(aaf);
					atype[a]= 2;
				}
				else{
 					PyErr_SetString( XG_PythonError, "invalid PyAscanfObject type does not contain ascanf pointer" );
// 					PyErr_SetString( PyExc_TypeError, "invalid PyAscanfObject type does not contain ascanf pointer" );
					goto PAC_ESCAPE;
				}
			}
#endif
			else if( PyInt_Check(arg) || PyLong_Check(arg) ){
				aargs[a]= PyInt_AsLong(arg);
				atype[a]= 3;
			}
			else if( PyBytes_Check(arg)
#ifdef IS_PY3K
				|| PyUnicode_Check(arg)
#endif
			){
			  static char *AFname= "AscanfCall-Static-StringPointer";
			  ascanf_Function *saf= &laf[a];
				memset( saf, 0, sizeof(ascanf_Function) );
				saf->type= _ascanf_variable;
				saf->function= ascanf_Variable;
				if( !(saf->name= PyObject_Name(arg)) ){
					saf->name= XGstrdup(AFname);
				}
				saf->is_address= saf->take_address= True;
				saf->is_usage= saf->take_usage= True;
				saf->internal= True;
#ifdef IS_PY3K
				if( PyUnicode_Check(arg) ){
				  PyObject *bytes = NULL;
				  char *str = NULL;
					PYUNIC_TOSTRING( arg, bytes, str );
					if( !str ){
						if( bytes ){
							Py_XDECREF(bytes);
						}
						goto PAC_ESCAPE;
					}
					saf->usage= parse_codes( XGstrdup(str) );
					Py_XDECREF(bytes);
				}
				else
#endif
				{
					saf->usage= parse_codes( XGstrdup( PyBytes_AsString(arg) ) );
				}
				aargs[a]= take_ascanf_address(saf);
				atype[a]= 4;
				if( i && af_array ){
					volatile_args+= 1;
				}
			}
			else if( PyArray_Check(arg)
				|| PyTuple_Check(arg)
				|| PyList_Check(arg)
			){
			  static char *AFname= "AscanfCall-Static-ArrayPointer";
			  ascanf_Function *saf= &laf[a];
			  PyArrayObject *parray;
				atype[a]= 6;
				if( PyList_Check(arg) ){
					if( !(arg= PyList_AsTuple(arg)) ){
 						PyErr_SetString( XG_PythonError, "unexpected failure converting argument to tuple" );
// 						PyErr_SetString( PyExc_RuntimeError, "unexpected failure converting argument to tuple" );
						goto PAC_ESCAPE;
/* 						return(NULL);	*/
					}
					else{
						atype[a]= 5;
					}
				}
				memset( saf, 0, sizeof(ascanf_Function) );
				saf->type= _ascanf_array;
				saf->function= ascanf_Variable;
				if( !(saf->name= PyObject_Name(arg)) ){
					saf->name= XGstrdup(AFname);
				}
				saf->is_address= saf->take_address= True;
				saf->internal= True;
				if( a ){
					saf->car= &laf[a-1];
				}
				else{
					saf->car= &laf[lafN-1];
				}
				if( PyTuple_Check(arg) ){
					saf->N= PyTuple_Size(arg);
					parray= NULL;
				}
				else{
					saf->N= PyArray_Size(arg);
					parray= (PyArrayObject*) arg;
					atype[a]= 7;
				}
				if( (saf->array= (double*) malloc( saf->N * sizeof(double) )) ){
				  int j;
					if( parray ){
					  PyArrayObject* xd= NULL;
					  double *PyArrayBuf= NULL;
					  PyArrayIterObject *it;
						if( (xd = (PyArrayObject*) PyArray_ContiguousFromObject( (PyObject*) arg, PyArray_DOUBLE, 0, 0 )) ){
							PyArrayBuf= (double*)PyArray_DATA(xd); /* size would be N*sizeof(double) */
						}
						else{
							it= (PyArrayIterObject*) PyArray_IterNew(arg);
						}
						for( j= 0; j< saf->N; j++ ){
							if( PyArrayBuf ){
								  /* 20061016: indices used to be i?!?! */
								saf->array[j]= PyArrayBuf[j];
							}
							else{
								saf->array[j]= PyFloat_AsDouble( parray->descr->f->getitem( it->dataptr, arg) );
								PyArray_ITER_NEXT(it);
							}
						}
						if( xd ){
							Py_XDECREF(xd);
						}
						else{
							Py_DECREF(it);
						}
					}
					else{
						for( j= 0; j< saf->N; j++ ){
							saf->array[j]= PyFloat_AsDouble( PyTuple_GetItem(arg,j) );
						}
					}
					aargs[a]= take_ascanf_address(saf);
					if( i && af_array ){
						volatile_args+= 1;
					}
				}
				else{
					PyErr_NoMemory();
					goto PAC_ESCAPE;
				}
			}
#if 0
			else{
 				PyErr_SetString( XG_PythonError, "arguments should be scalars, strings, arrays or ascanf pointers" );
// 				PyErr_SetString( PyExc_SyntaxError, "arguments should be scalars, strings, arrays or ascanf pointers" );
				goto PAC_ESCAPE;
			}
#else
			else{
			  static char *AFname= "AscanfCall-Static-PyObject";
			  ascanf_Function *saf= &laf[a];
				memset( saf, 0, sizeof(ascanf_Function) );
				saf->function= ascanf_Variable;
				saf->internal= True;
				saf= make_ascanf_python_object( saf, arg, "AscanfCall" );
				if( !saf->name ){
					if( saf->PyObject_Name ){
						saf->name= XGstrdup(saf->PyObject_Name);
					}
					else{
						saf->name= XGstrdup(AFname);
					}
				}
				aargs[a]= take_ascanf_address(saf);
				atype[a]= 4;
				if( i && af_array ){
					volatile_args+= 1;
				}
			}
#endif
		}
		if( a> aargc ){
			aargc= a;
		}
	}
	if( opts->returnArgs> 0 ){
		ret= PyTuple_New( opts->returnArgs+1 );
		deref= 1;
		repeats= 1;
		asarray= 0;
	}
	else if( repeats> 1 ){
		ret= PyTuple_New(repeats);
	}
	else{
		repeats= 1;
	}
	{ int i, storeTuple= (repeats>1 || opts->returnArgs>0)? True : False;
		if( asarray && !(aresult= (double*) PyMem_New( double, repeats)) ){
 			PyErr_SetString( XG_PythonError, "can't allocate storage for results array" );
// 			PyErr_SetString( PyExc_MemoryError, "can't allocate storage for results array" );
			goto PAC_ESCAPE;
		}
		for( i= 0; i< repeats; i++ ){
		  ascanf_Callback_Frame __ascb_frame;
		  int r2, level= -1, acm;
			__ascb_frame.args= (aargc)? aargs : NULL;
			__ascb_frame.result= &result;
			__ascb_frame.level= &level;
			__ascb_frame.compiled= NULL;
			__ascb_frame.self= af;
#if defined(ASCANF_ARG_DEBUG)
			__ascb_frame.expr= caller;
#endif
			if( opts->verbose == 1 ){
				ascanf_verbose = 1;
			}
			if( pragma_unlikely(ascanf_verbose) ){
				fprintf( StdErr, "\n#P#\t%s", af->name );
				if( aargc ){
				  int i;
					fprintf( StdErr, "[%s", ad2str( aargs[0], d3str_format, NULL ) );
					for( i= 1; i< aargc; i++ ){
						fprintf( StdErr, ",%s", ad2str( aargs[i], d3str_format, NULL ) );
					}
					fprintf( StdErr, "]" );
				}
				fprintf( StdErr, "==" );
			}
			ascanf_arg_error= 0;
			ascanf_emsg= NULL;
			if( pragma_likely((af->special_fun != SHelp_fun) || !aargc || atype[0] != 4) ){
				acm= ascanf_call_method( af, aargc, aargs, &result, &r2, &__ascb_frame, 0 );
			}
			else{
			  int internal;
			  ascanf_Function *saf = parse_ascanf_address( aargs[0], 0, caller, (int) ascanf_verbose, NULL );
				if( aargc > 1 && (atype[1] == 3 || atype[1] == 1) ){
					internal = aargs[1] != 0;
				}
				else{
					internal = 0;
				}
				result = DBG_SHelp( saf->usage, internal );
			}
			if( pragma_unlikely(ascanf_emsg) ){
				PyErr_Warn( PyExc_Warning, ascanf_emsg );
			}
			if( pragma_unlikely(ascanf_verbose) ){
				if( ascanf_arg_error || ascanf_emsg ){
					fputs( " (an error occurred", StdErr );
					if( ascanf_emsg ){
						fprintf( StdErr, ": %s", ascanf_emsg );
					}
					fputs( ") ==", StdErr );
				}
				fprintf( StdErr, "%s\n", ad2str( result, d3str_format, NULL ) );
			}
			if( opts->verbose <= 1 ){
				ascanf_verbose = aV;
			}
			if( !acm ){
				if( r2 ){
					af->value= result;
					if( asarray ){
						aresult[i]= result;
					}
					else if( deref ){
					  int take_usage;
					  ascanf_Function *raf= parse_ascanf_address( result, 0, caller, (int) ascanf_verbose, &take_usage );
						if( raf ){
						  PyObject *oo= Py_ImportVariableFromAscanf( &raf, &raf->name, 0, NULL, deref-1, 0 );
							if( storeTuple ){
								PyTuple_SetItem( ret, i, oo );
							}
							else{
								ret= oo;
							}
						}
						else{
							goto AC_fallback_return;
						}
					}
					else{
AC_fallback_return:;
						if( storeTuple ){
							PyTuple_SetItem( ret, i, PyFloat_FromDouble(result) );
						}
						else{
							ret= Py_BuildValue( "d", result );
						}
					}
				}
				else{
					PyErr_Warn( PyExc_Warning, "call returned no value" );
					if( asarray ){
						set_NaN( aresult[i] );
					}
					else if( storeTuple ){
						PyTuple_SetItem( ret, i, Py_None );
					}
				}
			}
		}
	}

PAC_ESCAPE:;
	if( asarray && aresult ){
	  npy_intp dim[]= {repeats,0};
		if( (ret= PyArray_SimpleNewFromData( 1, dim, PyArray_DOUBLE, (void*) aresult )) ){
			((PyArrayObject*)ret)->flags|= NPY_OWNDATA;
		}
	}
	if( !ret && !PyErr_Occurred() ){
		ret= Py_None;
		Py_XINCREF(ret);
	}
	else if( opts->returnArgs ){
	  int i;
		for( i= 0; i< opts->returnArgs; i++ ){
		  int a= opts->returnArg[i];
			if( a< aargc ){
				switch( atype[a] ){
					case 1:
						PyTuple_SetItem( ret, i+1, PyFloat_FromDouble(aargs[a]) );
						break;
					case 2:{
					  PyObject *val;
					  ascanf_Function *af= parse_ascanf_address( aargs[a], 0, caller, (int) ascanf_verbose, NULL );
#ifdef USE_COBJ
						val= PyCObject_FromVoidPtrAndDesc(af, af->function, PCO_destructor);
#else
						val= PyAscanfObject_FromAscanfFunction(af);
#endif
						PyTuple_SetItem( ret, i+1, Py_BuildValue( "O", val ) );
						break;
					}
					case 3:
						PyTuple_SetItem( ret, i+1, PyInt_FromLong( (long)aargs[a]) );
						break;
					case 4:
						PyTuple_SetItem( ret, i+1, PyString_FromString( laf[a].usage ) );
						break;
					case 5:{
					  PyObject *val;
					  int j;
						val= PyList_New(laf[a].N);
					case 6:
						if( atype[a]== 6 ){
							val= PyTuple_New(laf[a].N);
						}
						if( val ){
							for( j= 0; j< laf[a].N; j++ ){
								switch( atype[a] ){
									case 5:
										PyList_SetItem(val, j, PyFloat_FromDouble(laf[a].array[j]) );
										break;
									case 6:
										PyTuple_SetItem(val, j, PyFloat_FromDouble(laf[a].array[j]) );
										break;
								}
							}
						}
						else{
							val= Py_None;
						}
						PyTuple_SetItem( ret, i+1, val );
						break;
					}
					case 7:{
					  PyObject *val;
					  npy_intp dim[]= {laf[a].N,0};
					  double *array= (double*) PyMem_New( double, laf[a].N );
						if( array ){
							memmove( array, laf[a].array, laf[a].N * sizeof(double) );
						}
						if( array && (val= PyArray_SimpleNewFromData( 1, dim, PyArray_DOUBLE, (void*) array )) ){
 							((PyArrayObject*)val)->flags|= NPY_OWNDATA;
						}
						else{
							val= Py_None;
						}
						PyTuple_SetItem( ret, i+1, val );
						break;
					}
				}
			}
			else{
				fprintf( StdErr, " (warning: ignoring requested-but-invalid return-argument #%d) ", a );
				PyTuple_SetItem( ret, i+1, Py_None );
			}
		}
	}

	if( ascanf_verbose ){
		if( af== nDindex ){
			af= af_array;
		}
		if( volatile_args ){
			fprintf( StdErr, " (warning: %d references to volatile arguments were apparently stored in the %s %s: their value will probably have been lost [for setting a string, use call('strdup',(string,)) ]) ",
				volatile_args, AscanfTypeName(af->type), af->name
			);
		}
	}
	if( opts->call_reentrant ){
		if( laf ){
		  int i;
		  static char *name= "ascanf.callr deleted variable";
			for( i= 0; i< aargc; i++ ){
				xfree(laf[i].name);
				laf[i].name= name;
				xfree(laf[i].usage);
				xfree(laf[i].array);
				laf[i].type= _ascanf_novariable;
			}
			xfree(laf);
			xfree(atype);
			xfree(aargs);
		}
	}
	else{
		if( laf ){
		  int i;
		  static char *name= "ascanf.call deleted variable";
			for( i= 0; i< aargc; i++ ){
				laf[i].name= name;
				xfree(laf[i].usage);
				xfree(laf[i].array);
				laf[i].type= _ascanf_novariable;
			}
			if( af== NULL ){
				PyErr_Warn( PyExc_Warning, "ascanf.call: releasing internal memory" );
				xfree(laf);
				xfree(atype);
				xfree(aargs);
				lafN= 0;
			}
		}
	}
	ascanf_verbose = aV;
	return( ret );
}

/* non-reentrant function to call ascanf functions and procedures. Uses a static buffer to store arguments, in order
 \ to minimise allocation overhead.
 */
PyObject* python_AscanfCall ( PyObject *self, PyObject *args, PyObject *kw )
{
#ifdef ASCANF_ALTERNATE
  int argc, deref=0, asarray=0;
  long repeats= 1;
  char *kws[]= { "method", "arguments", "repeat", "as_array", "dereference", "verbose", NULL };
  PyObject *amethod, *arglist= NULL;
  ascanf_Function *af;
  PAO_Options opts;

	CHECK_INTERRUPTED();

	argc= PyTuple_Size(args);
	  /* modify this: should be a varargs-style treatment!
	   \ correction: pass the arguments via a tuple or a list.
	   */
	opts.verbose = 0;
	if( !PyArg_ParseTupleAndKeywords(args, kw, "O|Olii:call", kws, &amethod, &arglist, &repeats, &asarray, &deref, &opts.verbose ) ){
		return NULL;
	}
	deref= abs(deref);
	memset( &opts, 0, sizeof(PAO_Options) );

#ifdef USE_COBJ
	if( PyCObject_Check(amethod) ){
		if( !(af= PyCObject_AsVoidPtr(amethod)) || (PyCObject_GetDesc(amethod)!= af->function) ){
 			PyErr_SetString( XG_PythonError, "NULL or otherwise invalid PyCObject passed as ascanf method" );
// 			PyErr_SetString( PyExc_SyntaxError, "NULL or otherwise invalid PyCObject passed as ascanf method" );
			af= NULL;
			return( NULL );
		}
	}
#else
	if( PyAscanfObject_Check(amethod) ){
		if( !(af= PyAscanfObject_AsAscanfFunction(amethod)) ){
 			PyErr_SetString( XG_PythonError, "NULL or otherwise invalid PyAscanfObject passed as ascanf method" );
// 			PyErr_SetString( PyExc_SyntaxError, "NULL or otherwise invalid PyAscanfObject passed as ascanf method" );
			af= NULL;
			return( NULL );
		}
	}
#endif
	else if( PyFloat_Check(amethod) ){
		af= parse_ascanf_address( PyFloat_AsDouble(amethod), 0, "python_Call", (int) ascanf_verbose, NULL );
	}
	else if( PyBytes_Check(amethod) ){
		af= Py_getNamedAscanfVariable( PyBytes_AsString(amethod) );
	}
#ifdef IS_PY3K
	else if( PyUnicode_Check(amethod) ){
	  PyObject *bytes = NULL;
	  char *str = NULL;
		PYUNIC_TOSTRING( amethod, bytes, str );
		if( str ){
			af= Py_getNamedAscanfVariable(str);
		}
		if( bytes ){
			Py_XDECREF(bytes);
		}
	}
#endif
	else if( amethod== Py_None ){
		return( AscanfCall( NULL, NULL, repeats, asarray, deref, &opts, "python_AscanfCall" ) );
	}
	else{
		PyErr_SetString(
 			XG_PythonError,
// 			PyExc_TypeError,
#ifdef USE_COBJ
			"ascanf method should be a string or a pointer to an ascanf object (= a double or a PyCObject!)"
#else
			"ascanf method should be a string or a pointer to an ascanf object (= a double or a PyAscanfObject!)"
#endif
		);
		return(NULL);
	}

	if( !af
		|| !(af->type== _ascanf_function || af->type== _ascanf_procedure || af->type== NOT_EOF || af->type== NOT_EOF_OR_RETURN)
	){
 		PyErr_SetString( XG_PythonError, "ascanf method should be a function or a procedure!" );
// 		PyErr_SetString( PyExc_TypeError, "ascanf method should be a function or a procedure!" );
		return(NULL);
	}

	return( AscanfCall( af, arglist, repeats, asarray, deref, &opts, "python_AscanfCall" ) );
#else
	PyErr_Warn( PyExc_Warning, "ascanf.call functionality not available - recompile with -DASCANF_ALTERNATE" );
	Py_RETURN_NONE;
#endif
}

PyObject* python_AscanfCall2 ( PyObject *self, PyObject *args, PyObject *kw )
{
#ifdef ASCANF_ALTERNATE
  int argc, deref= 0, asarray=0;
  long repeats= 1;
  char *kws[]= { "method", "arguments", "repeat", "as_array", "dereference", "verbose", NULL };
  PyObject *amethod, *arglist= NULL;
  ascanf_Function *af;
  PAO_Options opts;

	CHECK_INTERRUPTED();

	argc= PyTuple_Size(args);
	  /* modify this: should be a varargs-style treatment!
	   \ correction: pass the arguments via a tuple or a list.
	   */
	opts.verbose = 0;
	if( !PyArg_ParseTupleAndKeywords(args, kw, "O|Olii:call", kws, &amethod, &arglist, &repeats, &asarray, &deref, &opts.verbose ) ){
		return NULL;
	}
	deref= abs(deref);
	memset( &opts, 0, sizeof(PAO_Options) );
	opts.call_reentrant= 1;

#ifdef USE_COBJ
	if( PyCObject_Check(amethod) ){
		if( !(af= PyCObject_AsVoidPtr(amethod)) || (PyCObject_GetDesc(amethod)!= af->function) ){
 			PyErr_SetString( XG_PythonError, "NULL or otherwise invalid PyCObject passed as ascanf method" );
// 			PyErr_SetString( PyExc_TypeError, "NULL or otherwise invalid PyCObject passed as ascanf method" );
			af= NULL;
			return( NULL );
		}
	}
#else
	if( PyAscanfObject_Check(amethod) ){
		if( !(af= PyAscanfObject_AsAscanfFunction(amethod)) ){
 			PyErr_SetString( XG_PythonError, "NULL or otherwise invalid PyAscanfObject passed as ascanf method" );
// 			PyErr_SetString( PyExc_TypeError, "NULL or otherwise invalid PyAscanfObject passed as ascanf method" );
			af= NULL;
			return( NULL );
		}
	}
#endif
	else if( PyFloat_Check(amethod) ){
		af= parse_ascanf_address( PyFloat_AsDouble(amethod), 0, "python_Call", (int) ascanf_verbose, NULL );
	}
	else if( PyBytes_Check(amethod) ){
		af= Py_getNamedAscanfVariable( PyBytes_AsString(amethod) );
	}
#ifdef IS_PY3K
	else if( PyUnicode_Check(amethod) ){
	  PyObject *bytes = NULL;
	  char *str = NULL;
		PYUNIC_TOSTRING( amethod, bytes, str );
		if( str ){
			af= Py_getNamedAscanfVariable(str);
		}
		if( bytes ){
			Py_XDECREF(bytes);
		}
	}
#endif
	else{
	PyErr_SetString(
 		XG_PythonError,
// 		PyExc_TypeError,
#ifdef USE_COBJ
			"ascanf method should be a string or a pointer to an ascanf object (= a double or a PyCObject!)"
#else
			"ascanf method should be a string or a pointer to an ascanf object (= a double or a PyAscanfObject!)"
#endif
		);
		return(NULL);
	}

	if( !af
		|| !(af->type== _ascanf_function || af->type== _ascanf_procedure || af->type== NOT_EOF || af->type== NOT_EOF_OR_RETURN)
	){
 		PyErr_SetString( XG_PythonError, "ascanf method should be a function or a procedure!" );
// 		PyErr_SetString( PyExc_TypeError, "ascanf method should be a function or a procedure!" );
		return(NULL);
	}

	return( AscanfCall( af, arglist, repeats, asarray, deref, &opts, "python_AscanfCall2" ) );
#else
	PyErr_Warn( PyExc_Warning, "ascanf.call functionality not available - recompile with -DASCANF_ALTERNATE" );
	Py_RETURN_NONE;
#endif
}


static int initialised= 0;

static PyObject *PyAscanfObject_size( PyAscanfObject *self )
{
	if( self->af ){
		return( PyInt_FromLong( (long) (self->af->type==_ascanf_array)? self->af->N : 1 ) );
	}
	else{
		return( PyInt_FromLong(0) );
	}
}

static PyObject *PyAscanfObject_Nargs( PyAscanfObject *self )
{
	if( self->af ){
		return( PyInt_FromLong( (long) (self->af->Nargs>0)? self->af->Nargs : ASCANF_MAX_ARGS ) );
	}
	else{
		return( PyInt_FromLong(0) );
	}
}

static PyObject *PyAscanfObject_returnArgs_getter( PyAscanfObject *self )
{
	if( self->opts->p_returnArg ){
		return( self->opts->p_returnArg );
	}
	else{
		Py_RETURN_NONE;
	}
}

static PyObject *PyAscanfObject_reentrant( PyAscanfObject *self, PyObject *args )
{ int val;
	if( args && PyObject_Length(args)> 0 && self->af ){
	  char *format= concat( "|i:", self->af->name, NULL ), *Format= "|i:PyAscanfObject_reentrant";
		if( PyArg_ParseTuple( args, (format)? format : Format, &val ) ){
			self->opts->call_reentrant= val;
		}
		else{
			if( PyErr_Occurred() ){
				PyErr_Print();
			}
		}
		xfree(format);
	}
	return( PyInt_FromLong( (long) self->opts->call_reentrant));
}

static PyObject *PyAscanfObject_returnArgs( PyAscanfObject *self, PyObject *args )
{ PyObject *ra;;
	if( args && PyObject_Length(args)> 0 && self->af ){
	  char *format= concat( "|O:", self->af->name, NULL ), *Format= "|O:PyAscanfObject_returnArgs";
	  int maxArgs= (self->af->Nargs>0)? self->af->Nargs : ASCANF_MAX_ARGS;
		if( PyArg_ParseTuple( args, (format)? format : Format, &ra ) ){
			if( PyList_Check(ra) ){
				if( !(ra= PyList_AsTuple(ra)) ){
 					PyErr_SetString( XG_PythonError, "unexpected failure converting argument to tuple" );
// 					PyErr_SetString( PyExc_RuntimeError, "unexpected failure converting argument to tuple" );
					ra= NULL;
				}
			}
			if( ra== Py_None ){
				self->opts->returnArgs= 0;
				if( self->opts->returnArg ){
					PyMem_Free(self->opts->returnArg);
				}
				if( self->opts->p_returnArg ){
					Py_XDECREF(self->opts->p_returnArg);
				}
				self->opts->returnArg= NULL;
				self->opts->p_returnArg= NULL;
			}
			else if( PyInt_Check(ra) || PyLong_Check(ra) ){
			  int idx= PyInt_AsLong(ra);
				if( idx>= 0 && idx< maxArgs ){
					self->opts->returnArgs= 1;
					if( self->opts->returnArg ){
						PyMem_Free(self->opts->returnArg);
					}
					if( self->opts->p_returnArg ){
						Py_XDECREF(self->opts->p_returnArg);
					}
					if( (self->opts->returnArg= (int*) PyMem_New(int, 1)) ){
						self->opts->returnArg[0]= idx;
						self->opts->p_returnArg= Py_BuildValue( "O", ra );
					}
					else{
						PyErr_NoMemory();
						self->opts->returnArgs= 0;
						self->opts->returnArg= NULL;
						self->opts->p_returnArg= NULL;
					}
				}
				else{
 					PyErr_SetString( XG_PythonError, "invalid argument reference" );
// 					PyErr_SetString(  PyExc_LookupError, "invalid argument reference" );
				}
			}
			else if( PyArray_Check(ra) || PyTuple_Check(ra) ){
			  int i, *idx= NULL, N= 0, n= 0, err= 0, ignored= 0;
			  PyArrayObject *parray;
				if( PyTuple_Check(ra) ){
					N= PyTuple_Size(ra);
					parray= NULL;
				}
				else{
					N= PyArray_Size(ra);
					parray= (PyArrayObject*) ra;
				}
				if( (idx= (int*) PyMem_New(int, N)) ){
					if( !parray ){
						for( i= 0; i< N; i++ ){
						  PyObject *val= PyTuple_GetItem(ra,i);
							if( PyInt_Check(val) || PyLong_Check(val) ){
								idx[n]= PyInt_AsLong(val);
								if( idx[n]< 0 || idx[n]>= maxArgs ){
									err+= 1;
								}
								n+= 1;
							}
							else{
								ignored+= 1;
							}
						}
					}
					else{
					  PyArrayObject* xd= NULL;
					  int *PyArrayBuf= NULL;
					  PyArrayIterObject *it;
						if( (xd = (PyArrayObject*) PyArray_ContiguousFromObject( (PyObject*) ra, PyArray_INT, 0, 0 )) ){
							PyArrayBuf= (int*)PyArray_DATA(xd);
						}
						else{
							it= (PyArrayIterObject*) PyArray_IterNew(ra);
						}
						for( i= 0; i< N; i++ ){
							if( PyArrayBuf ){
								if( PyArrayBuf[i]< 0 || PyArrayBuf[i]>= maxArgs ){
									err+= 1;
								}
								idx[n]= PyArrayBuf[i];
								n+= 1;
							}
							else{
							  PyObject *val= parray->descr->f->getitem( it->dataptr, ra);
								if( PyInt_Check(val) || PyLong_Check(val) ){
									idx[n]= PyInt_AsLong(val);
									if( idx[n]< 0 || idx[n]>= maxArgs ){
										err+= 1;
									}
									n+= 1;
								}
								else{
									ignored+= 1;
								}
								PyArray_ITER_NEXT(it);
							}
						}
						if( xd ){
							Py_XDECREF(xd);
						}
						else{
							Py_DECREF(it);
						}
					}
					if( n && !err ){
						self->opts->returnArgs= n;
						if( self->opts->returnArg ){
							PyMem_Free(self->opts->returnArg);
						}
						self->opts->returnArg= idx;
						if( self->opts->p_returnArg ){
							Py_XDECREF(self->opts->p_returnArg);
						}
						if( (self->opts->p_returnArg= PyTuple_New(n)) ){
							for( i= 0; i< n; i++ ){
								PyTuple_SetItem( self->opts->p_returnArg, i, PyInt_FromLong((long) idx[i]) );
							}
						}
						else{
 							PyErr_SetString( XG_PythonError, "failure creating new tuple" );
// 							PyErr_SetString(  PyExc_RuntimeError, "failure creating new tuple" );
							self->opts->p_returnArg= Py_BuildValue("O", ra);
						}
						if( ignored ){
							PyErr_Warn( PyExc_Warning, "invalid argument references (non int or long) were ignored" );
						}
					}
					else{
						if( !n ){
 							PyErr_SetString( XG_PythonError, "no int or long argument references were found" );
// 							PyErr_SetString(  PyExc_LookupError, "no int or long argument references were found" );
						}
						else{
 							PyErr_SetString( XG_PythonError, "argument references must all be >=0 and < Nargs" );
// 							PyErr_SetString(  PyExc_ValueError, "argument references must all be >=0 and < Nargs" );
						}
					}
				}
				else{
					PyErr_NoMemory();
				}
			}
			else if( ra ){
 				PyErr_SetString( XG_PythonError, "argument references must be int or long" );
// 				PyErr_SetString(  PyExc_TypeError, "argument references must be int or long" );
			}
		}
		xfree(format);
	}
	if( PyErr_Occurred() ){
		PyErr_Print();
	}
	if( self->opts->p_returnArg ){
		return( self->opts->p_returnArg );
	}
	else{
		Py_RETURN_NONE;
	}
}

static PyObject *PyAscanfObject_ProcedureCode( PyAscanfObject *self, PyObject *args )
{
	if( !self->af || self->af->type!= _ascanf_procedure ){
		PyErr_SetString(PyExc_TypeError,
			"PyAscanfObject_ProcedureCode called with something not an ascanf procedure" );
		return(NULL);
	}
	return( PyString_FromString( self->af->procedure->expr ) );
}

static int PyAscanfObject_NewValue( PyAscanfObject *self, PyObject *newvals, void *closure )
{ int ret= -1;
	if( self && self->af ){
		if( newvals==NULL || PyObject_Length(newvals)== 0 ){
			PyErr_SetString( PyExc_AttributeError, "missing or empty new value" );
		}
		else{
			  // PyObject_Length() on a scalar will set an error (that the type doesn't have the len() method)
			  // so we clear any errors existing at this point
			PyErr_Clear();
			if( Py_ExportVariableToAscanf( self, NULL, newvals, 0, self->af->internal, 0, NULL, 0 ) ){
				ret= 0;
				if( PyErr_Occurred() ){
					PyErr_Print();
				}
			}
			else{
				if( PyErr_Occurred() ){
					PyErr_Print();
				}
			}
		}
	}
	return(ret);
}

static PyObject *PyAscanfObject_Usage( PyAscanfObject *self, PyObject *args )
{ char *repr= NULL;
	if( self && self->af ){
		repr= self->af->usage;
	}
	return( PyString_FromString(repr) );
}


static int PyAscanfObject_NewUsage( PyAscanfObject *self, PyObject *newvals, void *closure )
{ int ret= -1;
	if( self && self->af ){
		if( newvals== Py_None ){
			xfree(self->af->usage);
			ret= 0;
		}
		else{
		  char *c = NULL;
#ifdef IS_PY3K
		  PyObject *bytes = NULL;
#endif
			if( PyBytes_Check(newvals) ){
				c = parse_codes(PyBytes_AsString(newvals));
			}
#ifdef IS_PY3K
			else if( PyUnicode_Check(newvals) ){
				PYUNIC_TOSTRING( newvals, bytes, c );
			}
#endif
			if( c ){
				xfree(self->af->usage);
				self->af->usage= strdup(c);
				ret= 0;
			}
			else{
				if( PyErr_Occurred() ){
					PyErr_Print();
				}
				PyErr_SetString( PyExc_AttributeError, "a string or None is required" );
			}
#ifdef IS_PY3K
			if( bytes ){
				Py_XDECREF(bytes);
			}
#endif
		}
	}
	return(ret);
}

static PyObject *PyAscanfObject_verbose( PyAscanfObject *self )
{
	if( self ){
		return Py_BuildValue( "i", self->opts->verbose );
	}
	else{
		Py_RETURN_NONE;
	}
}

static int PyAscanfObject_setverbose( PyAscanfObject *self, PyObject *newvals, void *closure )
{ int ret= -1;
	if( self ){
		if( newvals== Py_None ){
			self->opts->verbose = 0;
			ret= 0;
		}
		else{
			if( PyArg_Parse( newvals, "i:PyAscanfObject_setverbose", &self->opts->verbose ) ){
				ret = 0;
			}
			else{
				if( PyErr_Occurred() ){
					PyErr_Print();
				}
			}
		}
	}
	return(ret);
}

static PyMethodDef PAO_methods[] = {
	{ "reentrant", (PyCFunction)PyAscanfObject_reentrant, METH_VARARGS,
		"reentrant([0|1]): controls whether calling a function PyAscanfObject is done in reentrant fashion or not\n"
		" (see the docstrings of ascanf.call and ascanf.lcall)\n"
	},
	{ "returnArgs", (PyCFunction)PyAscanfObject_returnArgs, METH_VARARGS,
		"returnArgs(refs): ascanf functions return a single (double) value, but can also return additional values via\n"
		" pointer arguments, like C functions. By default, calling a function PyAscanfObject will return the single\n"
		" floating point return value, and discard all other returned values. The returnArgs can be used to specify\n"
		" additional arguments to the ascanf function that should be returned (in a tuple). References should be\n"
		" an (array, list, tuple of) int(s).\n"
		" For instance, if CA=ascanf.ImportVariable('&CopyArray'), setting CA.returnArgs(0) will cause argument #0\n"
		" (the first of the argument list) to be returned as the 2nd element in a tuple returned by CA.\n"
		" CA( [1,2], [3,4,5,6], 1) will return a list [3,4,5,6] as that 2nd tuple element. If a1 is an ascanf array,\n"
		" then calling CA( ascanf.ImportVariable('&a1'), [9,10,11,12], 1) will set a1 to [9,10,11,12], and return\n"
		" a PyAscanfObject reference to a1 in the return-tuple's 2nd element. Returned argument types match the specified\n"
		" type as closely as possible.\n"
		" Note that this functionality is only available when calling a PyAscanfObject directly.\n"
	},
	{ NULL, NULL}
};

static PyGetSetDef PAO_getsetlist[] = {
	{ "code", (getter)PyAscanfObject_ProcedureCode, NULL,
		"code(): return an ascanf procedure's code (as originally entered) in a string.\n"
		, NULL
	},
	{ "size", (getter)PyAscanfObject_size, NULL,
		"return the number of elements of ascanf arrays, 1 for all other objects.\n"
		, NULL
	},
	{ "Nargs", (getter)PyAscanfObject_Nargs, NULL,
		"Nargs: returns the number of arguments a function accepts.\n"
		, NULL
	},
	{ "returnsArgs", (getter)PyAscanfObject_returnArgs_getter, NULL /*(setter)PyAscanfObject_returnArgs*/,
		"see returnArgs\n"
		, NULL
	},
	{ "value",
		(getter)PyAscanfObject_Value,
		(setter)PyAscanfObject_NewValue,
		"returns or modifies the object's current value.\n"
		" NB: Where possible, this returns a \"direct window\" on the ascanf data. Thus, for an\n"
		" ascanf arrary referenced via foo, foo.value[i]=j will set element i to j! This is currently\n"
		" implemented only for arrays and scalars.\n"
		" Note that attributing a string to a scalar stores the string in the text field. This doesn't work\n"
		" for other types, and it probably shouldn't for scalars either.\n"
		" Note also that reading and modifying an ascanf object in this way are limited forms of calling\n"
		" ascanf.ImportVariable and ascanf.ExportVariable referencing the object.\n"
		, NULL
	},
	{ "verbose",
		(getter)PyAscanfObject_verbose,
		(setter)PyAscanfObject_setverbose,
		"if set, activates ascanf verbose mode for the duration of the call itself (verbose==1) or\n"
		" for the full extent of the evaluation, argument parsing and result dereferencing included\n"
		" (verbose>1).\n"
	},
	{ "address",
		(getter)AO_address, NULL,
		"returns the ascanf address (a floating point value)"
		, NULL
	},
	{ "what",
		(getter)AO_repr, NULL,
		"returns a representational string about the object"
		, NULL
	},
	{ "text",
		(getter) PyAscanfObject_Usage,
		(setter) PyAscanfObject_NewUsage,
		"returns the associated text string (usage, description, the contents of ascanf string variables, ...)"
		, NULL
	},
	{ NULL, NULL, NULL, NULL },
};

static PyObject *PyAscanfObject_call( PyAscanfObject *self, PyObject *args, PyObject *kwds )
{ PyObject *ret= NULL;
	if( kwds && PyObject_Length(kwds) ){
 		PyErr_SetString( XG_PythonError, "keyword (named) arguments are not supported for calling PyAscanfObject functions" );
// 		PyErr_SetString(  PyExc_TypeError, "keyword (named) arguments are not supported for calling PyAscanfObject functions" );
	}
	else if( self && self->af ){
		if( self->af->type== _ascanf_variable ){
			if( args==NULL || PyObject_Length(args)== 0 ){
				ret= Py_ImportVariableFromAscanf( &self->af, &self->af->name, 0, NULL, 0, 0 );
			}
			else{
			  PyObject *var;
				  /* No need to allocate a fancy format string containing af->name : "O" should never fail. */
				if( PyArg_ParseTuple(args, "O:PyAscanfObject_call", &var ) ){
					if( Py_ExportVariableToAscanf( self, NULL, var, 0, self->af->internal, 0, NULL, 0 ) ){
						ret= Py_ImportVariableFromAscanf( &self->af, &self->af->name, 0, NULL, 0, 0 );
					}
				}
				else{
					if( PyErr_Occurred() ){
						PyErr_Print();
					}
				}
			}
		}
		else{
			ret= AscanfCall( self->af, args, 0, 0, 1, self->opts, "PyAscanfObject_call()" );
		}
	}
	return(ret);
}

PyDoc_STRVAR(PyAscanfObject_Type__doc__,
	"External representation of an ascanf_Function object, the central internal type of the Ascanf scripting language.\n"
	" Changes to the referenced Ascanf variable/function will be reflected immediately in already existing Python objects.\n"
	" As such, these are the most efficient way to exchange values between the two languages.\n"
	" PyAscanfObjects can be cast to a float, which will return the ascanf address (for the rare cases where this is required).\n"
	" They can also be called:\n"
	" For a function foo, foo(a,b,...) is equivalent to ascanf.callr(foo, (a,b,...), repeats=0, asarray=0, dereference=1)\n"
	" \t(unless foo.reentrant(0) has been called, in which case foo(a,b,...) == ascanf.call(foo, (a,b,...), ...) )\n"
	" For an array foo pointing to an ascanf array foo, foo(a) is equivalent to @[&foo,a] and foo(a,b) to @[&foo,a,b]\n"
	" For a variable foo, foo() returns the value, foo(a) sets and returns the new value.\n"
	" If foo points to an ascanf array, foo.value with return a 'window upon' the array data, so foo.value[i]=x will modify\n"
	" the contents of element i of the ascanf array. Care should of course be taken with statements like bar=foo.value in this\n"
	" case, for bar will not be updated automatically to reflect changes in e.g. size to the original ascanf array.\n"
	" Also please read the docstring for the returnArg method, which describes how to access values returned through pointer\n"
	" arguments passed to the ascanf function.\n"
	" Please also note that all arguments that are Python objects passed by reference (i.e. everything that's not a scalar\n"
	" or an ascanf object) will be converted to a temporary ascanf variable, which entails a certain overhead.\n"
);

static PyNumberMethods AO_as_number= {
	NULL,	/* nb_add */
	NULL,	/* nb_subtract */
	NULL,	/* nb_multiply */
#ifndef IS_PY3K
	NULL,	/* nb_divide */
#endif
	NULL,	/* nb_remainder */
	NULL,	/* nb_divmod */
	NULL,	/* nb_power */
	NULL,	/* nb_negative */
	NULL,	/* nb_positive */
	NULL,	/* nb_absolute */
	NULL,	/* nb_nonzero or nb_bool in py3k */
	NULL,	/* nb_invert */
	NULL,	/* nb_lshift */
	NULL,	/* nb_rshift */
	NULL,	/* nb_and */
	NULL,	/* nb_xor */
	NULL,	/* nb_or */
#ifndef IS_PY3K
	NULL,	/* nb_coerce */
#endif
	NULL,	/* nb_int */
	NULL,	/* nb_long or nb_reserved in py3k */
     (unaryfunc) AO_address,	/* nb_float;	*/
};

PyTypeObject PyAscanfObject_Type = {
	PyVarObject_HEAD_INIT(NULL,0)
	"PyAscanfObject",			/*tp_name*/
	sizeof(PyAscanfObject),   		/*tp_basicsize*/
	0,					/*tp_itemsize*/
	/* methods */
	(destructor)PyAscanfObject_dealloc, /*tp_dealloc*/
	(printfunc)0,			/*tp_print*/
	(getattrfunc)0,   		/*tp_getattr*/
	(setattrfunc)0,   		/*tp_setattr*/
#ifndef IS_PY3K
	(cmpfunc)0,   			/*tp_compare*/
#else
	NULL,					/*tp_reserved*/
#endif
	(reprfunc) AO_repr,		/*tp_repr*/
	&AO_as_number,			/*tp_as_number*/
	NULL,				/*tp_as_sequence*/
	NULL,				/*tp_as_mapping*/
	(hashfunc)0,			/*tp_hash*/
	(ternaryfunc) PyAscanfObject_call,   		/*tp_call*/
	(reprfunc)AO_str,		/*tp_str*/

	/* Space for future expansion */
	0L,0L,0L,0L,
	PyAscanfObject_Type__doc__,		/* Documentation string */
	0, 0, 0, 0, 0, 0,
	PAO_methods,
	0,
	PAO_getsetlist,
};


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

int init_AscanfCall_module()
{
	if( !initialised ){
	  int ok;
		  /* Do some importing of functionality necessary to use numpy's arrays: */
		call_import_array(&ok);

		if( PyType_Ready(&PyAscanfObject_Type) < 0 ){
			return 0;
		}

		initialised= True;
	}
	return(1);
}
