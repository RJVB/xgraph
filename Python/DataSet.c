#include "config.h"
IDENTIFY( "library module for exporting DataSets to Python" );

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

typedef struct PDO_Options {
  // unused for now, but PyMem_New() can show weird behaviour allocating an empty structure, so we put in a single
  // stub field.
 	char stub;
} PDO_Options;

PyObject *PyDataSetObject_FromDataSet( DataSet *set )
{ PyDataSetObject *self= NULL;

	if( set ){
		if( (self = PyObject_NEW(PyDataSetObject, &PyDataSetObject_Type)) 
			&& (self->opts= PyMem_New( PDO_Options, 1))
		){
			self->set= set;
			memset( self->opts, 0, sizeof(PDO_Options) );
		}
		else{
			if( self ){
				PyObject_DEL(self);
				self= NULL;
			}
		}
	}
	else{
		PyErr_SetString(PyExc_TypeError,
			"PyDataSetObject_FromDataSet called with NULL pointer" );
	}

	return (PyObject *)self;
}

DataSet *PyDataSetObject_AsDataSet(PyObject *self)
{
	if (self) {
		if (self->ob_type == &PyDataSetObject_Type)
			return ((PyDataSetObject *)self)->set;
		PyErr_SetString(PyExc_TypeError,
				"PyDataSetObject_AsDataSet with non-DataSet-object");
	}
	if (!PyErr_Occurred() ){
		PyErr_SetString(PyExc_TypeError,
				"PyDataSetObject_AsDataSet called with null pointer");
	}
	return NULL;
}

int PyDataSetObject_SetDataSet(PyObject *self, DataSet *set )
{ PyDataSetObject* cself = (PyDataSetObject*)self;
	if( cself == NULL || !PyDataSetObject_Check(cself) ){
		PyErr_SetString(PyExc_TypeError, 
			"Invalid call to PyDataSetObject_SetDataSet");
		return 0;
	}
	cself->set = set;
	return 1;
}

static void PyDataSetObject_dealloc(PyDataSetObject *self)
{
#if 0
	if (self->destructor) {
		if(self->desc)
			((destructor2)(self->destructor))(self->cobject, self->desc);
		else
			(self->destructor)(self->cobject);
	}
#endif
	PyMem_Free(self->opts);
	PyObject_DEL(self);
}


static PyObject *DO_repr( PyDataSetObject *self )
{ char *repr= NULL;
  PyObject *ret= NULL;
	if( self ){
		if( self->set ){
		  char set_nr[128];
			snprintf( set_nr, sizeof(set_nr)/sizeof(char), "#%d", self->set->set_nr );
			repr= concat( "<PyDataSetObject set ", set_nr, " \"", (self->set->setName)? self->set->setName : "?", "\"", NULL );
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

static PyObject *DO_str( PyDataSetObject *self )
{ char *repr= NULL;
  PyObject *ret= NULL;
	if( self ){
		if( self->set ){
		  char set_nr[128];
			snprintf( set_nr, sizeof(set_nr)/sizeof(char), "#%d", self->set->set_nr );
			repr= concat( "<PyDataSetObject set ", set_nr, ">", NULL );
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

static PyObject *DO_long( PyDataSetObject *self )
{ int set_nr=-1;
  PyObject *ret= NULL;
	if( self ){
		if( self->set ){
			set_nr= self->set->set_nr;
		}
	}
	ret= PyInt_FromLong(set_nr);
	return(ret);
}

static PyObject *DO_double( PyDataSetObject *self )
{ double set_nr=-1;
  PyObject *ret= NULL;
	if( self ){
		if( self->set ){
			set_nr= self->set->set_nr;
		}
	}
	ret= Py_BuildValue( "d", set_nr);
	return(ret);
}

static int initialised= 0;

#ifdef IS_PY3K
PyObject *PyString_FromString( const char *str )
{ PyObject *ret = PyUnicode_DecodeLatin1( str, strlen(str), NULL );
			//	PyUnicode_DecodeFSDefaultAndSize( str, strlen(str) );
	if( !ret ){
		PyErr_Clear();
		ret = Py_BuildValue( "s", str );
		if( !ret ){
			PyErr_Clear();
			ret = PyBytes_FromString( str );
		}
	}
	return ret;
}
#endif

static PyObject *PyDataSetObject_size( PyDataSetObject *self )
{
	return( PyInt_FromLong( (long) self->set->numPoints ) );
}

static PyObject *PyDataSetObject_filename( PyDataSetObject *self )
{
	return( PyString_FromString( (self->set->fileName)? self->set->fileName : "" ) );
}

static int PyDataSetObject_new_filename( PyDataSetObject *self, PyObject *newval )
{ int ret= -1;
	if( self && self->set ){
		if( newval==NULL || PyObject_Length(newval)== 0 || (!PyBytes_Check(newval)
#ifdef IS_PY3K
				&& !PyUnicode_Check(newval)
#endif
			)
		){
			PyErr_SetString( PyExc_AttributeError, "missing, empty or invalid new value" );
		}
		else{
			  // PyObject_Length() on a scalar will set an error (that the type doesn't have the len() method)
			  // so we clear any errors existing at this point
			PyErr_Clear();
			xfree( self->set->fileName );
#ifdef IS_PY3K
			if( PyUnicode_Check(newval) ){
			  PyObject *bytes = NULL;
			  char *str = NULL;
				PYUNIC_TOSTRING( newval, bytes, str );
				if( str ){
					self->set->fileName= XGstrdup(str);
					ret = 0;
				}
				if( bytes ){
					Py_XDECREF(bytes);
				}
			}
			else
#endif
			{
				self->set->fileName= XGstrdup( PyBytes_AsString(newval) );
				ret= 0;
			}
		}
	}
	return(ret);
}

static PyObject *PyDataSetObject_name( PyDataSetObject *self )
{
	return( PyString_FromString( (self->set->setName)? self->set->setName : "" ) );
}

static int PyDataSetObject_new_name( PyDataSetObject *self, PyObject *newval )
{ int ret= -1;
	if( self && self->set ){
		if( newval==NULL || PyObject_Length(newval)== 0 || (!PyBytes_Check(newval)
#ifdef IS_PY3K
				&& !PyUnicode_Check(newval)
#endif
			)
	 	){
			PyErr_SetString( PyExc_AttributeError, "missing, empty or invalid new value" );
		}
		else{
			  // PyObject_Length() on a scalar will set an error (that the type doesn't have the len() method)
			  // so we clear any errors existing at this point
			PyErr_Clear();
			xfree( self->set->setName );
#ifdef IS_PY3K
			if( PyUnicode_Check(newval) ){
			  PyObject *bytes = NULL;
			  char *str = NULL;
				PYUNIC_TOSTRING( newval, bytes, str );
				if( str ){
					self->set->setName= XGstrdup(str);
					ret = 0;
				}
				if( bytes ){
					Py_XDECREF(bytes);
				}
			}
			else
#endif
			{
				self->set->setName= XGstrdup( PyBytes_AsString(newval) );
				ret= 0;
			}
		}
	}
	return(ret);
}

static PyObject *PyDataSetObject_title( PyDataSetObject *self )
{
	return( PyString_FromString( (self->set->titleText)? self->set->titleText : "" ) );
}

static int PyDataSetObject_new_title( PyDataSetObject *self, PyObject *newval )
{ int ret= -1;
	if( self && self->set ){
		if( newval==NULL || PyObject_Length(newval)== 0 || (!PyBytes_Check(newval)
#ifdef IS_PY3K
				&& !PyUnicode_Check(newval)
#endif
			)
		){
			PyErr_SetString( PyExc_AttributeError, "missing, empty or invalid new value" );
		}
		else{
			  // PyObject_Length() on a scalar will set an error (that the type doesn't have the len() method)
			  // so we clear any errors existing at this point
			PyErr_Clear();
			xfree( self->set->titleText );
#ifdef IS_PY3K
			if( PyUnicode_Check(newval) ){
			  PyObject *bytes = NULL;
			  char *str = NULL;
				PYUNIC_TOSTRING( newval, bytes, str );
				if( str ){
					self->set->titleText= XGstrdup(str);
					ret = 0;
				}
				if( bytes ){
					Py_XDECREF(bytes);
				}
			}
			else
#endif
			{
				self->set->titleText= XGstrdup( PyBytes_AsString(newval) );
				ret= 0;
			}
		}
	}
	return(ret);
}

static PyObject *PyDataSetObject_set_info( PyDataSetObject *self )
{
	return( PyString_FromString( (self->set->set_info)? self->set->set_info : "" ) );
}

static int PyDataSetObject_new_set_info( PyDataSetObject *self, PyObject *newval )
{ int ret= -1;
	if( self && self->set ){
		if( newval==NULL || PyObject_Length(newval)== 0 || (!PyBytes_Check(newval)
#ifdef IS_PY3K
				&& !PyUnicode_Check(newval)
#endif
			)
		){
			PyErr_SetString( PyExc_AttributeError, "missing, empty or invalid new value" );
		}
		else{
			  // PyObject_Length() on a scalar will set an error (that the type doesn't have the len() method)
			  // so we clear any errors existing at this point
			PyErr_Clear();
			xfree( self->set->set_info );
#ifdef IS_PY3K
			if( PyUnicode_Check(newval) ){
			  PyObject *bytes = NULL;
			  char *str = NULL;
				PYUNIC_TOSTRING( newval, bytes, str );
				if( str ){
					self->set->set_info= XGstrdup(str);
					ret = 0;
				}
				if( bytes ){
					Py_XDECREF(bytes);
				}
			}
			else
#endif
			{
				self->set->set_info= XGstrdup( PyBytes_AsString(newval) );
				ret= 0;
			}
		}
	}
	return(ret);
}

static PyObject *PyDataSetObject_XUnits( PyDataSetObject *self )
{
	return( PyString_FromString( (self->set->XUnits)? self->set->XUnits : "" ) );
}

static int PyDataSetObject_new_XUnits( PyDataSetObject *self, PyObject *newval )
{ int ret= -1;
	if( self && self->set ){
		if( newval==NULL || PyObject_Length(newval)== 0 || (!PyBytes_Check(newval)
#ifdef IS_PY3K
				&& !PyUnicode_Check(newval)
#endif
			)
		){
			PyErr_SetString( PyExc_AttributeError, "missing, empty or invalid new value" );
		}
		else{
			  // PyObject_Length() on a scalar will set an error (that the type doesn't have the len() method)
			  // so we clear any errors existing at this point
			PyErr_Clear();
			xfree( self->set->XUnits );
#ifdef IS_PY3K
			if( PyUnicode_Check(newval) ){
			  PyObject *bytes = NULL;
			  char *str = NULL;
				PYUNIC_TOSTRING( newval, bytes, str );
				if( str ){
					self->set->XUnits= XGstrdup(str);
					ret = 0;
				}
				if( bytes ){
					Py_XDECREF(bytes);
				}
			}
			else
#endif
			{
				self->set->XUnits= XGstrdup( PyBytes_AsString(newval) );
				ret= 0;
			}
		}
	}
	return(ret);
}

static PyObject *PyDataSetObject_YUnits( PyDataSetObject *self )
{
	return( PyString_FromString( (self->set->YUnits)? self->set->YUnits : "" ) );
}

static int PyDataSetObject_new_YUnits( PyDataSetObject *self, PyObject *newval )
{ int ret= -1;
	if( self && self->set ){
		if( newval==NULL || PyObject_Length(newval)== 0 || (!PyBytes_Check(newval)
#ifdef IS_PY3K
				&& !PyUnicode_Check(newval)
#endif
			)
		){
			PyErr_SetString( PyExc_AttributeError, "missing, empty or invalid new value" );
		}
		else{
			  // PyObject_Length() on a scalar will set an error (that the type doesn't have the len() method)
			  // so we clear any errors existing at this point
			PyErr_Clear();
			xfree( self->set->YUnits );
#ifdef IS_PY3K
			if( PyUnicode_Check(newval) ){
			  PyObject *bytes = NULL;
			  char *str = NULL;
				PYUNIC_TOSTRING( newval, bytes, str );
				if( str ){
					self->set->YUnits= XGstrdup(str);
					ret = 0;
				}
				if( bytes ){
					Py_XDECREF(bytes);
				}
			}
			else
#endif
			{
				self->set->YUnits= XGstrdup( PyBytes_AsString(newval) );
				ret= 0;
			}
		}
	}
	return(ret);
}

static PyObject *PyDataSetObject_ncols( PyDataSetObject *self )
{
	return( PyInt_FromLong( (long) self->set->ncols ) );
}

static int PyDataSetObject_new_ncols( PyDataSetObject *self, PyObject *newval )
{ int ret= -1;
	if( self && self->set ){
		if( newval==NULL || PyObject_Length(newval)== 0 || !(PyInt_Check(newval) || PyLong_Check(newval)) ){
			PyErr_SetString( PyExc_AttributeError, "missing, empty or invalid new value" );
		}
		else{
		  int N;
			  // PyObject_Length() on a scalar will set an error (that the type doesn't have the len() method)
			  // so we clear any errors existing at this point
			PyErr_Clear();
			N= (int) PyInt_AsLong(newval);
			ret= 0;
			if( N> 0 && self->set->ncols!= N ){
				if( self->set->numPoints ){
					if( (self->set->columns= realloc_columns( self->set, N )) ){
						Check_Columns( self->set );
					}
					else{
						PyErr_NoMemory();
						ret= -1;
					}
				}
				else{
					self->set->ncols= N;
				}
			}
		}
	}
	return(ret);
}

#define SELFSET_FIELDVAL(field)	((ActiveWin && ActiveWin!=StubWindow_ptr)? ActiveWin->field[self->set->set_nr] : self->set->field)

#define SELFSET_FIELDSET(field,type,val)	{ type nval=(val); if(ActiveWin && ActiveWin!=StubWindow_ptr){ \
		ActiveWin->field[self->set->set_nr]= nval; \
		if( self->set->field != nval ){ \
			self->set->field = nval; \
			self->set->init_pass= True; \
		} \
	} \
	else{ \
		self->set->field = nval; \
	} }

static PyObject *PyDataSetObject_xcol( PyDataSetObject *self )
{
	return( PyInt_FromLong( (long) SELFSET_FIELDVAL(xcol) ) );
}

static int PyDataSetObject_new_xcol( PyDataSetObject *self, PyObject *newval )
{ int ret= -1;
	if( self && self->set ){
		if( newval==NULL || PyObject_Length(newval)== 0 || !(PyInt_Check(newval) || PyLong_Check(newval)) ){
			PyErr_SetString( PyExc_AttributeError, "missing, empty or invalid new value" );
		}
		else{
			  // PyObject_Length() on a scalar will set an error (that the type doesn't have the len() method)
			  // so we clear any errors existing at this point
			PyErr_Clear();
			SELFSET_FIELDSET(xcol, int, PyInt_AsLong(newval));
			ret= 0;
		}
	}
	return(ret);
}

static PyObject *PyDataSetObject_ycol( PyDataSetObject *self )
{
// 	return( PyInt_FromLong( (long) self->set->ycol ) );
	return( PyInt_FromLong( (long) SELFSET_FIELDVAL(ycol) ) );
}

static int PyDataSetObject_new_ycol( PyDataSetObject *self, PyObject *newval )
{ int ret= -1;
	if( self && self->set ){
		if( newval==NULL || PyObject_Length(newval)== 0 || !(PyInt_Check(newval) || PyLong_Check(newval)) ){
			PyErr_SetString( PyExc_AttributeError, "missing, empty or invalid new value" );
		}
		else{
			  // PyObject_Length() on a scalar will set an error (that the type doesn't have the len() method)
			  // so we clear any errors existing at this point
			PyErr_Clear();
// 			self->set->ycol= PyInt_AsLong(newval);
			SELFSET_FIELDSET(ycol, int, PyInt_AsLong(newval));
			ret= 0;
		}
	}
	return(ret);
}

static PyObject *PyDataSetObject_ecol( PyDataSetObject *self )
{
// 	return( PyInt_FromLong( (long) self->set->ecol ) );
	return( PyInt_FromLong( (long) SELFSET_FIELDVAL(ecol) ) );
}

static int PyDataSetObject_new_ecol( PyDataSetObject *self, PyObject *newval )
{ int ret= -1;
	if( self && self->set ){
		if( newval==NULL || PyObject_Length(newval)== 0 || !(PyInt_Check(newval) || PyLong_Check(newval)) ){
			PyErr_SetString( PyExc_AttributeError, "missing, empty or invalid new value" );
		}
		else{
			  // PyObject_Length() on a scalar will set an error (that the type doesn't have the len() method)
			  // so we clear any errors existing at this point
			PyErr_Clear();
// 			self->set->ecol= PyInt_AsLong(newval);
			SELFSET_FIELDSET(ecol, int, PyInt_AsLong(newval));
			ret= 0;
		}
	}
	return(ret);
}

static PyObject *PyDataSetObject_lcol( PyDataSetObject *self )
{
// 	return( PyInt_FromLong( (long) self->set->lcol ) );
	return( PyInt_FromLong( (long) SELFSET_FIELDVAL(lcol) ) );
}

static int PyDataSetObject_new_lcol( PyDataSetObject *self, PyObject *newval )
{ int ret= -1;
	if( self && self->set ){
		if( newval==NULL || PyObject_Length(newval)== 0 || !(PyInt_Check(newval) || PyLong_Check(newval)) ){
			PyErr_SetString( PyExc_AttributeError, "missing, empty or invalid new value" );
		}
		else{
			  // PyObject_Length() on a scalar will set an error (that the type doesn't have the len() method)
			  // so we clear any errors existing at this point
			PyErr_Clear();
// 			self->set->lcol= PyInt_AsLong(newval);
			SELFSET_FIELDSET(lcol, int, PyInt_AsLong(newval));
			ret= 0;
		}
	}
	return(ret);
}

static PyObject *PyDataSetObject_Ncol( PyDataSetObject *self )
{
 	return( PyInt_FromLong( (long) self->set->Ncol ) );
}

static int PyDataSetObject_new_Ncol( PyDataSetObject *self, PyObject *newval )
{ int ret= -1;
	if( self && self->set ){
		if( newval==NULL || PyObject_Length(newval)== 0 || !(PyInt_Check(newval) || PyLong_Check(newval)) ){
			PyErr_SetString( PyExc_AttributeError, "missing, empty or invalid new value" );
		}
		else{
		  int val;
			  // PyObject_Length() on a scalar will set an error (that the type doesn't have the len() method)
			  // so we clear any errors existing at this point
			PyErr_Clear();
 			val= PyInt_AsLong(newval);
			if( self->set->Ncol!= val ){
				self->set->Ncol= val;
#if ADVANCED_STATS == 2
				self->set->init_pass= True;
#endif
			}
			ret= 0;
		}
	}
	return(ret);
}

static PyObject *PyDataSetObject_draw( PyDataSetObject *self )
{
	return( PyBool_FromLong( (long) SELFSET_FIELDVAL(draw_set) ) );
}

static int PyDataSetObject_new_draw( PyDataSetObject *self, PyObject *newval )
{ int ret= -1;
	if( self && self->set ){
		if( newval==NULL || PyObject_Length(newval)== 0
			|| !(PyInt_Check(newval) || PyLong_Check(newval) || PyBool_Check(newval))
		){
			PyErr_SetString( PyExc_AttributeError, "missing, empty or invalid new value" );
		}
		else{
		  int val;
			  // PyObject_Length() on a scalar will set an error (that the type doesn't have the len() method)
			  // so we clear any errors existing at this point
			PyErr_Clear();
			val= PyInt_AsLong(newval);
			if( ActiveWin && ActiveWin != StubWindow_ptr ){
				if( ActiveWin->draw_set[self->set->set_nr]!= val ){
					ActiveWin->draw_set[self->set->set_nr]= val;
					ActiveWin->redraw= 1;
				}
			}
			self->set->draw_set= val;
			ret= 0;
		}
	}
	return(ret);
}

static PyObject *PyDataSetObject_ColumnLabels( PyDataSetObject *self )
{ PyObject *retr= PyDict_New();
  LabelsList *llist= NULL;
	if( self->set->ColumnLabels ){
		llist= self->set->ColumnLabels;
	}
	else if( ActiveWin ){
		llist= ActiveWin->ColumnLabels;
	}
	if( llist ){
		while( llist ){
			if( llist->label ){
			  PyObject *key, *item, *value;

				PyDict_SetItem( retr, PyInt_FromLong(llist->column), PyString_FromString(llist->label) );

				key= PyString_FromString(llist->label);
				value= PyInt_FromLong(llist->column);
				if( (item= PyDict_GetItem(retr, key)) ){
				  // if the label already exists in the dict, that means multiple columns have the same label.
				  // We handle that situation by storing a list of column numbers.
					if( PyList_Check(item) ){
					  // 3rd or more column with the same label: grow the list
					  // no need to alter the dict entry otherwise.
						PyList_Append(item, value );
					}
					else{
					  // 2nd column with the same label: the existing dict key value is thus an integer (column number)
					  // attempt to create a list and store the existing and the current column numbers.
					  PyObject *list;
						if( (list= PyList_New(0)) ){
							PyList_Append(list, item);
							PyList_Append(list, value);
							PyDict_SetItem( retr, key, list );
							  // reference counting for item should be handled for us.
						}
					}
				}
				else{
					PyDict_SetItem( retr, key, value );
				}
			}
			if( llist->min!= llist->max ){
				llist++;
			}
			else{
				llist= NULL;
			}
		}
	}
	return( retr );
}

static PyObject *PyDataSetObject_ColumnLabelsString( PyDataSetObject *self )
{ char *scl;
  PyObject *retr;
	if( self->set->ColumnLabels ){
		scl = ColumnLabelsString( self->set, -1, NULL, 0, 0,NULL );
		retr= PyString_FromString( (scl)? scl : "" );
		xfree(scl);
	}
	else{
		retr= PyString_FromString( "" );
	}
	return( retr );
}

static int PyDataSetObject_new_ColumnLabelsString( PyDataSetObject *self, PyObject *newval )
{ int ret= -1;
	if( self && self->set ){
		if( newval==NULL || PyObject_Length(newval)== 0 || (!PyBytes_Check(newval)
#ifdef IS_PY3K
				&& !PyUnicode_Check(newval)
#endif
			)
		){
			PyErr_SetString( PyExc_AttributeError, "missing, empty or invalid new value" );
		}
		else{
			  // PyObject_Length() on a scalar will set an error (that the type doesn't have the len() method)
			  // so we clear any errors existing at this point
			PyErr_Clear();
#ifdef IS_PY3K
			if( PyUnicode_Check(newval) ){
			  PyObject *bytes = NULL;
			  char *str = NULL;
				PYUNIC_TOSTRING( newval, bytes, str );
				if( str ){
					ColumnLabelsString( self->set, -1, str, 1, 0,NULL );
					ret = 0;
				}
				if( bytes ){
					Py_XDECREF(bytes);
				}
			}
			else
#endif
			{
				ColumnLabelsString( self->set, -1, PyBytes_AsString(newval), 1, 0,NULL );
				ret= 0;
			}
		}
	}
	return(ret);
}

/* 20061021: inevitably, a lot of this kind of code is around multiple times... */
double *PySequence_AsDoubleArray( PyObject *var, size_t *len )
{ size_t N;
  double *array= NULL;
	if( PyTuple_Check(var) || PyComplex_Check(var) || PyArray_Check(var) ){
PSAD_tuple:;
	  int i, type;
		if( PyTuple_Check(var) ){
			N= PyTuple_Size(var);
			for( i= 0; i< N; i++ ){
			  PyObject *el= PyTuple_GetItem(var, i);
				if( !(el && (PyInt_Check(el) || PyLong_Check(el) || PyFloat_Check(el))) ){
 					PyErr_SetString( XG_PythonError, "type clash: only tuples with scalar, numeric elements are supported" );
// 					PyErr_SetString(  PyExc_TypeError, "type clash: only tuples with scalar, numeric elements are supported" );
					return(NULL);
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
		if( !(array= (double*) malloc( N * sizeof(double) )) ){
			PyErr_NoMemory();
			return(NULL);
		}
		{ double value;
		  PyArrayObject* xd= NULL;
		  double *PyArrayBuf= NULL;
		  PyArrayIterObject *it;
			if( type== 2 ){
				if( (xd = (PyArrayObject*) PyArray_CopyFromObject( (PyObject*) var, PyArray_DOUBLE, 0, 0 )) ){
					PyArrayBuf= (double*)PyArray_DATA(xd); /* size would be N*sizeof(double) */
				}
				else{
					it= (PyArrayIterObject*) PyArray_IterNew(var);
				}
			}
			for( i= 0; i< N; i++ ){
				switch( type ){
					case 0:
						value= PyFloat_AsDouble( PyTuple_GetItem(var,i) );
						break;
					case 1:
						value= (i)? PyComplex_ImagAsDouble(var) : PyComplex_RealAsDouble(var);
						break;
					case 2:{
						if( PyArrayBuf ){
							value= PyArrayBuf[i];
						}
						else{
						  PyArrayObject *parray= (PyArrayObject*) var;
							if( it->index < it->size ){
								value= PyFloat_AsDouble( parray->descr->f->getitem( it->dataptr, var) );
								PyArray_ITER_NEXT(it);
							}
							else{
								set_NaN(value);
							}
						}
						break;
					}
					default:
						break;
				}
				array[i]= value;
			}
			if( type==2 ){
				if( xd ){
					Py_XDECREF(xd);
				}
				else{
					Py_DECREF(it);
				}
			}
		}
	}
	else if( PySequence_Check(var) ){
		if( !(var= PySequence_Fast(var, "attempt to convert non-sequence object to a tuple")) ){
 			PyErr_SetString( XG_PythonError, "unexpected failure converting sequence to tuple" );
// 			PyErr_SetString(  PyExc_RuntimeError, "unexpected failure converting sequence to tuple" );
			return(NULL);
		}
		if( PyList_Check(var) ){
			if( !(var= PyList_AsTuple(var)) ){
 				PyErr_SetString( XG_PythonError, "unexpected failure converting list to tuple" );
// 				PyErr_SetString(  PyExc_RuntimeError, "unexpected failure converting list to tuple" );
				return(NULL);
			}
		}
		if( PyTuple_Check(var) ){
			goto PSAD_tuple;
		}
	}
	if( array && len ){
		*len= N;
	}
	return( array );
}

static PyObject *PyDataSetObject_assoc( PyDataSetObject *self )
{ PyObject *retr= NULL;
  int Ndim=1;
  npy_intp dims[1];
	dims[0]= self->set->numAssociations;
	if( dims[0] && self->set->Associations ){
		if( (retr= PyArray_SimpleNewFromData( Ndim, &dims[0], PyArray_DOUBLE, (void*) self->set->Associations )) ){
			((PyArrayObject*)retr)->flags&= ~NPY_OWNDATA;
		}
		return(retr);
	}
	else{
		dims[0]= 0;
		if( (retr= PyArray_SimpleNewFromData( Ndim, &dims[0], PyArray_DOUBLE, (void*) NULL )) ){
			((PyArrayObject*)retr)->flags|= NPY_OWNDATA;
			return( retr );
		}
		else{
			Py_RETURN_NONE;
		}
	}
}

static int PyDataSetObject_new_assoc( PyDataSetObject *self, PyObject *newval )
{ int ret= -1;
	if( self && self->set ){
		if( newval==NULL || PyObject_Length(newval)== 0 ){
			PyErr_SetString( PyExc_AttributeError, "missing, empty or invalid new value" );
		}
		else{
		  size_t N;
		  double *assoc;
			  // PyObject_Length() on a scalar will set an error (that the type doesn't have the len() method)
			  // so we clear any errors existing at this point
			PyErr_Clear();
			if( (assoc= PySequence_AsDoubleArray( newval, &N )) ){
				xfree(self->set->Associations);
				self->set->Associations= assoc;
				self->set->numAssociations= N;
			}
			ret= 0;
		}
	}
	return(ret);
}

static PyObject *PyDataSetObject_link( PyDataSetObject *self )
{
	return( PyInt_FromLong( (long) self->set->set_link ) );
}

static int PyDataSetObject_new_link( PyDataSetObject *self, PyObject *newval )
{ int ret= -1;
	if( self && self->set ){
		if( newval==NULL || PyObject_Length(newval)== 0 || !(PyInt_Check(newval) || PyLong_Check(newval)) ){
			PyErr_SetString( PyExc_AttributeError, "missing, empty or invalid new value" );
		}
		else{
			  // PyObject_Length() on a scalar will set an error (that the type doesn't have the len() method)
			  // so we clear any errors existing at this point
			PyErr_Clear();
			LinkSet2( self->set, (int) PyInt_AsLong(newval) );
			ret= 0;
		}
	}
	return(ret);
}

static PyObject *PyDataSetObject_data( PyDataSetObject *self )
{ PyObject *retr= NULL, **columns= NULL;
  npy_intp dims[2];
  double **scs= self->set->columns;
	dims[0]= self->set->ncols;
	dims[1]= self->set->numPoints;
	if( (columns= (PyObject**) PyMem_New( PyObject*, self->set->ncols )) ){
	  int i;
		for( i= 0; i< self->set->ncols; i++ ){
			 // we return pointers to the actual columns[i] data, so we don't set NPY_OWNDATA to prevent Python from
			 // deallocating our data!
			if( !(columns[i]=
					PyArray_SimpleNewFromData( 1, &dims[1], PyArray_DOUBLE, (void*) ((scs)? scs[i] : NULL) ))
			){
				PyErr_NoMemory();
			}
		}
		if( (retr= PyArray_SimpleNewFromData( 1, &dims[0], PyArray_OBJECT, (void*) columns )) ){
		  // the 'toplevel' array "holding" columns is to be handled by python, so we label it as deallocatable:
		  // (note that columns was allocated by PyMem_New() )
			((PyArrayObject*)retr)->flags|= NPY_OWNDATA;
		}
		 // 20101019: a return statement had been missing here for months?!
		return( retr );
	}
	else{
		PyErr_NoMemory();
		return(NULL);
	}
}

static PyObject *PyDataSetObject_trdata( PyDataSetObject *self )
{ PyObject *retr= NULL, **columns= NULL;
  npy_intp dims[2];
	dims[0]= 4;
	dims[1]= self->set->numPoints;
	if( (columns= (PyObject**) PyMem_New( PyObject*, dims[0] )) ){
		if( !(columns[0]= PyArray_SimpleNewFromData( 1, &dims[1], PyArray_DOUBLE, (void*) self->set->xvec )) ){
			PyErr_NoMemory();
		}
		if( !(columns[1]= PyArray_SimpleNewFromData( 1, &dims[1], PyArray_DOUBLE, (void*) self->set->yvec )) ){
			PyErr_NoMemory();
		}
		if( !(columns[2]= PyArray_SimpleNewFromData( 1, &dims[1], PyArray_DOUBLE, (void*) self->set->errvec )) ){
			PyErr_NoMemory();
		}
		if( !(columns[3]= PyArray_SimpleNewFromData( 1, &dims[1], PyArray_DOUBLE, (void*) self->set->lvec )) ){
			PyErr_NoMemory();
		}
		if( (retr= PyArray_SimpleNewFromData( 1, &dims[0], PyArray_OBJECT, (void*) columns )) ){
			((PyArrayObject*)retr)->flags|= NPY_OWNDATA;
		}
		return retr;
	}
	else{
		PyErr_NoMemory();
		return(NULL);
	}

	return(retr);
}

static PyMethodDef PDO_methods[] = {
	{ NULL, NULL}
};

static PyGetSetDef PDO_getsetlist[] = {
	{ "set_nr", (getter)DO_long, NULL, "the set number" },
	{ "size", (getter)PyDataSetObject_size, NULL,
		"return the number of points in the set.\n"
	},
	{ "file",
		(getter)PyDataSetObject_filename, 
		(setter)PyDataSetObject_new_filename,
		"the set's name (*LEGEND*)"
	},
	{ "name",
		(getter)PyDataSetObject_name, 
		(setter)PyDataSetObject_new_name,
		"the set's name (*LEGEND*)"
	},
	{ "title",
		(getter)PyDataSetObject_title,
		(setter)PyDataSetObject_new_title,
		"the set's title (*TITLE*)"
	},
	{ "info",
		(getter)PyDataSetObject_set_info,
		(setter)PyDataSetObject_new_set_info,
		"the set's info string(s) (*SET_INFO*)"
	},
	{ "xlabel",
		(getter)PyDataSetObject_XUnits, 
		(setter)PyDataSetObject_new_XUnits,
		"the set's X label (*XLABEL*)"
	},
	{ "ylabel",
		(getter)PyDataSetObject_YUnits, 
		(setter)PyDataSetObject_new_YUnits,
		"the set's Y label (*YLABEL*)"
	},
	{ "ncols",
		(getter)PyDataSetObject_ncols, 
		(setter)PyDataSetObject_new_ncols,
		"the set's number of columns"
	},
	{ "xcol",
		(getter)PyDataSetObject_xcol, 
		(setter)PyDataSetObject_new_xcol,
		"the set's column with X data"
	},
	{ "ycol",
		(getter)PyDataSetObject_ycol, 
		(setter)PyDataSetObject_new_ycol,
		"the set's column with Y data"
	},
	{ "ecol",
		(getter)PyDataSetObject_ecol, 
		(setter)PyDataSetObject_new_ecol,
		"the set's column with Error/Orientation data"
	},
	{ "lcol",
		(getter)PyDataSetObject_lcol, 
		(setter)PyDataSetObject_new_lcol,
		"the set's column with vector length data"
	},
	{ "Ncol",
		(getter)PyDataSetObject_Ncol, 
		(setter)PyDataSetObject_new_Ncol,
		"the set's column with the number of observations per point"
	},
	{ "draw",
		(getter)PyDataSetObject_draw, 
		(setter)PyDataSetObject_new_draw,
		"whether the set is drawn"
	},
	{ "columnlabels",
		(getter)PyDataSetObject_ColumnLabels,
		(setter)NULL,
		"the set's (or globally defined) column labels, stored in a dict that has both the col.nr->label\n"
		" and label->col.nr mappings. If multiple columns share the same label, the dict will map the label\n"
		" to a list of the column numbers."
		/* NB: a tuple would be more correct (immutable), but a list is easier to implement. */
	},
	{ "columnlabelstring",
		(getter)PyDataSetObject_ColumnLabelsString,
		(setter)PyDataSetObject_new_ColumnLabelsString,
		"the set's column labels, stored in a string"
	},
	{ "data",
		(getter)PyDataSetObject_data,
		(setter)NULL,
		"the set's actual data"
	},
	{ "transformed",
		(getter)PyDataSetObject_trdata,
		(setter)NULL,
		"the set's transformed data - X,Y,Error and vector Length"
	},
	{ "associations",
		(getter)PyDataSetObject_assoc,
		(setter)PyDataSetObject_new_assoc,
		"the set's associated data (*ASSOCIATE*)"
	},
	{ "link",
		(getter)PyDataSetObject_link, 
		(setter)PyDataSetObject_new_link,
		"the set's column with X data"
	},
	/* redraw method? */
	{ NULL, NULL, NULL, NULL },
};

static PyNumberMethods DO_as_number= {
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
	(unaryfunc) DO_long,	/* nb_int */
#ifndef IS_PY3K
	(unaryfunc) DO_long,	/* nb_long */
#else
	NULL,				/* nb_reserved */
#endif
     (unaryfunc) DO_double,	/* nb_float;	*/
};

PyDoc_STRVAR(PyDataSetObject_Type__doc__,
	"External representation of a DataSet, the central internal data unit.\n"
);

PyTypeObject PyDataSetObject_Type = {
	PyVarObject_HEAD_INIT(NULL,0)
	"PyDataSetObject",			/*tp_name*/
	sizeof(PyDataSetObject),   	/*tp_basicsize*/
	0,						/*tp_itemsize*/
	/* methods */
	(destructor)PyDataSetObject_dealloc, /*tp_dealloc*/
	(printfunc)0,				/*tp_print*/
	(getattrfunc)0,   			/*tp_getattr*/
	(setattrfunc)0,   			/*tp_setattr*/
#ifndef IS_PY3K
	(cmpfunc)0,   				/*tp_compare*/
#else
	NULL,					/*tp_reserved*/
#endif
	(reprfunc) DO_repr,			/*tp_repr*/
	&DO_as_number,				/*tp_as_number*/
	0,						/*tp_as_sequence*/
	0,						/*tp_as_mapping*/
	(hashfunc)0,				/*tp_hash*/
	(ternaryfunc) 0,   			/*tp_call*/
	(reprfunc)DO_str,			/*tp_str*/

	/* Space for future expansion */
	0L,0L,0L,0L,
	PyDataSetObject_Type__doc__,	/* Documentation string */
	0, 0, 0, 0, 0, 0,
	PDO_methods,
	0,
	PDO_getsetlist,
};

PyObject* python_DataSet ( PyObject *self, PyObject *args, PyObject *kw )
{ int setnr= (int) *ascanf_setNumber, maxid= (StartUp)? setNumber+1 : setNumber;
  char *kws[]= { "setnr", NULL };
  PyObject *retr= NULL;

	CHECK_INTERRUPTED();

	if( !PyArg_ParseTupleAndKeywords(args, kw, "|i:DataSet", kws, &setnr ) ){
		return NULL;
	}
	if( setnr< 0 || setnr>= maxid || !AllSets ){
 		PyErr_SetString( XG_PythonError, "setNumber negative or larger than the currently defined sets" );
// 		PyErr_SetString(  PyExc_ValueError, "setNumber negative or larger than the currently defined sets" );
	}
	else{
		if( !(retr= PyDataSetObject_FromDataSet( &AllSets[setnr] )) ){
			PyErr_NoMemory();
		}
	}
	return( retr );
}

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

int init_DataSet_module()
{
	if( !initialised ){
	  int ok;
		  /* Do some importing of functionality necessary to use numpy's arrays: */
		call_import_array(&ok);

		if( PyType_Ready(&PyDataSetObject_Type) < 0 ){
			return 0;
		}

		initialised= True;
	}
	return(1);
}
