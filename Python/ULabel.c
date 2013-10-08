#include "config.h"
IDENTIFY( "library module for exporting UserLabels to Python" );

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


static UserLabel *_GetULabelNr( LocalWin *wi, int nr)
{  UserLabel *ret= NULL;
	if( nr>= 0 && nr< wi->ulabels ){
	  UserLabel *ul= wi->ulabel;
		while( ul && nr>= 0 ){
			nr-= 1;
			ret= ul;
			ul= ul->next;
		}
	}
	return( ret );
}

PyObject *python_GetULabels( PyObject *self, PyObject *args )
{ PyObject *retr= PyDict_New();
  PyObject *ULabel= PyList_New(0), *Index= PyDict_New(), *Linked2= PyDict_New(), *Type= PyDict_New();
  LocalWin *wi= (ActiveWin)? ActiveWin : StubWindow_ptr;
	if( wi && wi->ulabels ){
	  int idx;
		for( idx= 0; idx< wi->ulabels; idx++ ){
		  UserLabel *ul;
		  PyObject *UL= PyDict_New();
			ul= _GetULabelNr( wi, idx );
			if( !ul ){
				fprintf( stderr, " (python_GetULabels(%d): unexpectedly couldn't get that label (%s)) ",
					idx, serror()
				);
			}
			else{
			  char *ltext= ((ul->labelbuf)? ul->labelbuf : ul->label);
			  PyObject *typeN, *linked2;
				PyDict_SetItemString( UL, "text", PyString_FromString( ltext ) );
				PyDict_SetItemString( UL, "start", Py_BuildValue( "[dd]", ul->x1, ul->y1 ) );
				PyDict_SetItemString( UL, "end", Py_BuildValue( "[dd]", ul->x2, ul->y2 ) );
				PyDict_SetItemString( UL, "linked2", (linked2= PyInt_FromLong( ul->set_link )) );
				if( ul->type>= UL_regular && ul->type< UL_types ){
					PyDict_SetItemString( UL, "type", (typeN= PyString_FromString( ULabelTypeNames[ul->type])) );
				}
				else{
					PyDict_SetItemString( UL, "type", (typeN= PyString_FromString("RL")) );
				}
				{ int type;
					PyDict_SetItemString( UL, "colour", PyString_FromString( ULabel_pixelCName( ul, &type ) ) );
					PyDict_SetItemString( UL, "clinked", PyInt_FromLong( (type==-1)? True : False ) );
				}

				PyList_Append( ULabel, UL );
				{ PyObject *elem= PyDict_GetItemString(Index, ltext);
					if( !elem ){
						PyDict_SetItemString( Index, ltext, PyInt_FromLong(idx) );
					}
					else if( PyInt_Check(elem) || PyLong_Check(elem) ){
					  PyObject *IDList= PyList_New(2);
						PyList_SET_ITEM(IDList, 0, elem);
						PyList_SET_ITEM( IDList, 1, PyInt_FromLong(idx) );
						PyDict_SetItemString( Index, ltext, IDList );
					}
					else if( PyList_Check(elem) ){
						PyList_Append(elem, PyInt_FromLong(idx) );
					}
					else{
						PyErr_SetString( XG_PythonError,
							"internal inconsistency in " __FILE__ ": index entry not an int nor a list" );
						return(NULL);
					}
				}
				{ PyObject *elem= PyDict_GetItem(Linked2, linked2);
					if( !elem ){
						PyDict_SetItem( Linked2, linked2, PyInt_FromLong(idx) );
					}
					else if( PyInt_Check(elem) || PyLong_Check(elem) ){
					  PyObject *IDList= PyList_New(2);
						PyList_SET_ITEM(IDList, 0, elem);
						PyList_SET_ITEM( IDList, 1, PyInt_FromLong(idx) );
						PyDict_SetItem( Linked2, linked2, IDList );
					}
					else if( PyList_Check(elem) ){
						PyList_Append(elem, PyInt_FromLong(idx) );
					}
					else{
						PyErr_SetString( XG_PythonError,
							"internal inconsistency in " __FILE__ ": linked2 entry not an int nor a list" );
						return(NULL);
					}
				}
				{ PyObject *elem= PyDict_GetItem(Type, typeN);
					if( !elem ){
						PyDict_SetItem( Type, typeN, PyInt_FromLong(idx) );
					}
					else if( PyInt_Check(elem) || PyLong_Check(elem) ){
					  PyObject *IDList= PyList_New(2);
						PyList_SET_ITEM(IDList, 0, elem);
						PyList_SET_ITEM( IDList, 1, PyInt_FromLong(idx) );
						PyDict_SetItem( Type, typeN, IDList );
					}
					else if( PyList_Check(elem) ){
						PyList_Append(elem, PyInt_FromLong(idx) );
					}
					else{
						PyErr_SetString( XG_PythonError,
							"internal inconsistency in " __FILE__ ": type entry not an int nor a list" );
						return(NULL);
					}
				}
			}
		}
		PyDict_SetItemString( retr, "count", PyInt_FromLong(wi->ulabels) );
	}
	else{
		PyDict_SetItemString( retr, "count", PyInt_FromLong(0) );
	}
	PyDict_SetItemString( retr, "label", ULabel );
	PyDict_SetItemString( retr, "index", Index );
	PyDict_SetItemString( retr, "linked2", Linked2 );
	PyDict_SetItemString( retr, "type", Type );

	return( retr );
}

static int initialised= 0;

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

int init_ULabel_module()
{
	if( !initialised ){
	  int ok;
		  /* Do some importing of functionality necessary to use numpy's arrays: */
		call_import_array(&ok);

		initialised= True;
	}
	return(1);
}
