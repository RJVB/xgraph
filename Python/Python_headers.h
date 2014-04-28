#ifndef _PYTHON_HEADERS_H

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#ifdef __APPLE__
#	ifdef RJVB
#		ifdef PYTHON25
#			include <Python2.5/Python.h>
#			include "python25_numpy.h"
#		elif PYTHON24
#			include <Python2.4/Python.h>
#			include "python24_numpy.h"
#		elif PYTHON26
#			include <Python2.6/Python.h>
#			include <Python2.6/bytesobject.h>
#			include <Python2.6/intobject.h>
#			include "python26_numpy.h"
#		elif PYTHON27
#			include <Python2.7/Python.h>
#			include <Python2.7/bytesobject.h>
#			include <Python2.7/intobject.h>
#			include "python27_numpy.h"
#		elif PYTHON32
#			include <Python3.2/Python.h>
#			include "python32_numpy.h"
#		elif PYTHON33
#			include <Python3.3/Python.h>
#			include "python33_numpy.h"
#		elif PYTHONsys
#			include <Python/Python.h>
#			include <Python/bytesobject.h>
#			if PY_MAJOR_VERSION < 3
#				include <Python/intobject.h>
#			endif
#			include "pythonsys_numpy.h"
#		elif PYTHONdefault
#			include <Python/Python.h>
#			include <Python/bytesobject.h>
#			if PY_MAJOR_VERSION < 3
#				include <Python/intobject.h>
#			endif
#			include "pythondefault_numpy.h"
#		else
#			include <Python/Python.h>
#			include <Python/bytesobject.h>
#			include "python23_numpy.h"
#		endif
#	else
#		include <Python/Python.h>
#		include <Python/bytesobject.h>
#		ifdef PYTHON25
#			include "python25_numpy.h"
#		elif PYTHON24
#			include "python24_numpy.h"
#		elif PYTHON26
#			include "python26_numpy.h"
#		elif PYTHON27
#			include "python27_numpy.h"
#		else
//#			include <Python/../../lib/python2.3/site-packages/numpy/core/include/numpy/arrayobject.h>
#			include "pythonsys_numpy.h"
#			ifndef Py_RETURN_NONE
#				define Py_RETURN_NONE return Py_INCREF(Py_None), Py_None
#			endif
#		endif
#	endif
	  /* This module better not depend on the Accelerate framework...! */
#	define __ACCELERATE__
#elif defined(__CYGWIN__) || defined(linux)
#	ifdef PYTHON25
#		include "python25_headers.h"
#		include "python25_numpy.h"
#	elif PYTHON26
#		include "python26_headers.h"
#		include "python26_numpy.h"
#	elif PYTHON27
#		include "python27_headers.h"
#		include "python27_numpy.h"
#	elif PYTHON31
#		include "python31_headers.h"
#		include "python31_numpy.h"
#	elif PYTHON32
//#		if defined(__CYGWIN__)
//#			include <python3.2m/Python.h>
//#			include <python3.2m/bytesobject.h>
//#		else
//#			include <python3.2/Python.h>
//#			include <python3.2/bytesobject.h>
//#		endif
#		include "python32_headers.h"
#		include "python32_numpy.h"
#	elif PYTHON33
//#		if defined(__CYGWIN__)
//#			include <python3.3m/Python.h>
//#			include <python3.3m/bytesobject.h>
//#		else
//#			include <python3.3/Python.h>
//#			include <python3.3/bytesobject.h>
//#		endif
#		include "python33_headers.h"
#		include "python33_numpy.h"
#	elif PYTHONsys
#		include "pythonsys_headers.h"
#		include "pythonsys_numpy.h"
#	elif PYTHONdefault
#		include "pythondefault_headers.h"
#		include "pythondefault_numpy.h"
#	endif
#else
#	include <Python.h>
#	include <bytesobject.h>
#	if PY_MAJOR_VERSION < 3
#		include <intobject.h>
#	endif
	  /* good question ... where are the numpy headers here?? */
#	include "python_numpy.h"
#endif

#if !defined(Py_RETURN_NONE)	// Python 2.3??
#	define Py_RETURN_NONE return Py_INCREF(Py_None), Py_None
#	define Py_EnterRecursiveCall(s)	0
#	define Py_LeaveRecursiveCall()	/**/
#endif

#if PY_MAJOR_VERSION >= 3
#	define IS_PY3K
#	define PyInt_Check		PyLong_Check
#	define PyInt_FromLong	PyLong_FromLong
#	define PyInt_AsLong		PyLong_AsLong
#	define PyFile_FromFile(fp,name,mode,closeit)	PyFile_FromFd(fileno((fp)),(name),(mode),-1,NULL,NULL,NULL,(closeit))
#	define PYUNIC_TOSTRING(arg,tmpObject,str)	\
				if( (tmpObject = PyUnicode_AsLatin1String((arg))) || PyUnicode_FSConverter( (arg), &(tmpObject) ) ){	\
					if( PyBytes_Check((tmpObject)) ){	\
						(str) = PyBytes_AsString((tmpObject));	\
					}	\
					else{	\
						(str)  = PyByteArray_AsString((tmpObject));	\
					}	\
				}	\
				else{	\
					PyErr_SetString( XG_PythonError, "unexpected failure converting unicode object" );	\
				}
	extern PyObject *PyString_FromString(const char *str);
#endif

#ifndef NPY_1_7_API_VERSION
#	define NPY_ARRAY_OWNDATA	NPY_OWNDATA
#	define PyArray_ENABLEFLAGS(o,f)	(PyArray_FLAGS((o)) |= (f))
#	define PyArray_CLEARFLAGS(o,f)	(PyArray_FLAGS((o)) &= ~(f))
#else
#	define NPY_OWNDATA	NPY_ARRAY_OWNDATA
#	define PyArray_DOUBLE	NPY_FLOAT64
#	define PyArray_INT		NPY_INT
#	define PyArray_LONG		NPY_LONG
#	define PyArray_OBJECT	NPY_OBJECT
#endif

#define _PYTHON_HEADERS_H
#endif
