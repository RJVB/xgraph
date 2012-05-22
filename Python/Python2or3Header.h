#ifndef _PYTHON_HEADER_H

#include <Python.h>
#include <bytesobject.h>

#if PY_MAJOR_VERSION >= 3
#	define IS_PY3K
#	define PyInt_FromLong	PyLong_FromLong
#else
#	include <intobject.h>
#endif

#define _PYTHON_HEADER_H
#endif
