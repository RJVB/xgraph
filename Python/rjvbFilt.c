/*
 * Python interface to filtering routines
 \ (c) 2005-2010 R.J.V. Bertin
 */
 
/*
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

*/

#ifdef __CYGWIN__
#	undef _WINDOWS
#	undef WIN32
#	undef MS_WINDOWS
#	undef _MSC_VER
#endif
#if defined(_WINDOWS) || defined(WIN32) || defined(MS_WINDOWS) || defined(_MSC_VER)
#	define MS_WINDOWS
#	define _USE_MATH_DEFINES
#endif

#include "Python2or3Header.h"
#include "Py_InitModule.h"
#if PY_MAJOR_VERSION >= 2
#	ifndef MS_WINDOWS
#		include <numpy/arrayobject.h>
#	else
// #		include <../lib/site-packages/numpy/core/include/numpy/arrayobject.h>
#		include <numpy/arrayobject.h>
#	endif // MS_WINDOWS
#else
#	error "not yet configured for this Python version"
#endif

#if defined(__GNUC__) && !defined(_GNU_SOURCE)
#	define _GNU_SOURCE
#endif

#include <stdio.h>
#include <math.h>
#include "NaN.h"

#include <errno.h>

#include "rjvbFilters.h"

#ifndef False
#	define False	0
#endif
#ifndef True
#	define True	1
#endif

#ifndef StdErr
#	define StdErr	stderr
#endif

#ifndef CLIP
#	define CLIP(var,low,high)	if((var)<(low)){\
	(var)=(low);\
}else if((var)>(high)){\
	(var)=(high);}
#endif
#define CLIP_EXPR(var,expr,low,high)	{ double l, h; if(((var)=(expr))<(l=(low))){\
	(var)=l;\
}else if((var)>(h=(high))){\
	(var)=h;}}

PyObject *FMError;

typedef enum { ContArray=1, Array=2, Tuple=3 } PSTypes;

typedef struct ParsedSequences{
	PSTypes type;
	double *array;
	npy_intp N, dealloc_array;
	PyArrayObject *ContArray;
} ParsedSequences;

static PyObject *ParseSequence( PyObject *var, ParsedSequences *pseq )
{ PyObject *ret;
	if( PyArray_Check(var) ){
	  PyArrayObject* xd= (PyArrayObject*) PyArray_ContiguousFromObject( (PyObject*) var, PyArray_DOUBLE, 0, 0 );
	  PyArrayIterObject *it;
		pseq->N = PyArray_Size(var);
		if( xd ){
			pseq->array = (double*) PyArray_DATA(xd);
			pseq->dealloc_array = False;
			pseq->type = ContArray;
			  // for decref'ing after finishing with the data operations:
			pseq->ContArray = xd;
			ret = var;
		}
		else{
		  int i, ok= True;
		  PyArrayObject *parray= (PyArrayObject*) var;
			pseq->ContArray = NULL;
			if( !(pseq->array = PyMem_New( double, pseq->N )) ){
				PyErr_NoMemory();
				return(NULL);
			}
			if( !(it= (PyArrayIterObject*) PyArray_IterNew(var)) ){
				PyMem_Free(pseq->array);
				return(NULL);
			}
			PyErr_Clear();
			for( i= 0; ok && i< pseq->N; i++ ){
				if( it->index < it->size ){
				  PyObject *elem= parray->descr->f->getitem( it->dataptr, var);
					if( PyInt_Check(elem) || PyLong_Check(elem) || PyFloat_Check(elem) ){
						pseq->array[i] = PyFloat_AsDouble(elem);
						PyArray_ITER_NEXT(it);
					}
					else{
						PyErr_SetString( FMError, "type clash: only arrays with scalar, numeric elements are supported" );
						ok = False;
					}
				}
			}
			Py_DECREF(it);
			if( ok ){
				pseq->dealloc_array = True;
				pseq->type = Array;
				ret = var;
			}
			else{
				PyMem_Free(pseq->array);
				pseq->array = NULL;
				ret = NULL;
			}
		}
	}
	else if( PyList_Check(var) ){
		if( !(var= PyList_AsTuple(var)) ){
 			PyErr_SetString( FMError, "unexpected failure converting list to tuple" );
			return(NULL);
		}
		else{
			goto handleTuple;
		}
	}
	else if( PyTuple_Check(var) ){
handleTuple:;
	  int i, ok= True;
		pseq->N = PyTuple_Size(var);
		if( !(pseq->array = PyMem_New( double, pseq->N )) ){
			PyErr_NoMemory();
			return(NULL);
		}
		for( i= 0; ok && i< pseq->N; i++ ){
		  PyObject *el= PyTuple_GetItem(var, i);
			if( (el && (PyInt_Check(el) || PyLong_Check(el) || PyFloat_Check(el))) ){
				pseq->array[i] = PyFloat_AsDouble(el);
			}
			else{
				PyErr_SetString( FMError, "type clash: only tuples with scalar, numeric elements are supported" );
				ok = False;
			}
		}
		if( ok ){
			pseq->dealloc_array = True;
			pseq->type = Tuple;
			ret = var;
		}
		else{
			PyMem_Free(pseq->array);
			pseq->array = NULL;
			ret = NULL;
		}
	}
	else{
		PyErr_SetString( FMError, "sequence must be a numpy.ndarray, tuple or list" );
		ret = NULL;
	}
	return(ret);
}

static PyObject *python_convolve( PyObject *self, PyObject *args )
{ PyObject *dataArg, *maskArg, *ret= NULL;
  ParsedSequences dataSeq, maskSeq;
  int nan_handling = True;
	
	if(!PyArg_ParseTuple(args, "OO|i:convolve", &dataArg, &maskArg, &nan_handling )){
		return NULL;
	}
	dataSeq.type = maskSeq.type = 0;
	if( !ParseSequence( dataArg, &dataSeq ) ){
		if( PyErr_Occurred() ){
			PyErr_Print();
		}
		PyErr_SetString( FMError, "Error occurred while parsing data argument" );
	}
	if( !ParseSequence( maskArg, &maskSeq ) ){
		if( PyErr_Occurred() ){
			PyErr_Print();
		}
		PyErr_SetString( FMError, "Error occurred while parsing mask argument" );
	}
	if( dataSeq.type && maskSeq.type ){
	  double *output;
		output = convolve( dataSeq.array, dataSeq.N, maskSeq.array, maskSeq.N, nan_handling );
		if( output ){
		  npy_intp dim[1]= {dataSeq.N};
			ret= PyArray_SimpleNewFromData( 1, dim, PyArray_DOUBLE, (void*) output );
			((PyArrayObject*)ret)->flags|= NPY_OWNDATA;
		}
	}
	if( dataSeq.type ){
		if( dataSeq.dealloc_array ){
			PyMem_Free(dataSeq.array);
		}
		if( dataSeq.type == ContArray ){
			Py_XDECREF(dataSeq.ContArray);
		}
	}
	if( maskSeq.type ){
		if( maskSeq.dealloc_array ){
			PyMem_Free(maskSeq.array);
		}
		if( maskSeq.type == ContArray ){
			Py_XDECREF(maskSeq.ContArray);
		}
	}
	return ret;
}

static PyObject *python_SavGolayCoeffs( PyObject *self, PyObject *args, PyObject *kw )
{ PyObject *ret = NULL;
  int fw, fo, deriv = 0;
  char *kws[] = { "halfwidth", "order", "deriv", NULL };
  unsigned long N;
  double *coeffs, *output;
	
	if(!PyArg_ParseTupleAndKeywords(args, kw, "ii|i:SavGolayCoeffs", kws, &fw, &fo, &deriv )){
		return NULL;
	}
	if( fw < 0 || fw > (MAXINT-1)/2 ){
		PyErr_Warn( FMError, "halfwidth must be positive and <= (MAXINT-1)/2; clipping" );
		CLIP( fw, 0, (MAXINT-1)/2 );
	}
	if( fo < 0 || fo > 2*fw ){
		PyErr_Warn( FMError, "order must be positive and <= the filter width" );
		  /* Put arbitrary upper limit on the order of the smoothing polynomial	*/
		CLIP( fo, 0, 2* fw );
	}
	if( deriv < -fo || deriv > fo ){
		PyErr_Warn( FMError, "derivative must be between -order and +order" );
		CLIP( deriv, -fo, fo );
	}
	N = fw*2+3;
	errno = 0;
	if( (coeffs= (double*) PyMem_New(double, (N+ 1) ))
	    && (output= (double*) PyMem_New(double, (N+ 1) ))
	){
	  int i;
	  npy_intp dim[1]= {N};
		if( !(fw== 0 && fo== 0 && deriv==0) ){
			if( savgol( &(coeffs)[-1], N, fw, fw, deriv, fo ) ){
				  /* Unwrap the coefficients into the target memory:	*/
				output[N/2] = coeffs[0];
				for( i = 1; i <= N/2; i++ ){
					output[N/2-i]= coeffs[i];
					output[N/2+i]= coeffs[N-i];
				}
			}
			else{
				PyMem_Free(output);
				output = NULL;
			}
		}
		else{
			memset( output, 0, N* sizeof(double) );
		}
		PyMem_Free(coeffs);
		if( output ){
			ret = PyArray_SimpleNewFromData( 1, dim, PyArray_DOUBLE, (void*) output );
			((PyArrayObject*)ret)->flags|= NPY_OWNDATA;
		}
	}
	else{
		PyErr_NoMemory();
	}
	return( ret );
}

static PyObject *python_SavGolayGain( PyObject *self, PyObject *args, PyObject *kw )
{ int deriv = 0;
  double delta;
  char *kws[] = { "delta", "deriv", NULL };
	
	if(!PyArg_ParseTupleAndKeywords(args, kw, "di|i:SavGolayGain", kws, &delta, &deriv )){
		return NULL;
	}
	return( Py_BuildValue("d", ((deriv>0)? pow(delta,deriv)/deriv : 1) ) );
}
	
static PyMethodDef Filt_methods[] =
{
	{ "convolve", (PyCFunction) python_convolve, METH_VARARGS,
		"convolve(Data,Mask[,nan_handling]): convolve the array pointed to by <Data> by <Mask>\n"
		" The result is returned in a numpy.ndarray equal in size to Data\n"
		" Data and Mask must be 1D homogenous, numerical sequences (numpy.ndarray, tuple, list)\n"
		" The <nan_handling> argument indicates what to do with gaps of NaN value(s) in the input:\n"
		" \t1: if possible, pad with the values surrounding the gap (step halfway)\n"
		" \t2: if possible, intrapolate linearly between the surrounding values\n"
		" \t(if not possible, simply pad with the first or last non-NaN value).\n"
		" The estimated values are replaced by the original NaNs after convolution.\n"
		" This routine uses direct (\"brain dead\") convolution.\n"
	},
	{ "SavGolayCoeffs", (PyCFunction) python_SavGolayCoeffs, METH_VARARGS|METH_KEYWORDS,
		"SavGolayCoeffs(halfwidth,order[,deriv=0]]): determine the coefficients\n"
		" for a Savitzky-Golay convolution filter. This returns a mask that can be used for\n"
		" convolution; the wider the filter, the more it smooths using a polynomial of the\n"
		" requested order (the higher the orde, the closer it will follow the input data). The\n"
		" deriv argument specifies an optional derivative that will be calculated during\n"
		" the smoothing; this is generally better than smoothing first and taking the derivative afterwards.\n"
		" After the convolution, the output is scaled by a gain factor; see SavGolayGain.\n"
	},
	{ "SavGolayGain", (PyCFunction) python_SavGolayGain, METH_VARARGS|METH_KEYWORDS,
		"SavGolayGain(delta,deriv): returns the gain factor by which the result of a convolution\n"
		" with a SavGolay mask has to be multiplied (1 for deriv==0). delta is the resolution of\n"
		" the dimension with respect to which the derivative is being taken, e.g. the sampling time interval.\n"
	},
	{ NULL, NULL }
};

#if defined(DL_EXPORT)
DL_EXPORT(void) initrjvbFilt(void)
#elif defined(IS_PY3K)
PyObject *PyInit_rjvbFilt(void)
#else
void initrjvbFilt(void)
#endif
{ PyObject *mod, *dict;

	import_array();

	mod=Py_InitModule("rjvbFilt", Filt_methods);

	FMError= PyErr_NewException( "rjvbFilt.error", NULL, NULL );
	Py_XINCREF(FMError);
	PyModule_AddObject( mod, "error", FMError );
	if( PyErr_Occurred() ){
		PyErr_Print();
	}

	dict=PyModule_GetDict(mod);

#ifdef IS_PY3K
	return mod;
#endif
}
