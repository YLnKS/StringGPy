#ifndef PY_UTILITIES_H
#define PY_UTILITIES_H

#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL cStringGPy_ARRAY_API

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_9_API_VERSION
#include <numpy/arrayobject.h>
#include <math.h>
#include <numpy/npy_common.h>

/* ==== Check that PyArrayObject is an int type and a vector ==============
    return 1 if an error and raise exception */ 
int  not_intvector(PyArrayObject *vec);

/* ==== Check that PyArrayObject is a double (Float) type and a vector ==============
    return 1 if an error and raise exception */ 
int  not_doublevector(PyArrayObject *vec);

/* ==== Check that PyArrayObject is a double (Float) type and a matrix ==============
    return 1 if an error and raise exception */ 
int  not_doublematrix(PyArrayObject *mat);

/* Avoids memory leaks when using PyArray_AsCArray with the only reference to an existing 
	Python object. */
int PyArray_AsCArray_Safe(PyObject **op, void *ptr, npy_intp *dims, int nd,
    PyArray_Descr* typedescr);
#endif