#include "py_utilities.h"

/* ==== Check that PyArrayObject is an int type and a vector ==============
    return 1 if an error and raise exception */ 
int  not_intvector(PyArrayObject *vec)  {
    if (((PyArray_TYPE(vec) != NPY_INT) & (PyArray_TYPE(vec) != NPY_INT8) & 
			(PyArray_TYPE(vec) != NPY_INT16) & (PyArray_TYPE(vec) != NPY_INT32) & 
			(PyArray_TYPE(vec) != NPY_INT64)) || PyArray_NDIM(vec) != 1)  {
        PyErr_SetString(PyExc_ValueError,
            "In not_intvector: array must be of type Int and 1 dimensional (n).");
        return 1;  }
    return 0;
}

/* ==== Check that PyArrayObject is a double (Float) type and a vector ==============
    return 1 if an error and raise exception */ 
int  not_doublevector(PyArrayObject *vec)  {
    if (PyArray_TYPE(vec) != NPY_DOUBLE || PyArray_NDIM(vec) != 1)  {
        PyErr_SetString(PyExc_ValueError,
            "In not_doublevector: array must be of type Float and 1 dimensional (n).");
        return 1;  }
    return 0;
}

/* ==== Check that PyArrayObject is a double (Float) type and a matrix ==============
    return 1 if an error and raise exception */ 
int  not_doublematrix(PyArrayObject *mat)  {
    if (PyArray_TYPE(mat) != NPY_DOUBLE || PyArray_NDIM(mat) != 2)  {
        PyErr_SetString(PyExc_ValueError,
            "In not_doublematrix: array must be of type Float and 2 dimensional (n x m).");
        return 1;  }
    return 0;
}

/* Avoids memory leaks when using PyArray_AsCArray with the only reference to an existing 
	Python object. */
int PyArray_AsCArray_Safe(PyObject **op, void *ptr, npy_intp *dims, int nd, PyArray_Descr* typedescr){
	PyObject *cleaning_copy = *op;
	int status=0;
	// Both *op and cleaning copy point to the same PyObject.
	status=PyArray_AsCArray(op, ptr, dims, nd, typedescr);
	// *Op now points to a new PyObject (though a copy of the previous), whose 
	//	data ptr points to.
	
	// We need to properly decrement the reference count of the object *op previously
	//	pointed to.
	Py_CLEAR(cleaning_copy);
	return status;
}

