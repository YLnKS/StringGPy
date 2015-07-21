#define PY_ARRAY_UNIQUE_SYMBOL cStringGPy_ARRAY_API

#include <Python.h>
#include <stdio.h>
#include <math.h>
#define NPY_NO_DEPRECATED_API NPY_1_9_API_VERSION
#include <numpy/arrayobject.h>
#include <numpy/npy_common.h>
#include <numpy/ufuncobject.h>
#include <assert.h>
#include <time.h>
#include "lin_alg_utilities.h"
#include "py_utilities.h"
#include "kernel_utilities.h"
#include "py_imports.h"
#include "sampl_utilities.h"

/* ==== Compute the covariance matrix of a Spectral Mixture kernel. ==== */
static PyObject *cStringGPy_sm_cov(PyObject *self, PyObject *args) {
	PyArrayObject *X1=NULL, *X2=NULL, *theta=NULL, *result=NULL;
    double *c_X1, *c_X2, *c_theta, **c_result;

    if (!PyArg_ParseTuple(args, "O!O!O!", &PyArray_Type, &X1, &PyArray_Type, &X2,\
    	&PyArray_Type, &theta)) return NULL;

    if (not_doublevector(X1)) return NULL;
    if (not_doublevector(X2)) return NULL;
    if (not_doublevector(theta)) return NULL;

     /* Make the matrix of results */
    int n_mixt,dims[2];
    dims[0]=PyArray_DIMS(X1)[0];
    dims[1]=PyArray_DIMS(X2)[0];
    n_mixt=PyArray_DIMS(theta)[0]/3;
    result=(PyArrayObject *) PyArray_FromDims(2, dims, NPY_DOUBLE);

	/* Retrieve C pointers to the data for ease of indexing */
	PyArray_AsCArray((PyObject **)&X1, &c_X1, PyArray_DIMS((PyArrayObject *)X1), 1, PyArray_DescrFromType(NPY_DOUBLE));	
	PyArray_AsCArray((PyObject **)&X2, &c_X2, PyArray_DIMS((PyArrayObject *)X2), 1, PyArray_DescrFromType(NPY_DOUBLE));
	PyArray_AsCArray((PyObject **)&theta, &c_theta, PyArray_DIMS((PyArrayObject *)theta), 1, PyArray_DescrFromType(NPY_DOUBLE));
	PyArray_AsCArray((PyObject **)&result, &c_result, PyArray_DIMS((PyArrayObject *)result), 2, PyArray_DescrFromType(NPY_DOUBLE));

	/* Compute the covariance matrix in place */
	sm_cov(c_result, c_X1, dims[0], c_X2, dims[1], c_theta, n_mixt);

    /* Cleaning up */
	PyArray_Free((PyObject *)X1, c_X1);
	PyArray_Free((PyObject *)X2, c_X2);
	PyArray_Free((PyObject *)theta, c_theta);

    return PyArray_Return(result);
}


/* ==== Compute the covariance matrix of a given kernel. ==== */
static PyObject *cStringGPy_cov(PyObject *self, PyObject *args) {
	PyArrayObject *X1=NULL, *X2=NULL, *theta=NULL, *result=NULL;
    double *c_X1, *c_X2, *c_theta, **c_result;
	char *type;

    if (!PyArg_ParseTuple(args, "O!O!O!s", &PyArray_Type, &X1, &PyArray_Type, &X2,\
    	&PyArray_Type, &theta, &type)) return NULL;

    if (not_doublevector(X1)) return NULL;
    if (not_doublevector(X2)) return NULL;
    if (not_doublevector(theta)) return NULL;

     /* Make the matrix of results */
    int n_mixt,dims[2];
    dims[0]=PyArray_DIMS(X1)[0];
    dims[1]=PyArray_DIMS(X2)[0];
	if((strcmp(type, "srq") == 0) | (((NULL != strstr(type, "_ma12")) & !(strcmp(type, "srq_ma12") == 0)))){
		n_mixt=PyArray_DIMS(theta)[0]/4;
	}else if (strcmp(type, "srq_ma12") == 0){
		n_mixt=PyArray_DIMS(theta)[0]/5;
	}else{
		n_mixt=PyArray_DIMS(theta)[0]/3;
	}
	
    result=(PyArrayObject *) PyArray_FromDims(2, dims, NPY_DOUBLE);

	/* Retrieve C pointers to the data for ease of indexing */
	PyArray_AsCArray((PyObject **)&X1, &c_X1, PyArray_DIMS((PyArrayObject *)X1), 1, PyArray_DescrFromType(NPY_DOUBLE));	
	PyArray_AsCArray((PyObject **)&X2, &c_X2, PyArray_DIMS((PyArrayObject *)X2), 1, PyArray_DescrFromType(NPY_DOUBLE));
	PyArray_AsCArray((PyObject **)&theta, &c_theta, PyArray_DIMS((PyArrayObject *)theta), 1, PyArray_DescrFromType(NPY_DOUBLE));
	PyArray_AsCArray((PyObject **)&result, &c_result, PyArray_DIMS((PyArrayObject *)result), 2, PyArray_DescrFromType(NPY_DOUBLE));

	/* Compute the covariance matrix in place */
	cov(c_result, c_X1, dims[0], c_X2, dims[1], c_theta, n_mixt, type);

    /* Free the memory of the pointer array, not the memory occupied by the data */
	PyArray_Free((PyObject *)X1, c_X1);
	PyArray_Free((PyObject *)X2, c_X2);
	PyArray_Free((PyObject *)theta, c_theta);

    return PyArray_Return(result);
}


/* ==== Compute the covariance matrix of a derivative GP under a Spectral Mixture kernel. ==== */
static PyObject *cStringGPy_sm_deriv_cov(PyObject *self, PyObject *args) {
	PyArrayObject *X1=NULL, *X2=NULL, *theta=NULL, *result=NULL;
    double *c_X1, *c_X2, *c_theta, **c_result;

    if (!PyArg_ParseTuple(args, "O!O!O!", &PyArray_Type, &X1, &PyArray_Type, &X2,\
    	&PyArray_Type, &theta)) return NULL;

    if (not_doublevector(X1)) return NULL;
    if (not_doublevector(X2)) return NULL;
    if (not_doublevector(theta)) return NULL;

    /* Make the matrix of results */
    int n_mixt,dims[2];
    dims[0]=2*PyArray_DIMS(X1)[0];
    dims[1]=2*PyArray_DIMS(X2)[0];
    n_mixt=PyArray_DIMS(theta)[0]/3;
    result=(PyArrayObject *) PyArray_FromDims(2, dims, NPY_DOUBLE);

	/* Retrieve C pointers to the data for ease of indexing */
	PyArray_AsCArray((PyObject **)&X1, &c_X1, PyArray_DIMS((PyArrayObject *)X1), 1, PyArray_DescrFromType(NPY_DOUBLE));	
	PyArray_AsCArray((PyObject **)&X2, &c_X2, PyArray_DIMS((PyArrayObject *)X2), 1, PyArray_DescrFromType(NPY_DOUBLE));
	PyArray_AsCArray((PyObject **)&theta, &c_theta, PyArray_DIMS((PyArrayObject *)theta), 1, PyArray_DescrFromType(NPY_DOUBLE));
	PyArray_AsCArray((PyObject **)&result, &c_result, PyArray_DIMS((PyArrayObject *)result), 2, PyArray_DescrFromType(NPY_DOUBLE));

	/* Compute the covariance matrix in place */
    sm_deriv_cov(c_result, c_X1, dims[0]/2, c_X2, dims[1]/2, c_theta, n_mixt);

    /* Cleaning up */
	PyArray_Free((PyObject *)X1, c_X1);
	PyArray_Free((PyObject *)X2, c_X2);
	PyArray_Free((PyObject *)theta, c_theta);

    return PyArray_Return(result);
}


/* ==== Compute the covariance matrix of a derivative GP under a Spectral Mixture kernel. ==== */
static PyObject *cStringGPy_deriv_cov(PyObject *self, PyObject *args) {
	PyArrayObject *X1=NULL, *X2=NULL, *theta=NULL, *result=NULL;
    double *c_X1, *c_X2, *c_theta, **c_result;
	char *type;

    if (!PyArg_ParseTuple(args, "O!O!O!s", &PyArray_Type, &X1, &PyArray_Type, &X2,\
    	&PyArray_Type, &theta, &type)) return NULL;

    if (not_doublevector(X1)) return NULL;
    if (not_doublevector(X2)) return NULL;
    if (not_doublevector(theta)) return NULL;

    /* Make the matrix of results */
    int n_mixt,dims[2];
    dims[0]=2*PyArray_DIMS(X1)[0];
    dims[1]=2*PyArray_DIMS(X2)[0];
    n_mixt=PyArray_DIMS(theta)[0]/3;
    result=(PyArrayObject *) PyArray_FromDims(2, dims, NPY_DOUBLE);
	
	/* Retrieve C pointers to the data for ease of indexing */
	PyArray_AsCArray((PyObject **)&X1, &c_X1, PyArray_DIMS((PyArrayObject *)X1), 1, PyArray_DescrFromType(NPY_DOUBLE));	
	PyArray_AsCArray((PyObject **)&X2, &c_X2, PyArray_DIMS((PyArrayObject *)X2), 1, PyArray_DescrFromType(NPY_DOUBLE));
	PyArray_AsCArray((PyObject **)&theta, &c_theta, PyArray_DIMS((PyArrayObject *)theta), 1, PyArray_DescrFromType(NPY_DOUBLE));
	PyArray_AsCArray((PyObject **)&result, &c_result, PyArray_DIMS((PyArrayObject *)result), 2, PyArray_DescrFromType(NPY_DOUBLE));
	
	/* Compute the covariance matrix in place */
    deriv_cov(c_result, c_X1, dims[0]/2, c_X2, dims[1]/2, c_theta, n_mixt, type);

    /* Cleaning up. */
	PyArray_Free((PyObject *)X1, c_X1);
	PyArray_Free((PyObject *)X2, c_X2);
	PyArray_Free((PyObject *)theta, c_theta);
	
    return PyArray_Return(result);
}

/* ==== Compute the covariance matrix of a String Spectral Mixture kernel at boundary times. ==== */
static PyObject *cStringGPy_string_sm_boundaries_deriv_cov(PyObject *self, PyObject *args) {
    PyArrayObject *b_times=NULL, *thetas=NULL, *result=NULL;
    double *c_b_times, **c_thetas, **c_result;
	unsigned int n_mixts; 
	
    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &b_times, &PyArray_Type, &thetas)) 
		return Py_BuildValue("");
	
	if (not_doublevector(b_times)) return Py_BuildValue("");
	if (not_doublematrix(thetas)) return Py_BuildValue("");
	
	/* Make the matrix of results */
    int dims[2],n_b_times;
	n_b_times=PyArray_DIMS(b_times)[0];
    dims[0]=2*n_b_times;
    dims[1]=2*n_b_times;
	
	assert((PyArray_DIMS(thetas)[1]%3)==0); // The number of hyper-parameters should be a multiple of 3.
    n_mixts=PyArray_DIMS(thetas)[1]/3;
	assert(PyArray_DIMS(b_times)[0] == 1+PyArray_DIMS(thetas)[0]); // The number of boundary elements should be 1 + the number of strings.

    result=(PyArrayObject *) PyArray_FromDims(2, dims, NPY_DOUBLE);
	
	/* Retrieve C pointers to data for ease of indexing. */
	PyArray_AsCArray((PyObject **)&b_times, &c_b_times, PyArray_DIMS((PyArrayObject *)b_times), 1, PyArray_DescrFromType(NPY_DOUBLE));	
	PyArray_AsCArray((PyObject **)&thetas, &c_thetas, PyArray_DIMS((PyArrayObject *)thetas), 2, PyArray_DescrFromType(NPY_DOUBLE));
	PyArray_AsCArray_Safe((PyObject **)&result, &c_result, PyArray_DIMS((PyArrayObject *)result), 2, PyArray_DescrFromType(NPY_DOUBLE));
	
	/* Compute the covariance matrix in place */
	string_sm_boundaries_deriv_cov(c_result, c_b_times, n_b_times, c_thetas, n_mixts);
	
    /* Cleanup */
	PyArray_Free((PyObject *)b_times, c_b_times);
	PyArray_Free((PyObject *)thetas, c_thetas);
	
    return PyArray_Return(result);
}

/* ==== Compute the covariance matrix of a String Spectral Mixture kernel at boundary times. ==== */
static PyObject *cStringGPy_string_boundaries_deriv_cov(PyObject *self, PyObject *args) {
    PyArrayObject *b_times=NULL, *thetas=NULL, *result=NULL;
    double *c_b_times, **c_thetas, **c_result;
	unsigned int n_mixts;
	char *type;
	
    if (!PyArg_ParseTuple(args, "O!O!s", &PyArray_Type, &b_times, &PyArray_Type, &thetas, &type)) 
		return Py_BuildValue("");
	
	if (not_doublevector(b_times)) return Py_BuildValue("");
	if (not_doublematrix(thetas)) return Py_BuildValue("");
	
	/* Make the matrix of results */
    int dims[2],n_b_times;
	n_b_times=PyArray_DIMS(b_times)[0];
    dims[0]=2*n_b_times;
    dims[1]=2*n_b_times;
	
	assert((PyArray_DIMS(thetas)[1]%3)==0); // The number of hyper-parameters should be a multiple of 3.
    n_mixts=PyArray_DIMS(thetas)[1]/3;
	assert(PyArray_DIMS(b_times)[0] == 1+PyArray_DIMS(thetas)[0]); // The number of boundary elements should be 1 + the number of strings.
    result=(PyArrayObject *) PyArray_FromDims(2, dims, NPY_DOUBLE);
	
	/* Retrieve C pointers to data for ease of indexing. */
	PyArray_AsCArray((PyObject **)&b_times, &c_b_times, PyArray_DIMS((PyArrayObject *)b_times), 1, PyArray_DescrFromType(NPY_DOUBLE));
	PyArray_AsCArray((PyObject **)&thetas, &c_thetas, PyArray_DIMS((PyArrayObject *)thetas), 2, PyArray_DescrFromType(NPY_DOUBLE));
	PyArray_AsCArray_Safe((PyObject **)&result, &c_result, PyArray_DIMS((PyArrayObject *)result), 2, PyArray_DescrFromType(NPY_DOUBLE));
	
	/* Compute the covariance matrix in place */
	string_boundaries_deriv_cov(c_result, c_b_times, n_b_times, c_thetas, n_mixts, type);
	
    /* Cleanup */
	PyArray_Free((PyObject *)b_times, c_b_times);
	PyArray_Free((PyObject *)thetas, c_thetas);

    return PyArray_Return(result);
}

/* ==== Compute the covariance matrix of a String Spectral Mixture kernel. ====
    Assumes PyArray is contiguous in memory.             */
static PyObject *cStringGPy_string_sm_deriv_cov(PyObject *self, PyObject *args) {
    PyArrayObject *s_times=NULL, *b_times=NULL, *thetas=NULL, *result=NULL;
    double *c_s_times, *c_b_times, **c_thetas, **c_result;

    if (!PyArg_ParseTuple(args, "O!O!O!", &PyArray_Type, &s_times, &PyArray_Type, &b_times,\
    	&PyArray_Type, &thetas)) return Py_BuildValue("");

    if (not_doublevector(s_times)) return Py_BuildValue("");
    if (not_doublevector(b_times)) return Py_BuildValue("");
    if (not_doublematrix(thetas)) return Py_BuildValue("");

    /* Make a new double vector of same dimension */
	unsigned int n_s_times, n_b_times, n_mixts;
	int dims[2];
	
	n_s_times=PyArray_DIMS(s_times)[0];
	n_b_times=PyArray_DIMS(b_times)[0];
	
	assert((PyArray_DIMS(thetas)[1]%3)==0); // The number of hyper-parameters should be a multiple of 3.
    n_mixts=PyArray_DIMS(thetas)[1]/3;

	assert(PyArray_DIMS(b_times)[0] == 1+PyArray_DIMS(thetas)[0]); // The number of boundary elements should be 1 + the number of strings.
	
	dims[0]=2*n_s_times;
	dims[1]=2*n_s_times;
	
	result=(PyArrayObject *) PyArray_FromDims(2, dims, NPY_DOUBLE);
  
	/* Retrieve C pointers to data for ease of indexing. */
	PyArray_AsCArray((PyObject **)&s_times, &c_s_times, PyArray_DIMS((PyArrayObject *)s_times), 1, PyArray_DescrFromType(NPY_DOUBLE));
	PyArray_AsCArray((PyObject **)&b_times, &c_b_times, PyArray_DIMS((PyArrayObject *)b_times), 1, PyArray_DescrFromType(NPY_DOUBLE));	
	PyArray_AsCArray((PyObject **)&thetas, &c_thetas, PyArray_DIMS((PyArrayObject *)thetas), 2, PyArray_DescrFromType(NPY_DOUBLE));
	PyArray_AsCArray_Safe((PyObject **)&result, &c_result, PyArray_DIMS((PyArrayObject *)result), 2, PyArray_DescrFromType(NPY_DOUBLE));

	/* Compute the covariance matrix in place */
	string_sm_deriv_cov(c_result, c_s_times, n_s_times, c_b_times, n_b_times, c_thetas, n_mixts);
	
    /* Cleaning up. */
	PyArray_Free((PyObject *)s_times, c_s_times);
	PyArray_Free((PyObject *)b_times, c_b_times);
	PyArray_Free((PyObject *)thetas, c_thetas);

    return PyArray_Return(result);
}


/* ==== Compute the covariance matrix of a string derivative GP for a given family of expert kernels. ==== */
static PyObject *cStringGPy_string_deriv_cov(PyObject *self, PyObject *args) {
    PyArrayObject *s_times=NULL, *b_times=NULL, *thetas=NULL, *result=NULL;
    double *c_s_times, *c_b_times, **c_thetas, **c_result;
	char *type;

    if (!PyArg_ParseTuple(args, "O!O!O!s", &PyArray_Type, &s_times, &PyArray_Type, &b_times,\
    	&PyArray_Type, &thetas, &type)) return Py_BuildValue("");

    if (not_doublevector(s_times)) return Py_BuildValue("");
    if (not_doublevector(b_times)) return Py_BuildValue("");
    if (not_doublematrix(thetas)) return Py_BuildValue("");

    /* Make a new double vector of same dimension */
	unsigned int n_s_times, n_b_times, n_mixts;
	int dims[2];
	
	n_s_times=PyArray_DIMS(s_times)[0];
	n_b_times=PyArray_DIMS(b_times)[0];
	
	assert(((PyArray_DIMS(thetas)[1]%3)==0) | (strcmp(type, "sm") !=0)); // The number of hyper-parameters should be a multiple of 3.
    n_mixts=PyArray_DIMS(thetas)[1]/3;

	assert(PyArray_DIMS(b_times)[0] == 1+PyArray_DIMS(thetas)[0]); // The number of boundary elements should be 1 + the number of strings.
	
	dims[0]=2*n_s_times;
	dims[1]=2*n_s_times;
	
	result=(PyArrayObject *) PyArray_FromDims(2, dims, NPY_DOUBLE);
  
	/* Retrieve C pointers to the data for ease of indexing */
	PyArray_AsCArray((PyObject **)&s_times, &c_s_times, PyArray_DIMS((PyArrayObject *)s_times), 1, PyArray_DescrFromType(NPY_DOUBLE));	
	PyArray_AsCArray((PyObject **)&b_times, &c_b_times, PyArray_DIMS((PyArrayObject *)b_times), 1, PyArray_DescrFromType(NPY_DOUBLE));
	PyArray_AsCArray((PyObject **)&thetas, &c_thetas, PyArray_DIMS((PyArrayObject *)thetas), 2, PyArray_DescrFromType(NPY_DOUBLE));	
	PyArray_AsCArray_Safe((PyObject **)&result, &c_result, PyArray_DIMS((PyArrayObject *)result), 2, PyArray_DescrFromType(NPY_DOUBLE));

	/* Compute the covariance matrix in place */
	string_deriv_cov(c_result, c_s_times, n_s_times, c_b_times, n_b_times, c_thetas, n_mixts, type);
	
    /* Cleanup */
	// b_times and s_times no longer point to the Python side object whose references
	//	were borrowed by a call to this function.
	PyArray_Free((PyObject *)s_times, c_s_times);
	PyArray_Free((PyObject *)b_times, c_b_times);
	PyArray_Free((PyObject *)thetas, c_thetas);

    return PyArray_Return(result);
}


/* ==== Compute the covariance matrix of a string derivative GP for a given family of expert kernels. ==== */
static PyObject *cStringGPy_string_cov(PyObject *self, PyObject *args) {
    PyArrayObject *s_times=NULL, *b_times=NULL, *thetas=NULL, *result=NULL;
    double *c_s_times, *c_b_times, **c_thetas, **c_result;
	char *type;

    if (!PyArg_ParseTuple(args, "O!O!O!s", &PyArray_Type, &s_times, &PyArray_Type, &b_times,\
    	&PyArray_Type, &thetas, &type)) return Py_BuildValue("");

    if (not_doublevector(s_times)) return Py_BuildValue("");
    if (not_doublevector(b_times)) return Py_BuildValue("");
    if (not_doublematrix(thetas)) return Py_BuildValue("");

    /* Make a new double vector of same dimension */
	unsigned int n_s_times, n_b_times, n_mixts;
	int dims[2];
	
	n_s_times=PyArray_DIMS(s_times)[0];
	n_b_times=PyArray_DIMS(b_times)[0];
	
	assert(((PyArray_DIMS(thetas)[1]%3)==0) | (strcmp(type, "sm") !=0)); // The number of hyper-parameters should be a multiple of 3.
    n_mixts=PyArray_DIMS(thetas)[1]/3;

	assert(PyArray_DIMS(b_times)[0] == 1+PyArray_DIMS(thetas)[0]); // The number of boundary elements should be 1 + the number of strings.
	
	dims[0]=n_s_times;
	dims[1]=n_s_times;
	
	result=(PyArrayObject *) PyArray_FromDims(2, dims, NPY_DOUBLE);
  
	/* Retrieve C pointers to data for ease of indexing. */
	PyArray_AsCArray((PyObject **)&s_times, &c_s_times, PyArray_DIMS((PyArrayObject *)s_times), 1, PyArray_DescrFromType(NPY_DOUBLE));
	PyArray_AsCArray((PyObject **)&b_times, &c_b_times, PyArray_DIMS((PyArrayObject *)b_times), 1, PyArray_DescrFromType(NPY_DOUBLE));
	PyArray_AsCArray((PyObject **)&thetas, &c_thetas, PyArray_DIMS((PyArrayObject *)thetas), 2, PyArray_DescrFromType(NPY_DOUBLE));
	PyArray_AsCArray_Safe((PyObject **)&result, &c_result, PyArray_DIMS((PyArrayObject *)result), 2, PyArray_DescrFromType(NPY_DOUBLE));

	/* Compute the covariance matrix in place */
	string_cov(c_result, c_s_times, n_s_times, c_b_times, n_b_times, c_thetas, n_mixts, type);
	
    /* Cleaning up */
	// b_times and s_times no longer point to the Python side object whose references
	//	were borrowed by a call to this function.
	PyArray_Free((PyObject *)s_times, c_s_times);
	PyArray_Free((PyObject *)b_times, c_b_times);
	PyArray_Free((PyObject *)thetas, c_thetas);

    return PyArray_Return(result);
}

static PyObject * cStringGPy_bessel_kv(PyObject *self, PyObject *args) {
    double nu, x, res;
    if (!PyArg_ParseTuple(args, "dd", &nu, &x)) 
		return Py_BuildValue("");
	res = bessel_kv(nu, x);
    return PyFloat_FromDouble(res);
}

static PyObject * cStringGPy_sample_sgp(PyObject *self, PyObject *args) {
    PyListObject *kernel_types=NULL, *kernel_hypers=NULL, *b_times=NULL, *s_times=NULL, *Ls=NULL;
    PyObject *list_result=NULL;

    if (!PyArg_ParseTuple(args, "O!O!O!O!O!", &PyList_Type, &kernel_types, &PyList_Type, &kernel_hypers, &PyList_Type, &b_times, &PyList_Type, &s_times, &PyList_Type, &Ls)) 
		return Py_BuildValue("");

	list_result=sample_sgp((PyObject *)kernel_types, (PyObject *)kernel_hypers, (PyObject *)b_times, (PyObject *)s_times, (PyObject *)Ls);
	return list_result;
}

static PyObject * cStringGPy_sample_sgps(PyObject *self, PyObject *args) {
    PyListObject *l_kernel_types=NULL, *l_kernel_hypers=NULL, *l_b_times=NULL, *l_s_times=NULL, *l_Ls=NULL;
    PyObject *list_result=NULL;

    if (!PyArg_ParseTuple(args, "O!O!O!O!O!", &PyList_Type, &l_kernel_types, &PyList_Type, &l_kernel_hypers, &PyList_Type, &l_b_times, &PyList_Type, &l_s_times, &PyList_Type, &l_Ls)) 
		return Py_BuildValue("");

	list_result=sample_sgps((PyObject *)l_kernel_types, (PyObject *)l_kernel_hypers, (PyObject *)l_b_times, (PyObject *)l_s_times, (PyObject *)l_Ls);
	return list_result;
}


static PyObject * cStringGPy_cond_eigen_anal(PyObject *self, PyObject *args) {
    PyListObject *kernel_types=NULL, *kernel_hypers=NULL, *b_times=NULL, *s_times=NULL;
    PyObject *list_result=NULL;

    if (!PyArg_ParseTuple(args, "O!O!O!O!", &PyList_Type, &kernel_types, &PyList_Type, &kernel_hypers, &PyList_Type, &b_times, &PyList_Type, &s_times)) 
		return Py_BuildValue("");

	list_result=cond_eigen_anal((PyObject *)kernel_types, (PyObject *)kernel_hypers, (PyObject *)b_times, (PyObject *)s_times);
	return list_result;
}

static PyObject * cStringGPy_cond_eigen_anals(PyObject *self, PyObject *args) {
    PyListObject *l_kernel_types=NULL, *l_kernel_hypers=NULL, *l_b_times=NULL, *l_s_times=NULL;
    PyObject *list_result=NULL;

    if (!PyArg_ParseTuple(args, "O!O!O!O!", &PyList_Type, &l_kernel_types, &PyList_Type, &l_kernel_hypers, &PyList_Type, &l_b_times, &PyList_Type, &l_s_times))
		return Py_BuildValue("");

	list_result=cond_eigen_anals((PyObject *)l_kernel_types, (PyObject *)l_kernel_hypers, (PyObject *)l_b_times, (PyObject *)l_s_times);
	return list_result;
}

static PyObject * cStringGPy_float_as_idx(PyObject *self, PyObject *args){
	PyFloatObject *p_val=NULL;
	PyObject *res=NULL;
	if (!PyArg_ParseTuple(args, "O!", &PyFloat_Type, &p_val))
		return Py_BuildValue("");

	res=float_as_idx((PyObject *)p_val);
	return res;
}

static PyObject * cStringGPy_compute_bound_ls(PyObject *self, PyObject *args) {
    PyListObject *l_kernel_types=NULL, *l_kernel_hypers=NULL, *l_b_times=NULL;
    PyObject *list_result=NULL;

    if (!PyArg_ParseTuple(args, "O!O!O!", &PyList_Type, &l_kernel_types, &PyList_Type, &l_kernel_hypers, &PyList_Type, &l_b_times))
		return Py_BuildValue("");

	list_result=compute_bound_ls((PyObject *)l_kernel_types, (PyObject *)l_kernel_hypers, (PyObject *)l_b_times);
	return list_result;
}

static PyObject * cStringGPy_sample_whtn_bound_conds(PyObject *self, PyObject *args) {
    PyListObject *l_b_times=NULL;
    PyObject *list_result=NULL;

    if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &l_b_times))
		return Py_BuildValue("");

	list_result=sample_whtn_bound_conds((PyObject *)l_b_times);
	return list_result;
}

static PyObject * cStringGPy_sample_whtn_string(PyObject *self, PyObject *args) {
    PyListObject *l_s_times=NULL;
    PyObject *list_result=NULL;

    if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &l_s_times))
		return Py_BuildValue("");

	list_result=sample_whtn_string((PyObject *)l_s_times);
	return list_result;
}

static PyObject * cStringGPy_compute_sgps_from_lxs(PyObject *self, PyObject *args) {
    PyListObject *l_kernel_types=NULL, *l_kernel_hypers=NULL, *l_b_times=NULL, *l_s_times=NULL, *l_Xb=NULL, *l_Xs=NULL;
    PyObject *l_bound_eig=NULL, *l_string_eig=NULL; // Can be equal to PyNone
    PyObject *list_result=NULL;

    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!OO", &PyList_Type, &l_kernel_types, &PyList_Type, &l_kernel_hypers, &PyList_Type, &l_b_times, &PyList_Type, &l_s_times,\
    		&PyList_Type, &l_Xb, &PyList_Type, &l_Xs, &l_bound_eig, &l_string_eig))
		return Py_BuildValue("");

	list_result=compute_sgps_from_lxs((PyObject *)l_kernel_types, (PyObject *)l_kernel_hypers, (PyObject *)l_b_times, (PyObject *)l_s_times, (PyObject *)l_Xb,\
		(PyObject *)l_Xs, l_bound_eig, l_string_eig);

	return list_result;
}

static PyObject *cStringGPy_elliptic_tranform_lx(PyObject *self, PyObject *args) {
    PyListObject *old_l_x=NULL, *new_l_x=NULL;
    PyFloatObject *a=NULL;
    PyObject *list_result=NULL;

    if (!PyArg_ParseTuple(args, "O!O!O!", &PyList_Type, &old_l_x, &PyList_Type, &new_l_x, &PyFloat_Type, &a))
		return Py_BuildValue("");

	list_result=elliptic_tranform_lx((PyObject *)old_l_x, (PyObject *)new_l_x, (PyObject *)a);

	return list_result;
}

static PyObject *cStringGPy_log_lik_whtn(PyObject *self, PyObject *args) {
    PyListObject *l_xb=NULL, *l_xs=NULL;
    PyObject *result=NULL;

    if (!PyArg_ParseTuple(args, "O!O!", &PyList_Type, &l_xb, &PyList_Type, &l_xs))
		return Py_BuildValue("");

	result=log_lik_whtn((PyObject *)l_xb, (PyObject *)l_xs);

	return result;
}

static PyObject * cStringGPy_sample_norm_hypers(PyObject *self, PyObject *args) {
    PyListObject *l_shape=NULL;
    PyObject *l_means=NULL, *std=NULL, *list_result=NULL;

    if (!PyArg_ParseTuple(args, "O!OO", &PyList_Type, &l_shape, &l_means, &std))
		return Py_BuildValue("");

	list_result=sample_norm_hypers((PyObject *)l_shape, (PyObject *)l_means, (PyObject *)std);
	return list_result;
}

static PyObject *cStringGPy_scaled_sigmoid(PyObject *self, PyObject *args) {
    PyListObject *l_hypers_max=NULL, *l_hypers_norm=NULL;
    PyObject *list_result=NULL;

    if (!PyArg_ParseTuple(args, "O!O!", &PyList_Type, &l_hypers_max, &PyList_Type, &l_hypers_norm))
		return Py_BuildValue("");

	list_result=scaled_sigmoid((PyObject *)l_hypers_max, (PyObject *)l_hypers_norm);

	return list_result;
}

static PyObject *cStringGPy_model_log_lik(PyObject *self, PyObject *args) {
    PyListObject *l_sgp=NULL;
    PyObject *result=NULL;
    PyArrayObject *data=NULL;
    PyStringObject *link_f_type=NULL, *ll_type=NULL;
    PyFloatObject  *noise_var=NULL;

    if (!PyArg_ParseTuple(args, "O!O!O!O!O!", &PyArray_Type, &data, &PyList_Type, &l_sgp, &PyString_Type, &link_f_type, &PyString_Type, &ll_type, &PyFloat_Type, &noise_var))
		return Py_BuildValue("");

	result=model_log_lik((PyObject *)data, (PyObject *)l_sgp, (PyObject *)link_f_type, (PyObject *)ll_type, (PyObject *)noise_var);

	return result;
}


static char cStringGPy_string_sm_boundaries_deriv_cov_docs[] =
	"string_sm_boundaries_deriv_cov(b_times, thetas, n_thetas): Computes the covariance matrix of a String Derivative GP at boundary times under a spectral mixture kernel.";
	
static char cStringGPy_string_sm_cov_deriv_docs[] =
	"string_sm_deriv_cov(s_times, b_times, thetas): Computes the covariance matrix of a String spectral mixture kernel. The number of mixture components is the number of columns of each row divided by 3.\n";
	
static char cStringGPy_sm_cov_docs[] =
	"sm_cov(X1, X2, theta): Computes the cross-covariance matrix of a GP at times X1 and X2 under a spectral mixture kernel.";	

static char cStringGPy_sm_deriv_cov_docs[] =
	"sm_deriv_cov(X1, X2, theta): Computes the cross-covariance matrix of a Derivative GP at times X1 and X2 under a spectral mixture kernel.";
		
static char cStringGPy_string_boundaries_deriv_cov_docs[] =
	"string_boundaries_deriv_cov(b_times, thetas, n_thetas, kernel_type): Computes the covariance matrix of a String Derivative GP at boundary times under a given kernel.";
	
static char cStringGPy_string_deriv_cov_docs[] =
	"string_deriv_cov(s_times, b_times, thetas, kernel_type): Computes the covariance matrix of a String Derivative GP.";
	
static char cStringGPy_string_cov_docs[] =
	"string_cov(s_times, b_times, thetas, kernel_type): Computes the covariance matrix of a String GP.";
	
static char cStringGPy_cov_docs[] =
	"cov(X1, X2, theta, kernel_type): Computes the cross-covariance matrix of a GP at times X1 and X2 under a given kernel.";	

static char cStringGPy_deriv_cov_docs[] =
	"deriv_cov(X1, X2, theta, kernel_type): Computes the cross-covariance matrix of a Derivative GP at times X1 and X2 under a given kernel.";
	
static char cStringGPy_bessel_kv_docs[] =
	"bessel_kv(nu, x): Equivalent to scipy.special.kv(nu, x).";

static char cStringGPy_sample_sgp_docs[] =
	"sample_sgp(kernel_types, kernel_hypers, boundary_times, string_times, Ls): Sample a path of a derivative string GP in parallel.";

static char cStringGPy_sample_sgps_docs[] =
	"sample_sgps(kernel_types, kernel_hypers, boundary_times, string_times, Ls): Sample independent paths of derivative string GP in parallel.";

static char cStringGPy_cond_eigen_anal_docs[] =
	"cond_eigen_anal(kernel_types, kernel_hypers, boundary_times, string_times): Perform eigenvalue analysis on string conditional covariance matrices in parallel, and return their L\
		factors (L=US^{0.5}), their inverse L factors, M=K(s_times, b_times)K(b_times, b_times)^{-1}, and the log-determinants of the conditional string covariance matrices.";
	
static char cStringGPy_cond_eigen_anals_docs[] =
	"cond_eigen_anals(kernel_types, kernel_hypers, boundary_times, string_times): Perform eigenvalue analyses on string conditional covariance matrices in parallel (across input dimensions), and return their L\
		factors (L=US^{0.5}), their inverse L factors, M=K(s_times, b_times)K(b_times, b_times)^{-1}, and the log-determinants of the conditional string covariance matrices.";
	
static char cStringGPy_float_as_idx_docs[] =
	"float_as_idx(val): Truncates a double up to 6 decimal points and returns a string.";

static char cStringGPy_compute_bound_ls_docs[] =
	"compute_bound_ls(l_kernel_types, l_kernel_hypers, l_b_times): Compute the SVD factors L in the decomposition of the covariance matrix of the boundary conditions at a time, conditional\
		on the boundary conditions at the previous time.";

static char cStringGPy_sample_whtn_bound_conds_docs[] =
	"sample_whtn_bound_conds(l_b_times): Sample i.i.d. standard normal, two per boundary time. The samples can effectively be thought of as whitened boundary condtions.";

static char cStringGPy_sample_whtn_string_docs[] =
	"sample_whtn_string(l_s_times): Sample i.i.d. standard normal, two per string time. The samples can effectively be thought of as whitened DSGP values at string times.";

static char cStringGPy_compute_sgps_from_lxs_docs[] =
	"compute_sgps_from_lxs(l_kernel_types, l_kernel_hypers, l_b_times, l_s_times, l_Xb, l_Xs, l_bound_eig, l_string_eig): Compute multiple DSGP from whitened samples and eigen factors L (or hyper-parameters when they are not available).";

static char cStringGPy_elliptic_tranform_lx_docs[] =
	"elliptic_tranform_lx(old, new, a): Return old*cos(a) + new*sin(a) element-wise.";

static char cStringGPy_log_lik_whtn_docs[] =
	"log_lik_whtn(l_xs, l_xb): Log-likelihood due to the whithened component.";

static char cStringGPy_sample_norm_hypers_docs[] =
	"sample_norm_hypers(l_shape, l_means, std): Sample scaled sigmoid of i.i.d standard normals.";

static char cStringGPy_scaled_sigmoid_docs[] =
	"scaled_sigmoid(l_hypers_max, l_hypers_norm): Scaled sigmoid transform.";

static char cStringGPy_model_log_lik_docs[] =
	"model_log_lik(data, l_sgp, link_f_type, ll_type, noise_var): Model Log-likelihood.";

static PyMethodDef cStringGPy_funcs[] = {
	{"bessel_kv", (PyCFunction)cStringGPy_bessel_kv, METH_VARARGS, cStringGPy_bessel_kv_docs},
	{"string_sm_boundaries_deriv_cov", (PyCFunction)cStringGPy_string_sm_boundaries_deriv_cov, METH_VARARGS, cStringGPy_string_sm_boundaries_deriv_cov_docs},
	{"string_sm_deriv_cov", (PyCFunction)cStringGPy_string_sm_deriv_cov, METH_VARARGS, cStringGPy_string_sm_cov_deriv_docs},
	{"sm_cov", (PyCFunction)cStringGPy_sm_cov, METH_VARARGS, cStringGPy_sm_cov_docs},
	{"sm_deriv_cov", (PyCFunction)cStringGPy_sm_deriv_cov, METH_VARARGS, cStringGPy_sm_deriv_cov_docs},
	{"string_boundaries_deriv_cov", (PyCFunction)cStringGPy_string_boundaries_deriv_cov, METH_VARARGS, cStringGPy_string_boundaries_deriv_cov_docs},
	{"string_deriv_cov", (PyCFunction)cStringGPy_string_deriv_cov, METH_VARARGS, cStringGPy_string_deriv_cov_docs},
	{"string_cov", (PyCFunction)cStringGPy_string_cov, METH_VARARGS, cStringGPy_string_cov_docs},
	{"cov", (PyCFunction)cStringGPy_cov, METH_VARARGS, cStringGPy_cov_docs},
	{"deriv_cov", (PyCFunction)cStringGPy_deriv_cov, METH_VARARGS, cStringGPy_deriv_cov_docs},
	{"sample_sgp", (PyCFunction)cStringGPy_sample_sgp, METH_VARARGS, cStringGPy_sample_sgp_docs},
	{"sample_sgps", (PyCFunction)cStringGPy_sample_sgps, METH_VARARGS, cStringGPy_sample_sgps_docs},
	{"cond_eigen_anal", (PyCFunction)cStringGPy_cond_eigen_anal, METH_VARARGS, cStringGPy_cond_eigen_anal_docs},
	{"cond_eigen_anals", (PyCFunction)cStringGPy_cond_eigen_anals, METH_VARARGS, cStringGPy_cond_eigen_anals_docs},
	{"float_as_idx", (PyCFunction)cStringGPy_float_as_idx, METH_VARARGS, cStringGPy_float_as_idx_docs},
	{"compute_bound_ls", (PyCFunction)cStringGPy_compute_bound_ls, METH_VARARGS, cStringGPy_compute_bound_ls_docs},
	{"sample_whtn_bound_conds", (PyCFunction)cStringGPy_sample_whtn_bound_conds, METH_VARARGS, cStringGPy_sample_whtn_bound_conds_docs},
	{"sample_whtn_string", (PyCFunction)cStringGPy_sample_whtn_string, METH_VARARGS, cStringGPy_sample_whtn_string_docs},
	{"compute_sgps_from_lxs", (PyCFunction)cStringGPy_compute_sgps_from_lxs, METH_VARARGS, cStringGPy_compute_sgps_from_lxs_docs},
	{"elliptic_tranform_lx", (PyCFunction)cStringGPy_elliptic_tranform_lx, METH_VARARGS, cStringGPy_elliptic_tranform_lx_docs},
	{"log_lik_whtn", (PyCFunction)cStringGPy_log_lik_whtn, METH_VARARGS, cStringGPy_log_lik_whtn_docs},
	{"sample_norm_hypers", (PyCFunction)cStringGPy_sample_norm_hypers, METH_VARARGS, cStringGPy_sample_norm_hypers_docs},
	{"scaled_sigmoid", (PyCFunction)cStringGPy_scaled_sigmoid, METH_VARARGS, cStringGPy_scaled_sigmoid_docs},
	{"model_log_lik", (PyCFunction)cStringGPy_model_log_lik, METH_VARARGS, cStringGPy_model_log_lik_docs},
	{NULL}
};

void initcStringGPy(void)
{
	Py_InitModule3("cStringGPy", cStringGPy_funcs, "String Gaussian Process module.");
	import_array();
	import_ufunc();
}