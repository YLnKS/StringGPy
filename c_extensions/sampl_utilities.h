#ifndef SAMPL_UTILITIES_H
#define SAMPL_UTILITIES_H

#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL cStringGPy_ARRAY_API

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_9_API_VERSION
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include <math.h>
#include <errno.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/time.h>
#include "lin_alg_utilities.h"
#include "py_imports.h"
#include "kernel_utilities.h"
#include "py_utilities.h"

/* 
	Sample a single path of a univariate string GP.
		*result: nx3 PyArrayObject that will be modified in place. The first column contains times, the second sGP values and the third derivatives.
		*kernel_types: list of kernel types with K elements. E.g: [sm, se, sma, sma12] etc...
		*kernel_hypers: list of K PyArrayObject containing unconditional string kernel hyper-parameters.
		*b_times: list of the K+1 boundary times.
		*s_times: list of K PyArrayObject containing string times.
		*Ls: list of K n_kxn_k PyArrayObject containing the L factor in the Cholesky or SVD decomposition of the covariance matrix of the 
			values of the DSGP at the string times conditional on the values at the boundaries.
			
	The boundary conditions are sampled sequentially, conditional on which the values within strings are sampled in parallel (process based). 
*/
PyObject *sample_sgp(PyObject *kernel_types, PyObject *kernel_hypers, PyObject *b_times, PyObject *s_times, PyObject *Ls);

/* 
	Sample independent paths of a univariate string GPs in parallel.
		This is essentially equivalent to calling sample_sgp in parallel.
*/
PyObject *sample_sgps(PyObject *l_kernel_types, PyObject *l_kernel_hypers, PyObject *l_b_times, PyObject *l_s_times, PyObject *l_Ls);

/* 
	Performs eigenvalue analysis of the conditional string times covariance matrices of a univariate string derivative Gaussian process. The following are derived
		from the singular value decompositions (C=USV) of the foregoing matrices, and returned by this function: $L=US^{\frac{1}{2}}$, $det(C)=\prod_{i}S[i]$ and $L^{-1}=S^{\frac{-1}{2}}V$.
		The multiplicative factor $M=cov_stimes_btimes*cov_btimes_btimes^{-1}$ in the formula of the mean of SDGP values at string times conditional on boundary conditions is also returned.
	
		*result: Kx4 PyListObject that will be modified in place. The first column contains L, the second column L^{-1}, the third $M$ and the fourth det(C).
		*kernel_types: list of kernel types with K elements. E.g: [sm, se, sma, sma12] etc...
		*kernel_hypers: list of K PyArrayObject containing unconditional string kernel hyper-parameters.
		*b_times: list of the K+1 boundary times.
		*s_times: list of K PyArrayObject containing string times.
		
	The computations are done in parallel.
*/
PyObject *cond_eigen_anal(PyObject *kernel_types, PyObject *kernel_hypers, PyObject *b_times, PyObject *s_times);

/*
	Perform multiple eigenvalue analyses in parallel.
		This is essentially equivalent to calling cong_eigen_anal in parallel.
*/
PyObject *cond_eigen_anals(PyObject *l_kernel_types, PyObject *l_kernel_hypers, PyObject *l_b_times, PyObject *l_s_times);


/*
	Truncate a double to 6 decimals digits and return it as a string.
*/
PyObject *float_as_idx(PyObject* val);


/*
	Sample the eigen factors L corresponding to the covariance matrices of DSGPs at a boundary time conditional on the previous.
*/
PyObject *compute_bound_ls(PyObject *l_kernel_types, PyObject *l_kernel_hypers, PyObject *l_b_times);


/*
	Sample i.i.d standard normals for boundary conditions.
*/
PyObject *sample_whtn_bound_conds(PyObject *l_b_times);


/*
	Sample i.i.d standard normals for inner (conditional) inner string values.
*/
PyObject *sample_whtn_string(PyObject *l_s_times);


/*
	Similar to sample_sgps_from_ls except that both L factors and whithened values are provided.
*/
PyObject *compute_sgps_from_lxs(PyObject *l_kernel_types, PyObject *l_kernel_hypers, PyObject *l_b_times, PyObject *l_s_times, PyObject *l_Xb, PyObject *l_Xs, PyObject *l_bound_eig, PyObject *l_string_eig);


/*
	Returns old*cos(a) + new*sin(a) element-wise.
*/
PyObject *elliptic_tranform_lx(PyObject *old_l_x, PyObject *new_l_x, PyObject *a);

/*
	Log-likelihood in the whitened case.
*/
PyObject *log_lik_whtn(PyObject *l_xb, PyObject *l_xs);

/*
	Sample i.i.d standard normals.
*/
PyObject *sample_norm_hypers(PyObject *l_shape, PyObject *l_means, PyObject *std);

/*
	Returns h_max/(1+exp(-h_norm)).
*/
PyObject *scaled_sigmoid(PyObject *l_hypers_max, PyObject *l_hypers_norm);

/*
	Compute the log likelihood of a DSGP.
*/
PyObject *model_log_lik(PyObject *data, PyObject *l_sgp, PyObject *link_f_type, PyObject *ll_type, PyObject *noise_var);

#endif