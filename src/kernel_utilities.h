
#ifndef KERNEL_UTILITIES_H
#define KERNEL_UTILITIES_H

#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL cStringGPy_ARRAY_API

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_9_API_VERSION
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <limits.h>
#include <assert.h>
#include <string.h>
#include "lin_alg_utilities.h"

/* ==== Evaluate the polynomial kernel for given inputs. ==== */
double poly_kernel(double x, double y, double *theta);

/* ==== Evaluate the derivative of the polynomial kernel with respect to x. ==== */
double dpoly_kernel_dx(double x, double y, double *theta);

/* ==== Evaluate the derivative of the polynomial kernel with respect to y. ==== */
double dpoly_kernel_dy(double x, double y, double *theta);

/* ==== Evaluate the cross derivative of the polynomial kernel with respect to x and y. ==== */
double dpoly_kernel_dxdy(double x, double y, double *theta);

/* ==== Evaluate an SE kernel for given inputs. ==== */
double se_kernel(double x, double y, double *theta);

/* ==== Evaluate the derivative of the SE kernel with respect to x. ==== */
double dse_kernel_dx(double x, double y, double *theta);

/* ==== Evaluate the derivative of the SE kernel with respect to y. ==== */
double dse_kernel_dy(double x, double y, double *theta);

/* ==== Evaluate the cross derivative of the SE kernel with respect to x and y. ==== */
double dse_kernel_dxdy(double x, double y, double *theta);

/* ==== Evaluate the RQ kernel for a given inputs. ==== */
double rq_kernel(double x, double y, double *theta);

/* ==== Evaluate the derivative of the RQ kernel with respect to x. ==== */
double drq_kernel_dx(double x, double y, double *theta);

/* ==== Evaluate the derivative of the RQ kernel with respect to y. ==== */
double drq_kernel_dy(double x, double y, double *theta);

/* ==== Evaluate the cross derivative of the RQ kernel with respect to x and y. ==== */
double drq_kernel_dxdy(double x, double y, double *theta);

/* ==== Evaluate the MA 1/2 kernel for a given inputs. ==== */
double ma12_kernel(double x, double y, double *theta);

/* ==== Evaluate the MA 3/2 kernel for a given inputs. ==== */
double ma32_kernel(double x, double y, double *theta);

/* ==== Evaluate the derivative of the MA 3/2 kernel with respect to x. ==== */
double dma32_kernel_dx(double x, double y, double *theta);

/* ==== Evaluate the derivative of the MA 3/2 kernel with respect to y. ==== */
double dma32_kernel_dy(double x, double y, double *theta);

/* ==== Evaluate the cross derivative of the MA 3/2 kernel with respect to x and y. ==== */
double dma32_kernel_dxdy(double x, double y, double *theta);

/* ==== Evaluate the MA 5/2 kernel for a given inputs. ==== */
double ma52_kernel(double x, double y, double *theta);

/* ==== Evaluate the derivative of the MA 5/2 kernel with respect to x. ==== */
double dma52_kernel_dx(double x, double y, double *theta);

/* ==== Evaluate the derivative of the MA 3/2 kernel with respect to y. ==== */
double dma52_kernel_dy(double x, double y, double *theta);

/* ==== Evaluate the cross derivative of the MA 5/2 kernel with respect to x and y. ==== */
double dma52_kernel_dxdy(double x, double y, double *theta);

/* ==== Evaluate the Matern kernel with fraction order nu for given inputs. ==== */
double ma_kernel(double x, double y, double *theta);

/* ==== Evaluate a Spectral Mixture kernel for a given input distance ==== */
double sse_kernel(double x, double y, double *theta, unsigned int n);

/* ==== Evaluate a spectral Matern 1/2 ==== */
double sma12_kernel(double x, double y, double *theta, unsigned int n);

/* ==== Evaluate a spectral Matern 3/2 ==== */
double sma32_kernel(double x, double y, double *theta, unsigned int n);

/* ==== Evaluate a spectral Matern 5/2 ==== */
double sma52_kernel(double x, double y, double *theta, unsigned int n);

/* ==== Evaluate a spectral RQ ==== */
double srq_kernel(double x, double y, double *theta, unsigned int n);

/* ==== Evaluate the derivative of the SM kernel with respect to x. ==== */
double dsrq_kernel_dx(double x, double y, double *theta, unsigned int n);

/* ==== Evaluate the derivative of the SM kernel with respect to y. ==== */
double dsrq_kernel_dy(double x, double y, double *theta, unsigned int n);

/* ==== Evaluate the cross derivative of the SM kernel with respect to x and y. ==== */
double dsrq_kernel_dxdy(double x, double y, double *theta, unsigned int n);

/* ==== Evaluate a spectral Matern 1/2 - Matern 1/2 ==== */
double sma12_ma12_kernel(double x, double y, double *theta, unsigned int n);

/* ==== Evaluate a spectral Matern 3/2 - Matern 1/2 ==== */
double sma32_ma12_kernel(double x, double y, double *theta, unsigned int n);

/* ==== Evaluate the derivative of the SM kernel with respect to x. ==== */
double dsma32_kernel_dx(double x, double y, double *theta, unsigned int n);

/* ==== Evaluate the derivative of the SM kernel with respect to y. ==== */
double dsma32_kernel_dy(double x, double y, double *theta, unsigned int n);

/* ==== Evaluate the cross derivative of the SM kernel with respect to x and y. ==== */
double dsma32_kernel_dxdy(double x, double y, double *theta, unsigned int n);

/* ==== Evaluate a spectral Matern 5/2 - Matern 1/2  ==== */
double sma52_ma12_kernel(double x, double y, double *theta, unsigned int n);

/* ==== Evaluate the derivative of the SM kernel with respect to x. ==== */
double dsma52_kernel_dx(double x, double y, double *theta, unsigned int n);

/* ==== Evaluate the derivative of the SM kernel with respect to y. ==== */
double dsma52_kernel_dy(double x, double y, double *theta, unsigned int n);

/* ==== Evaluate the cross derivative of the SM kernel with respect to x and y. ==== */
double dsma52_kernel_dxdy(double x, double y, double *theta, unsigned int n);

/* ==== Evaluate a Sparse Spectrum kernel for a given input distance ==== */
double ss_kernel(double x, double y, double *theta, unsigned int n);

/* ==== Evaluate the derivative of the Sparse Spectrum kernel with respect to x. ==== */
double dss_kernel_dx(double x, double y, double *theta, unsigned int n);

/* ==== Evaluate the derivative of the Sparse Spectrum kernel with respect to y. ==== */
double dss_kernel_dy(double x, double y, double *theta, unsigned int n);

/* ==== Evaluate the cross derivative of the Sparse Spectrum kernel with respect to x and y. ==== */
double dss_kernel_dxdy(double x, double y, double *theta, unsigned int n);

/* ==== Evaluate a kernel for a given inputs. ==== */
double kernel(double x, double y, double *theta, unsigned int n, char *type);

/* ==== Evaluate the derivative of a kernel with respect to x. ==== */
double dkernel_dx(double x, double y, double *theta, unsigned int n, char *type);

/* ==== Evaluate the derivative of a kernel with respect to y. ==== */
double dkernel_dy(double x, double y, double *theta, unsigned int n, char *type);

/* ==== Evaluate the cross derivative of a kernel with respect to x and y. ==== */
double dkernel_dxdy(double x, double y, double *theta, unsigned int n, char *type);

/* === Determine the number of spectral components from a PyArrayOject === */
int n_spectral_comp(PyArrayObject* theta, char*  type);

/* ==== Compute the cross-covariance matrix of a GP at X1 and X2 under a given kernel. ==== */
void cov(double **res, double *X1, unsigned int n_X1, double *X2, unsigned int n_X2, double *theta, unsigned int n_theta, char *type);

/* ==== Compute the cross-covariance matrix of a GP at X1 and X2 under a given kernel. ==== */
/* Unlike the previous function, the inputs X1 and X2 here are assumed to be multi-dimensional. The number n_col of columns of X1 and X2
	corresponds to the number of features. Each dimension has its own set of hyper-parameters. A sum of kernels is used. theta should be n_theta x n_col.
*/
void cov_multi_sum(double **res, double **X1, unsigned int n_X1, double **X2, unsigned int n_X2, double **theta, unsigned int n_theta, unsigned int n_col, char *type);

/* ==== Compute the cross-covariance matrix of a GP at X1 and X2 under a given kernel. ==== */
/* Unlike the previous function, the inputs X1 and X2 here are assumed to be multi-dimensional. The number n_col of columns of X1 and X2
	corresponds to the number of features. Each dimension has its own set of hyper-parameters. A product of kernels is used. theta should be n_theta x n_col.
*/
void cov_multi_prod(double **res, double **X1, unsigned int n_X1, double **X2, unsigned int n_X2, double **theta, unsigned int n_theta, unsigned int n_col, char *type);

/* ==== Compute the cross-covariance matrix of a derivative GP at X1 and X2 under a given kernel. ==== */
void deriv_cov(double **res, double *X1, unsigned int n_X1, double *X2, unsigned int n_X2, double *theta, unsigned int n_theta, char *type);

/* ==== Compute the cross-covariance matrix of a derivative GP at X1 and X2 under a given kernel. ==== */
/* Unlike the previous function, the inputs X1 and X2 here are assumed to be multi-dimensional. The number n_col of columns of X1 and X2
	corresponds to the number of features. Each dimension has its own set of hyper-parameters. A sum of kernels is used. theta should be n_theta x n_col.
*/
void deriv_cov_multi_sum(double **res, double **X1, unsigned int n_X1, double **X2, unsigned int n_X2, double **theta, unsigned int n_theta, unsigned int n_col, char *type);

/* ==== Compute the cross-covariance matrix of a derivative GP at X1 and X2 under a given kernel. ==== */
/* Unlike the previous function, the inputs X1 and X2 here are assumed to be multi-dimensional. The number n_col of columns of X1 and X2
	corresponds to the number of features. Each dimension has its own set of hyper-parameters. A product of kernels is used. theta should be n_theta x n_col.
*/
void deriv_cov_multi_prod(double **res, double **X1, unsigned int n_X1, double **X2, unsigned int n_X2, double **theta, unsigned int n_theta, unsigned int n_col, char *type);

/* ==== Evaluate the periodic kernel (McKay) for given inputs. ==== */
double period_kernel(double x, double y, double *theta);

/* ==== Evaluate the derivative of the periodic kernel (McKay) with respect to x. ==== */
double dperiod_kernel_dx(double x, double y, double *theta);

/* ==== Evaluate the derivative of the periodic kernel (McKay) with respect to y. ==== */
double dperiod_kernel_dy(double x, double y, double *theta);

/* ==== Evaluate the cross derivative of the periodic kernel (McKay) with respect to x and y. ==== */
double dperiod_kernel_dxdy(double x, double y, double *theta);

/* ==== Evaluate the locally periodic kernel (McKay) for given inputs. ==== */
double loc_period_kernel(double x, double y, double *theta);

/* ==== Evaluate the derivative of the locally periodic kernel (McKay) with respect to x. ==== */
double dloc_period_kernel_dx(double x, double y, double *theta);

/* ==== Evaluate the derivative of the locally periodic kernel (McKay) with respect to y. ==== */
double dloc_period_kernel_dy(double x, double y, double *theta);

/* ==== Evaluate the cross derivative of the locally periodic kernel (McKay) with respect to x and y. ==== */
double dloc_period_kernel_dxdy(double x, double y, double *theta);



/* ==== Compute the covariance matrix of a string derivative GP at string times, under a given kernel. ==== */
/* 
	res: pointer to the matrix to be updated in place. The matrix should be of size 2*n_s_times x 2*n_s_times.
	n_s_times: array of string times.
	b_times: array of boundary times, should be sorted (the function will sort it otherwise, and the matrix indices will correspond to sorted boundary times).
	n_b_times: number of boundary times. Also 1 + the number of strings.
	thetas: matrix of string hyper-parameters.
	n_mixts: Number of elements in the spectral mixture for each local expert kernel.
	type: type of kernel (se, rq, ma32, ma52, sm)
*/
void string_deriv_cov(double **res, double *s_times, unsigned int n_s_times, double *b_times, unsigned int n_b_times, double **thetas, unsigned int n_mixts, char *type);


/* ==== Compute the covariance matrix of a string GP at string times, under a given kernel. ==== */
/* 
	res: pointer to the matrix to be updated in place. The matrix should be of size n_s_times x n_s_times.
	n_s_times: array of string times.
	b_times: array of boundary times, should be sorted (the function will sort it otherwise, and the matrix indices will correspond to sorted boundary times).
	n_b_times: number of boundary times. Also 1 + the number of strings.
	thetas: matrix of string hyper-parameters.
	n_mixts: Number of elements in the spectral mixture for each local expert kernel.
	type: type of kernel (se, rq, ma32, ma52, sm)
*/
void string_cov(double **res, double *s_times, unsigned int n_s_times, double *b_times, unsigned int n_b_times, double **thetas, unsigned int n_mixts, char *type);


/* ==== Compute the cross-covariance matrix of a GP at X1 and X2 under a Spectral Mixture kernel. ==== */
void sm_cov(double **res, double *X1, unsigned int n_X1, double *X2, unsigned int n_X2, double *theta, unsigned int n_theta);

/* ==== Compute the cross-covariance matrix of a derivative GP at X1 and X2 under a Spectral Mixture kernel. ==== */
void sm_deriv_cov(double **res, double *X1, unsigned int n_X1, double *X2, unsigned int n_X2, double *theta, unsigned int n_theta);

/* ==== Compute the covariance matrix of a string derivative GP at boundary times, under a given kernel. ==== */
/* 
	res: pointer to the matrix to be updated in place. The matrix should be of size 2*n_b_times x 2*n_b_times.
	b_times: array of boundary times, should be sorted (the function will sort it otherwise, and the matrix indices will correspond to sorted boundary times).
	n_b_times: number of boundary times. Also 1 + the number of strings.
	thetas: matrix of string hyper-parameters.
	n_mixts: array of numbers of elements in the spectral mixture for each local expert kernel.
	type: kernel type (se, sm, rq, ma32, ma52)
*/
void string_boundaries_deriv_cov(double **res, double *b_times, unsigned int n_b_times, double **thetas, unsigned int n_mixts, char *type);

/* ==== Compute the covariance matrix of a string derivative GP at boundary times, under a spectral mixture kernel. ==== */
/* 
	res: pointer to the matrix to be updated in place.
	b_times: array of boundary times, assumed sorted and uniqued.
	n_b_times: number of boundary times. Also 1 +  the number of strings.
	thetas: matrix of string hyper-parameters.
	n_thetas: array of numbers of hyper-parameters of each local expert kernel.
*/
void string_sm_boundaries_deriv_cov(double **res, double *b_times, unsigned int n_b_times, double **thetas, unsigned int n_mixts);


/* ==== Compute the covariance matrix of a string derivative GP at string times, under a spectral mixture kernel. ==== */
/* 
	res: pointer to the matrix to be updated in place. The matrix should be of size 2*n_s_times x 2*n_s_times.
	n_s_times: array of string times.
	b_times: array of boundary times, should be sorted (the function will sort it otherwise, and the matrix indices will correspond to sorted boundary times).
	n_b_times: number of boundary times. Also 1 + the number of strings.
	thetas: matrix of string hyper-parameters.
	n_mixts: array of numbers of elements in the spectral mixture for each local expert kernel.
*/
void string_sm_deriv_cov(double **res, double *s_times, unsigned int n_s_times, double *b_times, unsigned int n_b_times, double **thetas, unsigned int n_mixts);
#endif