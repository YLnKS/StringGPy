#ifndef LIN_ALG_UTILITIES_H
#define LIN_ALG_UTILITIES_H
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL cStringGPy_ARRAY_API

#include <Python.h>
#include <math.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/types.h> 
#include <unistd.h>

#define MAX_CN 1000000.0

/*
	Utility function to print a matrix
*/
void printf_matrix(double **A, unsigned int n, unsigned int m);

/*
  Utility function to print a vector
*/
void printf_vector(double *A, unsigned int n);

/*
	Utility function to use with qsort to compare doubles
*/
int cmp_double(const void *x, const void *y);

/*
   Recursive definition of determinate using expansion by minors.
*/
double determinant(double **A,int n);

/*
   Find the cofactor matrix of a square matrix
*/
void co_factor(double **res, double **A,int n);

/*
   Transpose of a square matrix, do it in place.
*/
void transpose(double **A,int n);

/*
   Transpose of a n x m matrix, and return a copy.
*/
void transpose_copy(double **res, double **A, int n, int m);

/*
	Computes (in place) the inverse of a square matrix using Cramer rules.
*/
void invert(double **res, double **A, int n);

/*
	Computes (in place) the inverse of a 2x2 matrix.
*/
void invert2(double **res, double **A);

/* 
	Compute (in place) the matrix product between two matrix A and B, of size n_A x m and m x m_B respectively. 	
*/
void matrix_prod(double **res, double **A, unsigned int n_A, unsigned int m, double **B, unsigned int m_B);

/* 
	res = A+B; where both A and B are n x m matrices.
*/
void matrix_add(double **res, double **A, double **B, int n, int m);

/* 
	res = A-B; where both A and B are n x m matrices.
*/
void matrix_sub(double **res, double **A, double **B, int n, int m);

/*
   Return a new n x m matrix of with zeros.
*/
double ** new_zeros(int n, int m);

/*
   Free the memory occupied by a matrix represented as pointer of pointers.
	The number of columns does not matter.
*/
void free_mem_matrix(double **A, int n_rows);

/*
   Take a vector 'col' of size n_row and create an n_row x n_col matrix with columns identical to col.
	This is done in place and the result is stored in res.
*/
void reshape_col(double **res, double *col, unsigned int n_row, unsigned int n_col);

/*
	Performs a Cholesky factorisation in place, and return 0 in case the matrix isn't positive semi-definite.
*/
int cholesky(double **orig, int n, double **chol);

/*
	Cholesky factorisation for 2x2 matrices.
*/
int cholesky2(double **orig, double **chol);


/*
  Converts a n-by-m matrix of double into a column-major row with nxm elements. 
*/
void to_column_major(double **from, double *to, unsigned int n, unsigned int m);

/*
  Converts a column-major row with nxm elements into a n-by-m matrix of double. 
*/
void from_column_major(double **to, double *from, unsigned int n, unsigned int m);

/*
  Converts a n-by-m matrix of double into a row-major row with nxm elements. 
*/
void to_row_major(double **from, double *to, unsigned int n, unsigned int m);

/*
  Converts a column-major row with nxm elements into a n-by-m matrix of double. 
*/
void from_row_major(double **to, double *from, unsigned int n, unsigned int m);

/*
  cov=USV
*/
void svd(double **cov, double **U, double *S, double **V, unsigned int n);

/*
  cov=USV with cov 2x2 and U, V orthogonal.
*/
void svd2(double **cov, double **U, double *S, double **V);

/*
	Computes the jitter to use to adjust the diagonal of a possibly ill-conditioned 
		positive semi-definite matrix.
*/
double ill_cond_jit(double *eig_vals, int n);

/*
	Returns L=US^{1/2}, where cov=USU.T is the SVD of cov (from numpy.dual.svd)
*/
void svd_l(double **cov, int n, double **L);

/*
	Compute a matrix L such that cov=LL^T. 
		Start with Cholesky factorisation if possible
		and revert to SVD if it fails.
*/
int l_factor(double **cov, int n, double **L);

/* === Draw n i.i.d standard normal and return a ndarray (PyArrayObject). === */
double **randn(int n);

/* === Sample a multivariate Gaussian in place. === */
void multivariate_normal(double **result, double **cov, int n, double **L);

/*
	Computes (in place) the inverse of a square matrix using SVD decomposition and adjusting for ill conditioning.
*/
void invert_robust(double **cov, int n); 

/* 
	Initialise random seeds
*/
time_t random_seed(void);
#endif