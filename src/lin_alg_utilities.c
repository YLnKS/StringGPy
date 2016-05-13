#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL cStringGPy_ARRAY_API

#include "lin_alg_utilities.h"


/*
  Utility function to print a matrix
*/
void printf_matrix(double **A, unsigned int n, unsigned int m){
  int i=0, j=0;

  for(i=0; i<n; i++){
    printf("[");
    for(j=0; j<m; j++){
      printf("%.6f, ", A[i][j]);
    }
    printf("]\n");
  }
}

/*
  Utility function to print a vector
*/
void printf_vector(double *A, unsigned int n){
  int i=0;

    printf("[");
    for(i=0; i<n; i++){
     	printf("%.6f, ", A[i]);
    }
    printf("]\n");
}

/*
	Utility function to use with qsort to compare doubles
*/
int cmp_double(const void *x, const void *y)
{
  double xx = *(double*)x, yy = *(double*)y;
  if (xx < yy) return -1;
  if (xx > yy) return  1;
  return 0;
}

/*
   Return a new n x m matrix of with zeros.
*/
double ** new_zeros(int n, int m){
	double ** res=NULL;
	int i,j;
	res=malloc(n*sizeof(double *));
	for (i=0;i<n;i++) {
		res[i]= malloc(m*sizeof(double));
	}
	for (i=0;i<n;i++) {
		for (j=0;j<m;j++){
			res[i][j]=0.0;
		}
	}
	return res;
}

/*
   Free the memory occupied by a matrix represented as pointer of pointers.
	The number of columns does not matter.
*/
void free_mem_matrix(double **A, int n_rows){
	int i;
	for(i = 0; i < n_rows; i++)
		free(A[i]);
	free(A);	
}


/*
   Recursive definition of determinate using expansion by minors.
*/
double determinant(double **A,int n)
{
   int i,j,j1,j2;
   double det = 0;
   double **m = NULL;

   if (n < 1) { /* Error */

   } else if (n == 1) { /* Shouldn't get used */
      det = A[0][0];
   } else if (n == 2) {
      det = A[0][0] * A[1][1] - A[1][0] * A[0][1];
   } else {
      det = 0;
      for (j1=0;j1<n;j1++) {
         m = malloc((n-1)*sizeof(double *));
         for (i=0;i<n-1;i++)
            m[i] = malloc((n-1)*sizeof(double));
         for (i=1;i<n;i++) {
            j2 = 0;
            for (j=0;j<n;j++) {
               if (j == j1)
                  continue;
               m[i-1][j2] = A[i][j];
               j2++;
            }
         }
         det += pow(-1.0,j1+2.0) * A[0][j1] * determinant(m,n-1);
         for (i=0;i<n-1;i++)
            free(m[i]);
         free(m);
      }
   }
   return(det);
}

/*
   Find the cofactor matrix of a square matrix
*/
void co_factor(double **res, double **A, int n)
{
   int i,j,ii,jj,i1,j1;
   double det;
   double **c;

   c = malloc((n-1)*sizeof(double *));
   for (i=0;i<n-1;i++)
     c[i] = malloc((n-1)*sizeof(double));

   for (j=0;j<n;j++) {
      for (i=0;i<n;i++) {

         /* Form the adjoint a_ij */
         i1 = 0;
         for (ii=0;ii<n;ii++) {
            if (ii == i)
               continue;
            j1 = 0;
            for (jj=0;jj<n;jj++) {
               if (jj == j)
                  continue;
               c[i1][j1] = A[ii][jj];
               j1++;
            }
            i1++;
         }

         /* Calculate the determinate */
         det = determinant(c,n-1);

         /* Fill in the elements of the cofactor */
         res[i][j] = pow(-1.0,i+j+2.0) * det;
      }
   }
   for (i=0;i<n-1;i++)
      free(c[i]);
   free(c);
}

/*
   Transpose of a square matrix, do it in place.
*/
void transpose(double **A,int n)
{
   int i,j;
   double tmp;

   for (i=1;i<n;i++) {
      for (j=0;j<i;j++) {
         tmp = A[i][j];
         A[i][j] = A[j][i];
         A[j][i] = tmp;
      }
   }
}

/*
	Computes (in place) the inverse of a square matrix user Cramer rules.
*/
void invert(double **res, double **A, int n){
	double det, eps=0.01, min_det=0.000000001;
	unsigned int i,j,k;
	
	det = determinant(A, n);
	k=0; 
	while((det == 0.0) & (k < 50)){
		// Improve conditioning by adding a jitter on the diagonal.
		for(i=0; i<n; i++){
			A[i][i] += eps;
		}
		det = determinant(A, n);
		k += 1;
	}
	// Safety measure.
	if(det < min_det)
		det = min_det;
	
	assert(det != 0.0);
	// Compute the co-factor in place.
	co_factor(res, A, n);
	// Transpose the co-factor in place.
	transpose(res, n);
	// Divide the transpose of the co-factor by the determinant.
	for (i=0; i<n; i++) {
		for (j=0; j<n; j++) {
			res[i][j]=res[i][j]/det;
		}
	}
}

/*
	Computes (in place) the inverse of a 2x2 matrix.
*/
void invert2(double **res, double **A){
	double det, eps=0.0001;
	unsigned int i;
	
	det = A[0][0]*A[1][1] - A[0][1]*A[1][0];
	if(det == 0.0){
		// Improve conditioning by adding a jitter on the diagonal.
		for(i=0; i<2; i++){
			A[i][i] += eps;
		}
		det = A[0][0]*A[1][1] - A[0][1]*A[1][0];
	}
	
	assert(det != 0.0);
	res[0][0] = A[1][1]/det;
	res[0][1] = -A[0][1]/det;
	res[1][0] = -A[1][0]/det;
	res[1][1] = A[0][0]/det;
	return;
}

/*
   Transpose of a n x m matrix, and store the result in res.
*/
void transpose_copy(double **res, double **A, int n, int m){
	int i,j;
	for(i=0; i<n; i++){
		for(j=0; j<m; j++){
			res[j][i]=A[i][j];
		}
	}
	return;
}

/* 
	Computes (in place) the matrix product between two matrix A and B, of size n_A x m and m x m_B respectively.
*/
void matrix_prod(double **res, double **A, unsigned int n_A, unsigned int m, double **B, unsigned int m_B){
	int i,j,k;
	for (i=0; i<n_A; i++) {
		for(j=0; j<m_B; j++) {
			res[i][j]=0.0;
			for(k=0; k<m; k++){
				res[i][j] += A[i][k]*B[k][j];
			}
		}
	}
	// Return the pointer. Remember to free the data memory (n_A x m_B) when done!
	return;
}

/* 
	res = A+B; where both A and B are n x m matrices.
*/
void matrix_add(double **res, double **A, double **B, int n, int m){
	int i,j;
	for(i=0; i<n; i++){
		for(j=0; j<m; j++){
			res[i][j]=A[i][j]+B[i][j];
		}
	}
	return;
}

/* 
	res = A-B; where both A and B are n x m matrices.
*/
void matrix_sub(double **res, double **A, double **B, int n, int m){
	int i,j;
	for(i=0; i<n; i++){
		for(j=0; j<m; j++){
			res[i][j]=A[i][j]-B[i][j];
		}
	}
	return;
}

/*
   Take a vector 'col' of size n_row and create an n_row x n_col matrix with columns identical to col.
	This is done in place and the result is stored in res.

	col is assumed to be row-major.
*/
void reshape_col(double **res, double *col, unsigned int n_row, unsigned int n_col){
	int i,j;
	for(i=0; i<n_row; i++){
		for(j=0; j<n_col; j++){
			res[i][j]=col[j+n_row*i];
		}
	}
}

/* 
   orig=chol*chol^T, chol lower triangular.
 */
int cholesky(double **orig, int n, double **chol)
{
  int i, j, k;
  int retval = 1;

  for (i=0; i<n; i++) {

    chol[i][i] = orig[i][i];
    for (k=0; k<i; k++)
		  chol[i][i] -= chol[i][k]*chol[i][k];
    	
    if (chol[i][i] <= 0) {
			retval = 0;
			return retval;
		}
    
    chol[i][i] = sqrt(chol[i][i]);

    for (j=i+1; j<n; j++) {
			chol[j][i] = orig[j][i];

		  for (k=0; k<i; k++)
        chol[j][i] -= chol[i][k]*chol[j][k];
		  chol[j][i] /= chol[i][i];
    }
  }

   return retval;
}

// Cholesky factorisation for 2x2 matrices
int cholesky2(double **orig, double **chol){
	chol[0][0]=sqrt(orig[0][0]);
	chol[1][0]=orig[1][0]/chol[0][0];

  if(orig[1][1] - chol[1][0]*chol[1][0]<0)
    return 0;

	chol[1][1]=sqrt(orig[1][1] - chol[1][0]*chol[1][0]);
  return 1;
}

/*
  Converts a n-by-m matrix of double into a column-major row with nxm elements. 
*/
void to_column_major(double **mat, double *col, unsigned int n, unsigned int m){
  int i=0, j=0;

  for(i=0; i<n; i++){
    for(j=0; j<m; j++){
      col[i+n*j]=mat[i][j];
    }
  }
}

/*
  Converts a column-major row with nxm elements into a n-by-m matrix of double. 
*/
void from_column_major(double **mat, double *col, unsigned int n, unsigned int m){
  int i=0, j=0;

  for(i=0; i<n; i++){
    for(j=0; j<m; j++){
      mat[i][j]=col[i+n*j];
    }
  }
}

/*
  Converts a n-by-m matrix of double into a column-major row with nxm elements. 
*/
void to_row_major(double **mat, double *col, unsigned int n, unsigned int m){
  int i=0, j=0;

  for(i=0; i<n; i++){
    for(j=0; j<m; j++){
      col[j+i*m]=mat[i][j];
    }
  }
}

/*
  Converts a column-major row with nxm elements into a n-by-m matrix of double. 
*/
void from_row_major(double **mat, double *col, unsigned int n, unsigned int m){
  int i=0, j=0;

  for(i=0; i<n; i++){
    for(j=0; j<m; j++){
      mat[i][j]=col[j+i*m];
    }
  }
}


/*
  cov=USV with cov 2x2 and U, V orthogonal.
*/
void svd2(double **cov, double **U, double *S, double **V){
	double e=0.0, f=0.0, g=0.0, h=0.0, q=0.0, r=0.0;
	double theta=0.0, phi=0.0, a1=0.0, a2=0.0;

	e = (cov[0][0] + cov[1][1])/2.0;
	f = (cov[0][0] - cov[1][1])/2.0;
	g = (cov[1][0] + cov[0][1])/2.0;
	h = (cov[1][0] - cov[0][1])/2.0;
	q = sqrt(e*e + h*h);
	r = sqrt(f*f + g*g);
	
	S[0] = fmaxf(q + r, 0.0);
	S[1] = fmaxf(q - r, 0.0);
	
	a1 = atan2(g, f);
	a2 = atan2(h, e);
	theta = (a2 - a1)/2.0;
	phi = (a2 + a1)/2.0;

	U[0][0] = cos(phi);
	U[0][1] = -sin(phi);
	U[1][0] = sin(phi);
	U[1][1] = cos(phi);

	V[0][0] = cos(theta);
	V[0][1] = -sin(theta);
	V[1][0] = sin(theta);
	V[1][1] = cos(theta);
}

/*
  cov=USV, U, V orthogonal
*/
void svd(double **cov, double **U, double *S, double **V, unsigned int n){
  int info = 0;
  double *_cov=NULL, *_U=NULL, *_V=NULL, *work=NULL; 

  if(n==2){
	svd2(cov, U, S, V);
  }
  else{
  	PyErr_SetString(PyExc_ValueError, "LAPACK SVD is not yet supported");
  }
}

/*
	Returns L=US^{1/2}, where cov=USU.T is the SVD of cov.
*/
void svd_l(double **cov, int n, double **L){
	double **U=NULL, *S=NULL, **V=NULL;
	int i=0, j=0;

	U=(double **)new_zeros(n, n);
	V=(double **)new_zeros(n, n);
	S=malloc(n*sizeof(double));

	for(i=0; i<n; i++)
		S[i]=0.0;

  	svd(cov, U, S, V, n); // Compute the SVD of cov in-place

	for(i=0; i<n; i++){
		for(j=0; j<n; j++){
			L[i][j]=U[i][j]*sqrt(S[j]);
		}
	}

	free(S);
	free_mem_matrix((double **)U, n);
	free_mem_matrix((double **)V, n);
}

/*
	Compute a matrix L such that cov=LL^T. 
		Start with Cholesky factorisation if possible
		and revert to SVD if it fails.
*/
int l_factor(double **cov, int n, double **L){
	int status=0;
  status=cholesky(cov, n, L);
	if(status==0)
		svd_l(cov, n, L);

	return status;
}

/* === Sample a multivariate Gaussian in place. === */
void multivariate_normal(double **result, double **cov, int n, double **L){
	double **stdn;
	int was_null=0;

	// Use the Cholesky factorisation if it is provided
	if(L==NULL){
		was_null=1;
		// Cholesky factor
		L=new_zeros(n, n);
		l_factor(cov, n, L);
	}

	// Sample 2 i.i.d. standard normal
	stdn=randn(n); // Remember to free this	
	matrix_prod(result, L, n, n, stdn, 1); // As stdn is i.i.d standard normal, L*stdn is normal with mean 0 and cov L*L^T

	// Cleanup
	if(was_null==1)
		free_mem_matrix(L, n);
	free_mem_matrix(stdn, n);
}


/* === Draw n i.i.d standard normal and return a ndarray (PyArrayObject). === */
double **randn(int n){
	double **result;
	result=new_zeros(n, 1);
	
	int i=0;
	double current, next, u, v=0.0;

	if(n==1){
		u=((double)rand()/(double)RAND_MAX);
		v=((double)rand()/(double)RAND_MAX);
		current = sqrt(-2.0*log(u))*cos(2.0*M_PI*v);
		result[0][0]=current;
	}
	else{
		for(i=0; i<n/2; i++){
			u=((double)rand()/(double)RAND_MAX);
			v=((double)rand()/(double)RAND_MAX);
			current = sqrt(-2.0*log(u))*cos(2.0*M_PI*v);
			next = sqrt(-2.0*log(u))*sin(2.0*M_PI*v);
			result[2*i][0] = current;
			result[2*i+1][0] = next;
		}
	}

	if(2*(i+1)<n){
		u=((double)rand()/(double)RAND_MAX);
		v=((double)rand()/(double)RAND_MAX);
		current = sqrt(-2.0*log(u))*cos(2.0*M_PI*v);
		next = sqrt(-2.0*log(u))*sin(2.0*M_PI*v);
		result[2*i+2][0] = current;
	}

	// Remember to free result
	return result;
}

/*
	Computes the jitter to use to adjust the diagonal of a possibly ill-conditioned 
		positive semi-definite matrix.
*/
double ill_cond_jit(double *eig_vals, int n){
	double min_eig=fabs(eig_vals[0]), max_eig=fabs(eig_vals[0]), oc=1.0, nc=1.0, eps=0.0;
	int i=0;

	for(i=1; i<n; i++){
		min_eig=fminf(min_eig, fabs(eig_vals[i]));
		max_eig=fmaxf(max_eig, fabs(eig_vals[i]));
	}

	// The maximum conditioning number allowed is MAX_CN
	oc=max_eig/min_eig;
	if(oc > MAX_CN){
		//printf("Warning: Cond. Numb. pretty bad -- %f!\n", oc);
		nc = fminf(oc, MAX_CN);
		eps = min_eig*(oc-nc)/(nc-1.0);
	}
	return eps;
}

/*
	Computes (in place) the inverse of a square matrix using SVD decomposition and adjusting for ill-conditioning.
*/
void invert_robust(double **cov, int n){
  double **U=NULL, *S=NULL, **V=NULL, **D=NULL, **tmp=NULL, eps=0.0;
  int i=0, j=0;

  U=new_zeros(n, n);
  V=new_zeros(n, n);

  for(i=0; i<n; i++){
    for(j=0; j<n; j++){
      U[i][j]=cov[i][j];
    }    
  }
  S=malloc(n*sizeof(double));
  svd(cov, U, S, V, n); // Compute the SVD of cov in-place
  eps=ill_cond_jit(S, n); // Compute the (possibly null) jitter to add to the eigenvalues to cap the conditioning number to 1e8.

  D=new_zeros(n, n);
  for(i=0; i<n; i++){
    D[i][i]=1.0/(eps+S[i]); 
  }

  tmp=new_zeros(n, n);
  matrix_prod(tmp, U, n, n, D, n);
  matrix_prod(cov, tmp, n, n, V, n); // cov^{-1}=U(S+eps)^{-1}V

  // Cleanup
  free(S);
  free_mem_matrix(U, n);
  free_mem_matrix(V, n);
  free_mem_matrix(D, n);
  free_mem_matrix(tmp, n);
}

/* 
	Initialise random seeds
*/
time_t random_seed(){
	struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);

    // Seed random generator with nanosecond time.
   	return (time_t)(ts.tv_nsec + getpid());
}