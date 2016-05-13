#include "kernel_utilities.h"

/*
   Return a new n x m matrix of with zeros.
*/
static double ** c_new_zeros(int n, int m){
    return new_zeros(n, m);

}

/*
   Free the memory occupied by a matrix represented as pointer of pointers.
    The number of columns does not matter.
*/
static void c_free_mem_matrix(double **A, int n_rows){
    free_mem_matrix(A, n_rows);
}

/* ==== Compute the cross-covariance matrix of a derivative GP at X1 and X2 under a given kernel. ==== */
static void c_deriv_cov(double **res, double *X1, unsigned int n_X1, double *X2, unsigned int n_X2, double *theta, unsigned int n_theta, char *type){
    deriv_cov(res, X1, n_X1, X2, n_X2, theta, n_theta, type);
}

/* === Determine the number of spectral components from a PyArrayObject === */
static int c_n_spectral_comp(int n, char*  type){
    int n_mixt;
    if(strcmp(type, "srq") == 0){
        n_mixt = n/4;
    }else{
        n_mixt = n/3;
    }
    return n_mixt;
}

/*
  cov=USV with cov 2x2 and U, V orthogonal.
*/
static void c_svd2(double **cov, double **U, double *S, double **V){
    svd2(cov, U, S, V);   
}

/*
    Computes (in place) the inverse of a square matrix using SVD decomposition and adjusting for ill conditioning.
*/
static void c_invert_robust(double **cov, int n){
    invert_robust(cov, n);
}

/* 
    Compute (in place) the matrix product between two matrix A and B, of size n_A x m and m x m_B respectively.     
*/
static void c_matrix_prod(double **res, double **A, unsigned int n_A, unsigned int m, double **B, unsigned int m_B){
    matrix_prod(res, A, n_A, m, B, m_B);
}

/* 
    res = A+B; where both A and B are n x m matrices.
*/
static void c_matrix_add(double **res, double **A, double **B, int n, int m){
    matrix_add(res, A, B, n, m);
}

/* 
    res = A-B; where both A and B are n x m matrices.
*/
static void c_matrix_sub(double **res, double **A, double **B, int n, int m){
    matrix_sub(res, A, B, n, m);
}

/*
    Utility function to print a matrix
*/
static void c_printf_matrix(double **A, unsigned int n, unsigned int m){
    printf_matrix(A, n, m);   
}

/*
  Utility function to print a vector
*/
static void c_printf_vector(double *A, unsigned int n){
    printf_vector(A, n);
}

/*
  Utility function to shuffle the order of a vector in a specfic order
*/
static void c_shuffle(double* inp, double* out, unsigned long* ind, unsigned long n){
    unsigned long i;
    for(i=0; i<n; i++){
        out[i] = inp[ind[i]];
    }
}


/*
  Utility function to detect duplicates in a sorted array.
*/
static void c_mark_duplicate(double* inp, unsigned int* dup, unsigned long n){
    if(n>0){
        unsigned long i;
        dup[0] = 0;
        for(i=1; i<n; i++){
            if(inp[i] == inp[i-1]){
                dup[i] = 1;
            }else{
                dup[i] = 0;
            }
        }
    }  
}


/*
    Row-major to flat index (3D n1 x n2 x n3 array)
*/
static int flat_rm_index3(int i, int j, int k, int n2, int n3){
    return k + n3*(j + i*n2);
}

/*
    Row-major to flat index (2D n x d array)
*/
static int flat_rm_index2(int i, int j, int d){
    return j + i*d;
}


/*
    Column-major to flat index (3D n1 x n2 x n3 array)
*/
static int flat_cm_index3(int i, int j, int k, int n1, int n2){
    return i + n1*(j + k*n2);
}

/*
    Column-major to flat index (2D n x d array)
*/
static int flat_cm_index2(int i, int j, int n){
    return i + j*n;
}


/*
    Utility function to compute conditional mean and covariance matrix factors.
        X and l_hypers should be column-major. l_factors and m_factors should be row-major.
*/
static void c_factors(int i, int j, int n, int d, double* l_factors, double* m_factors, double* l_hypers,
        double* X, int n_theta, char* k_type){
    int n_mixt = c_n_spectral_comp(n_theta, k_type);
    int k1, k2;
    double **cov_tt, **cov_ptt, **cov_tpt, **cov_ptpt, **cov_tgpt, **m, **U, **V, *S, **_tmp1, **_tmp2, *t, *pt, *theta;
    int prev_i;

    theta = &l_hypers[flat_cm_index3(0,i,j,n_theta,n)];
    t = (double*) malloc(sizeof(double));
    pt = (double*) malloc(sizeof(double));
    t[0] = X[flat_cm_index2(i,j,n)];

    if(i == 0){
        U = new_zeros(2, 2);
        V = new_zeros(2, 2);
        S = (double*) malloc(2*sizeof(double));
        S[0] = 0.0;
        S[1] = 0.0;

        cov_tt = new_zeros(2, 2);
        deriv_cov(cov_tt, t, 1, t, 1, theta, n_mixt, k_type);
        svd2(cov_tt, U, S, V); // Compute the SVD of cov in-place

        for (k1= 0; k1<2; k1++){
            for (k2= 0; k2<2; k2++){
                l_factors[flat_rm_index3(k1,k2,j,2*n,d)] = U[k1][k2]*sqrt(S[k2]);
                m_factors[flat_rm_index3(k1,k2,j,2*n,d)] = 0.0;
            }
        }
        free(S);
        S=NULL;
        free_mem_matrix(U, 2);
        free_mem_matrix(V, 2);
        free_mem_matrix(cov_tt, 2);
    }
    else{
        // Handle cases where there are two points have the same coordinates in
        //   some dimension.
        prev_i = i-1;
        while((prev_i > 0) && (X[flat_cm_index2(prev_i,j,n)] == X[flat_cm_index2(i,j,n)])){
            prev_i -= 1;
        }

        t[0] = X[flat_cm_index2(i, j, n)];
        pt[0] = X[flat_cm_index2(prev_i, j, n)];

        if (X[flat_cm_index2(prev_i,j,n)] == X[flat_cm_index2(i,j,n)]) {
            U = new_zeros(2, 2);
            V = new_zeros(2, 2);
            S = (double*) malloc(2*sizeof(double));
            S[0] = 0.0;
            S[1] = 0.0;

            cov_tt = new_zeros(2, 2);
            deriv_cov(cov_tt, t, 1, t, 1, theta, n_mixt, k_type);
            svd2(cov_tt, U, S, V); // Compute the SVD of cov in-place

            for (k1= 0; k1<2; k1++){
                for (k2= 0; k2<2; k2++){
                    l_factors[flat_rm_index3(k1,k2,j,2*n,d)] = U[k1][k2]*sqrt(S[k2]);
                    m_factors[flat_rm_index3(k1,k2,j,2*n,d)] = 0.0;
                }
            }
            free(S);
            S=NULL;
            free_mem_matrix(U, 2);
            free_mem_matrix(V, 2);
            free_mem_matrix(cov_tt, 2);
        }else{
            cov_tt   = new_zeros(2, 2);
            cov_tpt  = new_zeros(2, 2);
            cov_ptt  = new_zeros(2, 2);
            cov_ptpt = new_zeros(2, 2);

            deriv_cov(cov_tt, t, 1, t, 1, theta, n_mixt, k_type);
            deriv_cov(cov_ptt, pt, 1, t, 1, theta, n_mixt, k_type);
            deriv_cov(cov_tpt, t, 1, pt, 1, theta, n_mixt, k_type);
            deriv_cov(cov_ptpt, pt, 1, pt, 1, theta, n_mixt, k_type);
            invert_robust(cov_ptpt, 2);

            _tmp1 = new_zeros(2, 2);
            _tmp2 = new_zeros(2, 2);
            matrix_prod(_tmp1, cov_ptpt, 2, 2, cov_ptt, 2);
            matrix_prod(_tmp2, cov_tpt, 2, 2, _tmp1, 2);
            matrix_sub(_tmp1, cov_tt, _tmp2, 2, 2);

            U = new_zeros(2, 2);
            V = new_zeros(2, 2);
            S = (double*) malloc(2*sizeof(double));

            svd2(_tmp1, U, S, V); // Compute the SVD in-place.
            for (k1= 0; k1<2; k1++){
                for (k2= 0; k2<2; k2++){
                    l_factors[flat_rm_index3(k1,2*i+k2,j,2*n,d)] = U[k1][k2]*sqrt(S[k2]);
                }
            }
            m = new_zeros(2, 2);
            matrix_prod(m, cov_tpt, 2, 2, cov_ptpt, 2);
            for (k1= 0; k1<2; k1++){
                for (k2= 0; k2<2; k2++){
                    m_factors[flat_rm_index3(k1,2*i+k2,j,2*n,d)] = m[k1][k2];
                }
            }
            free(S);
            S=NULL;
            free_mem_matrix(U, 2);
            free_mem_matrix(V, 2);
            free_mem_matrix(cov_tt, 2);
            free_mem_matrix(cov_tpt, 2);
            free_mem_matrix(cov_ptt, 2);
            free_mem_matrix(cov_ptpt, 2);
            free_mem_matrix(m, 2);
            free_mem_matrix(_tmp1, 2);
            free_mem_matrix(_tmp2, 2);
        }
    }
    free(t);
    free(pt);
}


static double c_gamma(double x){
    return tgamma(x);
}

/* ==== Compute the covariance matrix of a string GP at string times, under a given expert kernel. ==== */
/* 
    res: pointer to the matrix to be updated in place. The matrix should be of size n_s_times x n_s_times, contiguous and row-major.
    n_s_times: array of string times.
    b_times: array of boundary times, should be sorted (the function will sort it otherwise, and the matrix indices will correspond to sorted boundary times).
    n_b_times: number of boundary times. Also 1 + the number of strings.
    thetas: matrix of string hyper-parameters of size (n_b_times - 1) x n_theta, contiguous and row-major.
    n_mixts: Number of elements in the spectral mixture for each local expert kernel.
    k_type: type of kernel (se, rq, ma32, ma52, sm)
*/
static void c_string_cov(double *res, double *s_times, unsigned int n_s_times, double *b_times,\
        unsigned int n_b_times, double *thetas, unsigned int n_theta, char *k_type){
    
    int n_mixt = c_n_spectral_comp(n_theta, k_type);

    // From double pointer (row-major) to pointer of double pointer
    double **res_d_pt = new_zeros(n_s_times, n_s_times);

    // From double pointer (row-major) to pointer of double pointer
    double **thetas_d_pt = new_zeros(n_b_times-1, n_theta);
    from_row_major(thetas_d_pt, thetas, n_b_times-1, n_theta);

    // Main call
    string_cov(res_d_pt, s_times, n_s_times, b_times, n_b_times, thetas_d_pt, n_mixt, k_type);

    // Copy the result back to the memory of the numpy array
    to_row_major(res_d_pt, res, n_s_times, n_s_times);

    // Free local pointers of pointers
    free_mem_matrix(res_d_pt, n_s_times);
    free_mem_matrix(thetas_d_pt, n_b_times-1);
}

static void c_string_deriv_cov(double *res, double *s_times, unsigned int n_s_times, double *b_times,\
        unsigned int n_b_times, double *thetas, unsigned int n_theta, char *k_type){
    
    int n_mixt = c_n_spectral_comp(n_theta, k_type);

    // From double pointer (row-major) to pointer of double pointer
    double **res_d_pt = new_zeros(2*n_s_times, 2*n_s_times);

    // From double pointer (row-major) to pointer of double pointer
    double **thetas_d_pt = new_zeros(n_b_times-1, n_theta);
    from_row_major(thetas_d_pt, thetas, n_b_times-1, n_theta);

    // Main call
    string_deriv_cov(res_d_pt, s_times, n_s_times, b_times, n_b_times, thetas_d_pt, n_mixt, k_type);

    // Copy the result back to the memory of the numpy array
    to_row_major(res_d_pt, res, 2*n_s_times, 2*n_s_times);

    // Free local pointers of pointers
    free_mem_matrix(res_d_pt, 2*n_s_times);
    free_mem_matrix(thetas_d_pt, n_b_times-1);
}
