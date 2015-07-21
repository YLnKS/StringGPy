#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL cStringGPy_ARRAY_API

#include "sampl_utilities.h"

/* 
	Sample a single path of a univariate string GP.
		*result: nx3 PyArrayObject that will be modified in place. The first column contains the times, the second column the sGP and the third its derivative.
		*kernel_types: list of kernel types with K elements. E.g: [sm, se, sma, sma12] etc...
		*kernel_hypers: list of K PyArrayObject containing unconditional string kernel hyper-parameters.
		*b_times: list of the K+1 boundary times.
		*s_times: list of K PyArrayObject containing string times.
		*Ls: list of K n_kxn_k PyArrayObject containing the L factor in the Cholesky or SVD decomposition of the covariance matrix of the 
			values of the DSGP at the string times conditional on the values at the boundaries.
			
	The boundary conditions are sampled sequentially, conditional on which the values within strings are sampled in parallel (process based). 
*/
PyObject *sample_sgp(PyObject *kernel_types, PyObject *kernel_hypers, PyObject *b_times, PyObject *s_times, PyObject *Ls){
	Py_ssize_t K;
	K = PyList_Size(kernel_types);

	if((PyList_Size(kernel_hypers) != K) || (PyList_Size(b_times) != (K+1)) || (PyList_Size(s_times) != K) || (PyList_Size(Ls) != K)){
		PyErr_SetString(PyExc_ValueError, "List lengths should be consistent.");
		return NULL;
	}

	if(K==0){
		PyErr_SetString(PyExc_ValueError, "There should be at least one string.");
		return NULL;
	}

	char *kernel_type=NULL;
	double *theta=NULL;
	double b_time, prev_b_time, prev_zt, prev_zt_prime;
	double **c_cov_prev=NULL, **c_cov_new=NULL, **c_cov_new_prev=NULL, **c_n=NULL, **c_tmp=NULL, **mean_cond=NULL, **b_cond=NULL, **cov_cond=NULL, **c_M=NULL, **c_result=NULL;
	int n_mixt=0, k=0, tot_count=0, n_k=0, dims[2]={0};
	PyObject *p_result=NULL, *p_theta_k=NULL;
	tot_count=1+K;
	
	// Initialise the random seed.
	srand((unsigned)random_seed());
	
	// Determine the total number of times at which to evaluate the SDGP
	for(k=0; k<K; k++){
		n_k=PyList_Size(PyList_GetItem(s_times, k));
		if(PyList_Check(PyList_GetItem(s_times, k)) & (n_k>0)){
			tot_count+=n_k;
		}
	}

	// Empty array
	if(K==0){
		dims[0]=0;
		dims[1]=0;
		p_result=PyArray_FromDims(2, dims, NPY_DOUBLE);
		return p_result;
	}

	// Prepare the list of results
	dims[0]=tot_count;
	dims[1]=3;
	p_result=PyArray_FromDims(2, dims, NPY_DOUBLE);
	PyArray_AsCArray_Safe(&p_result, &c_result, PyArray_DIMS((PyArrayObject *)p_result), 2, PyArray_DescrFromType(NPY_DOUBLE));

	// Step 1: Generate boundary conditions sequentially
	for(k=0; k<K+1; k++){
		// Retrieve kernel type, hyper-parameters, and boundary times for the k-th string
		b_time = PyFloat_AsDouble(PyList_GetItem(b_times, k));

		if(k<K){
			// The last 2 boundary times share string parameters
			kernel_type=PyString_AsString(PyList_GetItem(kernel_types, (Py_ssize_t)k));
			p_theta_k=PyList_GetItem(kernel_hypers, k);
			PyArray_AsCArray(&p_theta_k, &theta, PyArray_DIMS((PyArrayObject *)p_theta_k), 1, PyArray_DescrFromType(NPY_DOUBLE));
			n_mixt=n_spectral_comp((PyArrayObject *)PyList_GetItem(kernel_hypers, k), kernel_type);
		}
		

		// Covariance matrix of the DSGP at b_time
		c_cov_new=new_zeros(2, 2);
		c_cov_new[0][0]=kernel(b_time, b_time, theta, n_mixt, kernel_type);
		c_cov_new[0][1]=dkernel_dy(b_time, b_time, theta, n_mixt, kernel_type);
		c_cov_new[1][0]=dkernel_dx(b_time, b_time, theta, n_mixt, kernel_type);
		c_cov_new[1][1]=dkernel_dxdy(b_time, b_time, theta, n_mixt, kernel_type);

		if(k==0){
			// The centred Gaussian
			c_n=new_zeros(2, 1);
			multivariate_normal(c_n, c_cov_new, 2, NULL);
		}else{
			// k > 0: sample conditional on the previous sample
			c_tmp=new_zeros(2, 2);
			invert2(c_tmp, c_cov_prev); //tmp = cov_prev^{-1}

			c_cov_new_prev=new_zeros(2, 2);
			c_cov_new_prev[0][0]=kernel(b_time, prev_b_time, theta, n_mixt, kernel_type);
			c_cov_new_prev[0][1]=dkernel_dy(b_time, prev_b_time, theta, n_mixt, kernel_type);
			c_cov_new_prev[1][0]=dkernel_dx(b_time, prev_b_time, theta, n_mixt, kernel_type);
			c_cov_new_prev[1][1]=dkernel_dxdy(b_time, prev_b_time, theta, n_mixt, kernel_type);

			c_M=new_zeros(2, 2);
			matrix_prod(c_M, c_cov_new_prev, 2, 2, c_tmp, 2); //tmp2=np.dot(cov_new_prev, cov_prev^{-1})
			transpose(c_cov_new_prev, 2); //cov_new_prev = cov_new_prev.T in place
			matrix_prod(c_tmp, c_M, 2, 2, c_cov_new_prev, 2); //tmp=np.dot(np.dot(cov_new_prev, cov_prev^{-1}), cov_new_prev.T)
			cov_cond=new_zeros(2, 2);
			matrix_sub(cov_cond, c_cov_new, c_tmp, 2, 2); //cov_cond = cov_new_new - np.dot(np.dot(cov_new_prev, cov_prev^{-1}), cov_new_prev.T)

			// The centred Gaussian
			c_n=new_zeros(2, 1);
			multivariate_normal(c_n, cov_cond, 2, NULL);

			// Conditional mean
			mean_cond=new_zeros(2, 1);
			b_cond=new_zeros(2, 1); // Previous boundary condition
			b_cond[0][0]=prev_zt;
			b_cond[1][0]=prev_zt_prime;
			matrix_prod(mean_cond, c_M, 2, 2, b_cond, 1);

			// Add the conditional mean to the centred draw
			matrix_add(c_n, c_n, mean_cond, 2, 1);
		}

		if(k!=K-1)
			PyArray_Free(p_theta_k, theta);

		c_result[k][0]=b_time;
		c_result[k][1]=c_n[0][0];
		c_result[k][2]=c_n[1][0];

		if(k>0)
			// Cleanup c_cov_prev
			free_mem_matrix(c_cov_prev, 2);
			
		c_cov_prev=c_cov_new;
		prev_b_time=b_time;
		prev_zt=c_n[0][0];
		prev_zt_prime=c_n[1][0];

		// Cleanup the rest
		free_mem_matrix(c_n, 2);
		
		if(k>0){
			free_mem_matrix(c_cov_new_prev, 2);
			free_mem_matrix(c_tmp, 2);
			free_mem_matrix(mean_cond, 2);
			free_mem_matrix(cov_cond, 2);
			free_mem_matrix(b_cond, 2);
			free_mem_matrix(c_M, 2);
		}
	}
	free_mem_matrix(c_cov_new, 2);

	// Step 2: Generate string values in parallel
	double **p_string_results=NULL;
	p_string_results=malloc(K*sizeof(double *));

	for(k=0; k<K; k++)
		p_string_results[k]=NULL;	

	double **double_tmp=NULL, *tmp=NULL;
	int lvl=0, p=0; // Level in the process family tree (from 0, top parents)
	pid_t *cPids =NULL;
	cPids = malloc(K*sizeof(pid_t));

	int fds[K][2];
	PyObject *L_k=NULL;
	double **c_L_k=NULL, **cond_cov_k=NULL, **cond_mean_k=NULL, **bound_cond_k=NULL, **M_k=NULL, *b_t_k=NULL, *s_t_k=NULL;
	double **cov_new_new=NULL, **cov_new_old=NULL, **cov_old_old=NULL, **cov_old_new=NULL;
	int i=0, status=0;
	
	for(k=0; k<K; k++){
		if((!PyList_Check(PyList_GetItem(s_times, k))) || (PyList_Size(PyList_GetItem(s_times, k))==0))
		{
			cPids[k]=0; // No need to spin-off a child process.
			continue; // There is no inner string time on the string.
		}

		if(lvl==0){
			n_k=PyList_Size(PyList_GetItem(s_times, k));
			L_k=PyList_GetItem(Ls, k);
			kernel_type=PyString_AsString(PyList_GetItem(kernel_types, (Py_ssize_t)k));
			n_mixt=n_spectral_comp((PyArrayObject *)PyList_GetItem(kernel_hypers, k), kernel_type);
			p_theta_k=PyList_GetItem(kernel_hypers, k);
			PyArray_AsCArray(&p_theta_k, &theta, PyArray_DIMS((PyArrayObject *)p_theta_k), 1, PyArray_DescrFromType(NPY_DOUBLE));

			b_t_k=malloc(2*sizeof(double));
			b_t_k[0]=PyFloat_AsDouble(PyList_GetItem(b_times, k));
			b_t_k[1]=PyFloat_AsDouble(PyList_GetItem(b_times, k+1));

			s_t_k=malloc(n_k*sizeof(double));
			for(i=0; i<n_k; i++)
				s_t_k[i]=PyFloat_AsDouble(PyList_GetItem(PyList_GetItem(s_times, k), i));

			cov_new_new=new_zeros(2*n_k, 2*n_k);
			cov_new_old=new_zeros(2*n_k, 4);
			cov_old_new=new_zeros(4, 2*n_k);
			cov_old_old=new_zeros(4, 4);
			M_k=new_zeros(2*n_k, 4);
			deriv_cov(cov_new_new, s_t_k, n_k, s_t_k, n_k, theta, n_mixt, kernel_type);
			deriv_cov(cov_new_old, s_t_k, n_k, b_t_k, 2, theta, n_mixt, kernel_type);
			deriv_cov(cov_old_old, b_t_k, 2, b_t_k, 2, theta, n_mixt, kernel_type);
			PyArray_Free(p_theta_k, theta);
			
			invert_robust(cov_old_old, 4); // It is critical to use an adjusted pseudo-inverse as the conditional cov might be ill-conditioned.
			matrix_prod(M_k, cov_new_old, 2*n_k, 4, cov_old_old, 4);

			status=pipe(fds[k]);

			if(status==-1)
				PyErr_SetString(PyExc_ValueError, "Pipe failed."); // Pipe failed

			p=fork();
			if(p == 0){
				/* This code will only be executed by children */
				
				// It is important to re-initialise the random seed to make the behaviour random across children (the seed depends on the time in nanoseconds and the pid).
				srand((unsigned)random_seed());
				
				tmp=malloc(2*n_k*sizeof(double));
				lvl = 1;
				close(fds[k][0]); // The child does not need to receive data from the parent
				// Sample the derivative string GP at string times conditional on the boundary conditions
				if((!PyArray_Check(L_k)) | (PyArray_DIMS((PyArrayObject *)L_k)[0] != n_k) | (PyArray_DIMS((PyArrayObject *)L_k)[1] != n_k)){
					// The L factor wasn't provided by the user or was wrong: sample from the covariance matrix.
					// 1. Compute the conditional covariance matrix of the SDGP at boundary times
					cond_cov_k=new_zeros(2*n_k, 2*n_k);
					transpose_copy(cov_old_new, cov_new_old, 2*n_k, 4);
					matrix_prod(cond_cov_k, M_k, 2*n_k, 4, cov_old_new, 2*n_k);
					matrix_sub(cond_cov_k, cov_new_new, cond_cov_k, 2*n_k, 2*n_k); 

					// 2. Sample a centred Gaussian vector with the above covariance matrix
					double_tmp=new_zeros(2*n_k, 1);
					multivariate_normal(double_tmp, cond_cov_k, 2*n_k, NULL);
				}else{
					// The user provided the L factor
					// 1. Sample a centred Gaussian vector using the provided L factor
					PyArray_AsCArray(&L_k, &c_L_k, PyArray_DIMS((PyArrayObject *)L_k), 2, PyArray_DescrFromType(NPY_DOUBLE));
					double_tmp=new_zeros(2*n_k, 1);
					multivariate_normal(double_tmp, NULL, 2*n_k, c_L_k);
					PyArray_Free(L_k, c_L_k);
				}
				// 3. Compute the conditional mean and add it to the previously generated sample
				// 3.1 Retrieve the previously computed boundary conditions.
				bound_cond_k=new_zeros(4, 1);
				bound_cond_k[0][0]=c_result[k][1];
				bound_cond_k[1][0]=c_result[k][2];
				bound_cond_k[2][0]=c_result[k+1][1];
				bound_cond_k[3][0]=c_result[k+1][2];

				// 3.2 Compute the conditional mean
				cond_mean_k=new_zeros(2*n_k, 1);
				matrix_prod(cond_mean_k, M_k, 2*n_k, 4, bound_cond_k, 1);
				free_mem_matrix(bound_cond_k, 4);

				// 3.3 Add the conditional mean to the centred sample
				matrix_add(double_tmp, double_tmp, cond_mean_k, 2*n_k, 1);

				// 4. Transfer the data to the parent process
				for(i=0; i<n_k; i++){
					tmp[2*i]=double_tmp[2*i][0];
					tmp[1+2*i]=double_tmp[1+2*i][0];
				}

				status=write(fds[k][1], tmp, 2*n_k*sizeof(double));

				if(status<0){
					PyErr_SetString(PyExc_ValueError, "An error occured while sending data to the parent.");
					printf("Oh dear, something went wrong with write() in %s! %s File descriptor %d\n", __func__, strerror(errno), fds[k][1]);
				}

				free(tmp);
				free(b_t_k);
				free(s_t_k);
				free(cPids);
				free_mem_matrix(p_string_results, K);
				free_mem_matrix(cov_new_new, 2*n_k);
				free_mem_matrix(cov_new_old, 2*n_k);
				free_mem_matrix(cov_old_new, 4);
				free_mem_matrix(cov_old_old, 4);
				free_mem_matrix(cond_cov_k, 2*n_k);
				free_mem_matrix(double_tmp, 2*n_k);
				free_mem_matrix(M_k, 2*n_k);
				free_mem_matrix(cond_mean_k, 2*n_k);
				exit(0);
				
			}else{
				/* This code will only be executed by the parent */
				cPids[k] = p; // Record the PID of the child to properly wait for it later.
				close(fds[k][1]); // The parent does not need to send data to its children

				free(b_t_k);
				free(s_t_k);
				free_mem_matrix(cov_new_new, 2*n_k);
				free_mem_matrix(cov_new_old, 2*n_k);
				free_mem_matrix(cov_old_new, 4);
				free_mem_matrix(cov_old_old, 4);
				free_mem_matrix(M_k, 2*n_k);
			}
		}
	}

	
	if(lvl == 0){
		/* Listen to the children on the pipes */
		for(k=0; k<K; k++){
			if((!PyList_Check(PyList_GetItem(s_times, k))) || (PyList_Size(PyList_GetItem(s_times, k))==0))
				continue; // Nothing to read.

			n_k=PyList_Size(PyList_GetItem(s_times, k));
			p_string_results[k]=malloc(2*n_k*sizeof(double));
			status=read(fds[k][0], p_string_results[k], 2*n_k*sizeof(double));
			close(fds[k][0]);
			
			if(status<0){
				PyErr_SetString(PyExc_ValueError, "An error occured while receiving data from a child.");
				printf("Oh dear, something went wrong with read() in %s! %s File descriptor %d\n", __func__, strerror(errno), fds[k][0]);
			}
		}

		/* Wait for children to exit */
		int waitc;
		do {
		   waitc=0;
			for (k=0; k<K; k++) {
			   if (cPids[k]>0) {
				  if (waitpid(cPids[k], NULL, 0) != 0) {// WNOHANG -> 0: Should hang to avoid too many calls.
					 /* Child is done */
					 cPids[k]=0;
				  }
				  else {
					 /* Still waiting on this child */
					 waitc=1;
					}
				}
			   /* Give up timeslice and prevent hard loop */
			   sleep(0);
			}
		} while (waitc);

		// All children are done. Append the results to p_result
		int i=0, n_k=0, idx=0;
		double s_t_k_i=0.0;
		PyObject *_s=NULL;
		idx=1+K;
		for(k=0; k<K; k++){
			n_k=PyList_Size(PyList_GetItem(s_times, k));
			if(PyList_Check(PyList_GetItem(s_times, k)) & (n_k>0))
			{
				_s = PyList_GetItem(s_times, k);
				for(i=0;i<n_k;i++){
					s_t_k_i=PyFloat_AsDouble(PyList_GetItem(_s, i));
					c_result[idx][0]=s_t_k_i;
					c_result[idx][1]=p_string_results[k][2*i];
					c_result[idx][2]=p_string_results[k][1+2*i];

					idx+=1;
				}
			}
			free(p_string_results[k]);
		}

		/* Cleanup */
		free(cPids);
		free(p_string_results);
	}
	return p_result;
}


/* 
	Sample independent paths of a univariate string GPs in parallel.
		This is essentially equivalent to calling sample_sgp in parallel.
*/
PyObject *sample_sgps(PyObject *l_kernel_types, PyObject *l_kernel_hypers, PyObject *l_b_times, PyObject *l_s_times, PyObject *l_Ls){
	Py_ssize_t D;
	D=PyList_Size(l_kernel_types);

	if((PyList_Size(l_kernel_hypers) != D) || (PyList_Size(l_b_times) != D) || (PyList_Size(l_s_times) != D) || (PyList_Size(l_Ls) != D))
		PyErr_SetString(PyExc_ValueError, "List lengths should be consistent.");

	int fds[D][2];
	pid_t *cPids =NULL;
	int lvl=0, d=0, p=0, status=0, n=0, k=0, K=0, i=0, n_k=0, dims[2]={0};
	PyObject *kernel_types=NULL, *kernel_hypers=NULL, *b_times=NULL, *s_times=NULL, *Ls=NULL, *sample=NULL, *p_result=NULL;
	double *tmp=NULL, **c_sample=NULL;

	cPids=malloc(D*sizeof(pid_t));

	for(d=0; d<D; d++){
		if((!PyList_Check(PyList_GetItem(l_s_times, d))) || (PyList_Size(PyList_GetItem(l_s_times, d))==0))
		{
			cPids[d]=0; // No need to spin-off a child process.
			continue; // There is no inner string time on the string.
		}

		if(lvl==0){
			status=pipe(fds[d]);

			if(status==-1){
				PyErr_SetString(PyExc_ValueError, "Pipe failed."); // Pipe failed
			}

			p=fork();
			if(p == 0){
				/* This code will only be executed by children */
				lvl = 1;
				close(fds[d][0]); // The child does not need to receive data from the parent
				kernel_types=PyList_GetItem(l_kernel_types, d);
				kernel_hypers=PyList_GetItem(l_kernel_hypers, d);
				b_times=PyList_GetItem(l_b_times, d);
				s_times=PyList_GetItem(l_s_times, d);
				Ls=PyList_GetItem(l_Ls, d);

				// Sample the dsgp
				sample=sample_sgp(kernel_types, kernel_hypers, b_times, s_times, Ls);
				PyArray_AsCArray_Safe(&sample, &c_sample, PyArray_DIMS((PyArrayObject *)sample), 2, PyArray_DescrFromType(NPY_DOUBLE));

				// Unpack the sample as row major and write to the pipe
				n=(int)PyArray_DIMS((PyArrayObject *)sample)[0];

				tmp=malloc(3*n*sizeof(double));
				for(i=0; i<n; i++){
					tmp[3*i]=c_sample[i][0];
					tmp[3*i+1]=c_sample[i][1];
					tmp[3*i+2]=c_sample[i][2];
				}

				status=write(fds[d][1], tmp, 3*n*sizeof(double));
				if(status<0)
					PyErr_SetString(PyExc_ValueError, "An error occured while sending data to the parent.");


				// Cleaning up
				PyArray_Free(sample, c_sample);
				free(tmp);
				free(cPids);

				exit(0);
			}
			else{
				/* This code will only be executed by the parent */
				cPids[d] = p; // Record the PID of the child to properly wait for it later.
				close(fds[d][1]); // The parent does not need to send data to its children.
			}
		}
	}

	if(lvl == 0){
		p_result=PyList_New(D);

		/* Listen to children on the pipes */
		for(d=0; d<D; d++){
			if((!PyList_Check(PyList_GetItem(l_s_times, d))) || (PyList_Size(PyList_GetItem(l_s_times, d))==0))
				continue; // There is no inner string time on the string.

			s_times=PyList_GetItem(l_s_times, d);
			kernel_types=PyList_GetItem(l_kernel_types, d);
			K=PyList_Size(kernel_types);

			// Determine the total number of times at which the SDGP was evaluated
			n=K+1;
			for(k=0; k<K; k++){
				n_k=PyList_Size(PyList_GetItem(s_times, k));
				if(PyList_Check(PyList_GetItem(s_times, k)) & (n_k>0)){
					n+=n_k;
				}
			}

			tmp=malloc(3*n*sizeof(double));
			status=read(fds[d][0], tmp, 3*n*sizeof(double));
			close(fds[d][0]);

			if(status<0){
				sample=PyList_New(0);
				PyErr_SetString(PyExc_ValueError, "An error occured while receiving data from a child.");
			}
			else{
				dims[0]=n;
				dims[1]=3;
				sample=PyArray_FromDims(2, dims, NPY_DOUBLE);
				PyArray_AsCArray_Safe(&sample, &c_sample, PyArray_DIMS((PyArrayObject *)sample), 2, PyArray_DescrFromType(NPY_DOUBLE));
				for(i=0; i<n; i++){
					c_sample[i][0]=tmp[3*i];
					c_sample[i][1]=tmp[3*i+1];
					c_sample[i][2]=tmp[3*i+2];			
				}
			}
			PyList_SetItem(p_result, d, sample);

			// Cleanup
			free(tmp);
		}

		/* Wait for children to exit */
		int waitc, *statuses=NULL;
		statuses=malloc(D*sizeof(int));
		for(d=0; d<D; d++)
			statuses[d]=0;

		do {
		   waitc=0;
			for (d=0; d<D; d++) {
			   if (cPids[d]>0) {
				  	if (waitpid(cPids[d], &statuses[d], 0) != 0) {// WNOHANG -> 0: Should hang to avoid too many calls.
						/* Child is done */
						cPids[d]=0;
						if(WIFSTOPPED(statuses[d]))
							printf("Child %d stop by signal %d\n", d, WSTOPSIG(statuses[d]));
				  	}
				  	else {
						/* Still waiting on this child */
						waitc=1;
					}
				}
			   /* Give up timeslice and prevent hard loop */
			   sleep(0);
			}
		} while (waitc);
		free(statuses);
	}

	// Cleanup
	free(cPids);

	return p_result;
}



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
PyObject *cond_eigen_anal(PyObject *kernel_types, PyObject *kernel_hypers, PyObject *b_times, PyObject *s_times){
	Py_ssize_t K;
	K=PyList_Size(kernel_types);

	if((PyList_Size(kernel_hypers) != K) || (PyList_Size(b_times) != (K+1)) || (PyList_Size(s_times) != K))
		PyErr_SetString(PyExc_ValueError, "List lengths should be consistent.");

	double **c_results=NULL, *tmp=NULL, *theta=NULL, *b_t_k=NULL, *s_t_k=NULL;
	double **cov_new_new=NULL, **cov_new_old=NULL, **cov_old_old=NULL, **cov_old_new=NULL, **cond_cov=NULL, **M=NULL, **L=NULL, **L_I=NULL;
	double **U=NULL, *S=NULL, **V=NULL, eps=0.0, det=0.0;

	c_results=malloc(K*sizeof(double *));
	int fds[K][2];
	int n_k=0, status=0, k=0, n_mixt=0, i=0, j=0, u=0, lvl=0, p=0;
	char *kernel_type;
	
	for(k=0; k<K; k++)
		c_results[k]=NULL;	

	pid_t *cPids =NULL;
	cPids=malloc(K*sizeof(pid_t));

	PyObject *p_theta_k=NULL, *p_result=NULL;
	
	/*
		Step 1: Kick off the jobs in parallel and wait for completion.
	*/
	for(k=0; k<K; k++){
		if((!PyList_Check(PyList_GetItem(s_times, k))) || (PyList_Size(PyList_GetItem(s_times, k))==0))
		{
			cPids[k]=0; // No need to spin-off a child process.
			continue; // There is no inner string time on the string.
		}
		
		if(lvl==0){
			n_k=PyList_Size(PyList_GetItem(s_times, k));
			
			status=pipe(fds[k]);
			if(status==-1)
				PyErr_SetString(PyExc_ValueError, "Pipe failed."); // Pipe failed	
			
			p=fork();
			if(p == 0){
				// Allocate memory for data to be transferred to the parent process.
				//	We will be transferring two n_k2 x n_k2 matrices, one n_k2 x 4 and a scalar. 
				tmp=malloc((1 + 2*4*n_k*n_k + 2*n_k*4)*sizeof(double));
				lvl = 1;
				/* This code will only be executed by children */
				close(fds[k][0]); // The child does not need to receive data from the parent

				/*
					Perform the eigen analysis here.
				*/
				kernel_type=PyString_AsString(PyList_GetItem(kernel_types, (Py_ssize_t)k));
				n_mixt=n_spectral_comp((PyArrayObject *)PyList_GetItem(kernel_hypers, k), kernel_type);
				p_theta_k=PyList_GetItem(kernel_hypers, k);
				PyArray_AsCArray(&p_theta_k, &theta, PyArray_DIMS((PyArrayObject *)p_theta_k), 1, PyArray_DescrFromType(NPY_DOUBLE));

				b_t_k=malloc(2*sizeof(double));
				b_t_k[0]=PyFloat_AsDouble(PyList_GetItem(b_times, k));
				b_t_k[1]=PyFloat_AsDouble(PyList_GetItem(b_times, k+1));

				s_t_k=malloc(n_k*sizeof(double));
				for(i=0; i<n_k; i++)
					s_t_k[i]=PyFloat_AsDouble(PyList_GetItem(PyList_GetItem(s_times, k), i));

				cov_new_new=new_zeros(2*n_k, 2*n_k);
				cov_new_old=new_zeros(2*n_k, 4);
				cov_old_new=new_zeros(4, 2*n_k);
				cov_old_old=new_zeros(4, 4);
				M=new_zeros(2*n_k, 4);

				deriv_cov(cov_new_new, s_t_k, n_k, s_t_k, n_k, theta, n_mixt, kernel_type);
				deriv_cov(cov_new_old, s_t_k, n_k, b_t_k, 2, theta, n_mixt, kernel_type);
				deriv_cov(cov_old_old, b_t_k, 2, b_t_k, 2, theta, n_mixt, kernel_type);
				PyArray_Free(p_theta_k, theta);
			
				invert_robust(cov_old_old, 4); // It is critical to use an adjusted pseudo-inverse as the conditional cov might be ill-conditioned.
				matrix_prod(M, cov_new_old, 2*n_k, 4, cov_old_old, 4);

				cond_cov=new_zeros(2*n_k, 2*n_k);
				transpose_copy(cov_old_new, cov_new_old, 2*n_k, 4);
				matrix_prod(cond_cov, M, 2*n_k, 4, cov_old_new, 2*n_k);
				matrix_sub(cond_cov, cov_new_new, cond_cov, 2*n_k, 2*n_k); 

				U=new_zeros(2*n_k, 2*n_k);
				V=new_zeros(2*n_k, 2*n_k);
				S=malloc(2*n_k*sizeof(double));

				for(i=0; i<2*n_k; i++)
					S[i]=0.0;

				svd(cond_cov, U, S, V, 2*n_k); // Compute the SVD of cov in-place

				L=new_zeros(2*n_k, 2*n_k);
				L_I=new_zeros(2*n_k, 2*n_k);

				eps=ill_cond_jit(S, 2*n_k); // Improve conditioning if necessary.
				for(i=0; i<2*n_k; i++){
					for(j=0; j<2*n_k; j++){
						L[i][j]=U[i][j]*sqrt(eps+S[j]);
						L_I[j][i]=U[i][j]/sqrt(eps+S[j]);
					}
				}

				// Compute the determinant 
				det=0.0;
				for(i=0; i<2*n_k; i++){
					det+=log(fabs(eps+S[i]));
				}
			
				// Unpack the results gathered from children and put them in a PyListObject.
				for(u=0; u < 1 + 2*4*n_k*n_k + 2*n_k*4; u++){
					// 1. The first n_k2 x n_k2 terms correspond to L (row major).
					if(u<4*n_k*n_k){
						i=u/(2*n_k);
						j=u-(2*n_k*i);
						tmp[u]=L[i][j];
					}
					// 2. The next n_k2 x n_k2 terms correspond to L^{-1} (row major).
					else if(u<2*4*n_k*n_k){
						i=(u-4*n_k*n_k)/(2*n_k);
						j=(u-4*n_k*n_k)-(2*n_k*i);
						tmp[u]=L_I[i][j];
					}
					// 3. The next n_k2 x 4 terms correspond to M (row major).
					else if(u < 2*4*n_k*n_k + 2*n_k*4){
						i=(u-2*4*n_k*n_k)/4;
						j=(u-2*4*n_k*n_k)-(4*i);
						tmp[u]=M[i][j];
					}
					// 4. The last term corresponds to the determinant.
					else{
						tmp[u]=det; // log(det)
					}
				}

				status=write(fds[k][1], tmp, (1 + 2*4*n_k*n_k + 2*n_k*4)*sizeof(double));

				// Cleaning up
				free(tmp);
				free(b_t_k);
				free(s_t_k);
				free(cPids);
				free(S);
				free_mem_matrix(c_results, K);
				free_mem_matrix(U, 2*n_k);
				free_mem_matrix(V, 2*n_k);
				free_mem_matrix(M, 2*n_k);
				free_mem_matrix(L, 2*n_k);
				free_mem_matrix(L_I, 2*n_k);
				free_mem_matrix(cov_new_new, 2*n_k);
				free_mem_matrix(cov_old_new, 4);
				free_mem_matrix(cov_new_old, 2*n_k);
				free_mem_matrix(cov_old_old, 4);
				free_mem_matrix(cond_cov, 2*n_k);
				
				if(status<0){
					PyErr_SetString(PyExc_ValueError, "An error occured while sending data to the parent.");
					printf("Oh dear, something went wrong with write() in %s! %s\n", __func__, strerror(errno));
				}
				
				exit(0);
			}else{
				/* This code will only be executed by the parent */
				cPids[k]=p; // Record the PID of the child to properly wait for it later.
				close(fds[k][1]); // The parent does not need to send data to its children
			}	
		}
	}
	
	/*
		Step 2: Collect the results and package them in a PyListObject.
	*/
	if(lvl == 0){
		/* Listen to the children on the pipes */
		for(k=0; k<K; k++){
			if((!PyList_Check(PyList_GetItem(s_times, k))) || (PyList_Size(PyList_GetItem(s_times, k))==0))
				continue; // Nothing to read.

			n_k=PyList_Size(PyList_GetItem(s_times, k));
			c_results[k]=malloc((1 + 2*4*n_k*n_k + 2*n_k*4)*sizeof(double));
			status=read(fds[k][0], c_results[k], (1 + 2*4*n_k*n_k + 2*n_k*4)*sizeof(double));
			close(fds[k][0]);

			if(status<0){
				PyErr_SetString(PyExc_ValueError, "An error occured while receiving data from a child.");
				printf("Oh dear, something went wrong with read() in %s! %s\n", __func__, strerror(errno));
			}
		}

		/* Wait for children to exit */
		int waitc;
		do {
		   waitc=0;
			for (k=0; k<K; k++) {
			   if (cPids[k]>0) {
				  if (waitpid(cPids[k], NULL, 0) != 0) {// WNOHANG -> 0: Should hang to avoid too many calls.
					 /* Child is done */
					 cPids[k]=0;
				  }
				  else {
					 /* Still waiting on this child */
					 waitc=1;
					}
				}
			   /* Give up timeslice and prevent hard loop */
			   sleep(0);
			}
		} while (waitc);

		// All children are done. Append the results to p_result
		int dims[2];
		PyObject *p_tuple=NULL, *p_L=NULL, *p_L_I=NULL, *p_M=NULL;
		double **c_L=NULL, **c_L_I=NULL, **c_M=NULL;
		
		// Prepare the list of results
		p_result=PyList_New(K);
		
		for(k=0; k<K; k++){
			n_k=PyList_Size(PyList_GetItem(s_times, k));
			if(PyList_Check(PyList_GetItem(s_times, k)) & (n_k>0))
			{
				dims[0]=2*n_k;
				dims[1]=2*n_k;
				
				p_L=PyArray_FromDims(2, dims, NPY_DOUBLE);
				PyArray_AsCArray_Safe((PyObject **)&p_L, &c_L, PyArray_DIMS((PyArrayObject *)p_L), 2, PyArray_DescrFromType(NPY_DOUBLE));
				
				p_L_I=PyArray_FromDims(2, dims, NPY_DOUBLE);
				PyArray_AsCArray_Safe((PyObject **)&p_L_I, &c_L_I, PyArray_DIMS((PyArrayObject *)p_L_I), 2, PyArray_DescrFromType(NPY_DOUBLE));
				
				dims[1]=4;
				p_M=PyArray_FromDims(2, dims, NPY_DOUBLE);
				PyArray_AsCArray_Safe((PyObject **)&p_M, &c_M, PyArray_DIMS((PyArrayObject *)p_M), 2, PyArray_DescrFromType(NPY_DOUBLE));
				
				// Unpack the results gathered from children and put them in a PyListObject.
				for(u=0; u < 1 + 2*4*n_k*n_k + 2*n_k*4; u++){
					// 1. The first n_k2 x n_k2 terms correspond to L (row major).
					if(u<4*n_k*n_k){
						i=u/(2*n_k);
						j=u-(2*n_k*i);
						c_L[i][j]=c_results[k][u];
					}
					// 2. The next n_k2 x n_k2 terms correspond to L^{-1} (row major).
					else if(u<2*4*n_k*n_k){
						i=(u-4*n_k*n_k)/(2*n_k);
						j=(u-4*n_k*n_k)-(2*n_k*i);
						c_L_I[i][j]=c_results[k][u];
					}
					// 3. The next n_k2 x 4 terms correspond to M (row major).
					else if(u < 2*4*n_k*n_k + 2*n_k*4){
						i=(u-2*4*n_k*n_k)/4;
						j=(u-2*4*n_k*n_k)-(4*i);
						c_M[i][j]=c_results[k][u];
					}
					// 4. The last term corresponds to the log-det.
					else{
						det=c_results[k][u];
					}
				}
				
				p_tuple=PyList_New(4);
				PyList_SetItem(p_tuple, 0, p_L);
				PyList_SetItem(p_tuple, 1, p_L_I);
				PyList_SetItem(p_tuple, 2, p_M);
				PyList_SetItem(p_tuple, 3, (PyObject *)PyFloat_FromDouble(det));
				PyList_SetItem(p_result, k, p_tuple);
			}else{
				// When the string has no string times, return an empty list
				p_tuple=PyList_New(0);
				PyList_SetItem(p_result, k, p_tuple);
			}
			free(c_results[k]);
		}

		/* Cleanup */
		free(cPids);
		free(c_results);
	}
	return p_result;
}


/*
	Perform multiple eigenvalue analyses in parallel.
		This is essentially equivalent to calling cong_eigen_anal in parallel.
*/
PyObject *cond_eigen_anals(PyObject *l_kernel_types, PyObject *l_kernel_hypers, PyObject *l_b_times, PyObject *l_s_times){
	Py_ssize_t D;
	D=PyList_Size(l_kernel_types);

	if((PyList_Size(l_kernel_hypers) != D) || (PyList_Size(l_b_times) != D) || (PyList_Size(l_s_times) != D))
		PyErr_SetString(PyExc_ValueError, "List lengths should be consistent.");

	int fds[D][2];
	pid_t *cPids =NULL;
	int lvl=0, d=0, p=0, status=0, k=0, K=0, i=0, j=0, u=0, n_k=0, dims[2]={0};
	PyObject *kernel_types=NULL, *kernel_hypers=NULL, *b_times=NULL, *s_times=NULL, *ana=NULL, *p_result=NULL;
	PyObject *p_L=NULL, *p_L_I=NULL, *p_M=NULL, *row=NULL;
	double *tmp=NULL, det=0.0, **c_L=NULL, **c_L_I=NULL, **c_M=NULL;
	int tot_count=0, idx=0;

	cPids=malloc(D*sizeof(pid_t));

	for(d=0; d<D; d++){
		if((!PyList_Check(PyList_GetItem(l_s_times, d))) || (PyList_Size(PyList_GetItem(l_kernel_hypers, d))==0))
		{
			cPids[d]=0; // No need to spin-off a child process.
			continue; // There is no inner string time on the string.
		}

		if(lvl==0){
			status=pipe(fds[d]);

			if(status==-1){
				PyErr_SetString(PyExc_ValueError, "Pipe failed."); // Pipe failed
			}

			p=fork();
			if(p == 0){
				/* This code will only be executed by children */
				lvl = 1;
				close(fds[d][0]); // The child does not need to receive data from the parent
				kernel_types=PyList_GetItem(l_kernel_types, d);
				kernel_hypers=PyList_GetItem(l_kernel_hypers, d);
				b_times=PyList_GetItem(l_b_times, d);
				s_times=PyList_GetItem(l_s_times, d);

				// Perform eigenvalue analysis
				ana=cond_eigen_anal(kernel_types, kernel_hypers, b_times, s_times);

				// Unpack the sample as row major
				K=PyList_Size(ana); // Number of strings in dimension d
				
				// Evaluate the memory required
				tot_count=0;
				for(k=0; k<K; k++){
					n_k=PyList_Size(PyList_GetItem(s_times, k));
					
					if(n_k>0)
						tot_count += 1 + 2*4*n_k*n_k + 2*n_k*4;
				}
				
				tmp=malloc(tot_count*sizeof(double));
				idx=0;
				for(k=0; k<K; k++){
					n_k=PyList_Size(PyList_GetItem(s_times, k));
					
					if(n_k==0)
						continue;
					
					row=PyList_GetItem(ana, k);
					
					p_L=PyList_GetItem(row, 0);
					PyArray_AsCArray((PyObject **)&p_L, &c_L, PyArray_DIMS((PyArrayObject *)p_L), 2, PyArray_DescrFromType(NPY_DOUBLE));
					
					p_L_I=PyList_GetItem(row, 1);
					PyArray_AsCArray((PyObject **)&p_L_I, &c_L_I, PyArray_DIMS((PyArrayObject *)p_L_I), 2, PyArray_DescrFromType(NPY_DOUBLE));
					
					p_M=PyList_GetItem(row, 2);
					PyArray_AsCArray((PyObject **)&p_M, &c_M, PyArray_DIMS((PyArrayObject *)p_M), 2, PyArray_DescrFromType(NPY_DOUBLE));
					
					det=PyFloat_AsDouble(PyList_GetItem(row, 3));
					
					for(u=0; u < 1 + 2*4*n_k*n_k + 2*n_k*4; u++){
						// 1. The first n_k2 x n_k2 terms correspond to L (row major).
						if(u<4*n_k*n_k){
							i=u/(2*n_k);
							j=u-(2*n_k*i);
							tmp[u+idx]=c_L[i][j];
						}
						// 2. The next n_k2 x n_k2 terms correspond to L^{-1} (row major).
						else if(u<2*4*n_k*n_k){
							i=(u-4*n_k*n_k)/(2*n_k);
							j=(u-4*n_k*n_k)-(2*n_k*i);
							tmp[u+idx]=c_L_I[i][j];
						}
						// 3. The next n_k2 x 4 terms correspond to M (row major).
						else if(u < 2*4*n_k*n_k + 2*n_k*4){
							i=(u-2*4*n_k*n_k)/4;
							j=(u-2*4*n_k*n_k)-(4*i);
							tmp[u+idx]=c_M[i][j];
						}
						// 4. The last term corresponds to the determinant.
						else{
							tmp[u+idx]=det; // log(det)
						}	
					}
					idx += 1 + 2*4*n_k*n_k + 2*n_k*4;
					
					// Cleanup
					PyArray_Free(p_L, c_L);
					PyArray_Free(p_L_I, c_L_I);
					PyArray_Free(p_M, c_M);
				}

				// Write to the pipe
				status=write(fds[d][1], tmp, tot_count*sizeof(double));

				// Cleaning up
				free(tmp);
				free(cPids);
				Py_DECREF(ana);
				
				if(status<0){
					PyErr_SetString(PyExc_ValueError, "An error occured while sending data to the parent.");
					printf("Oh dear, something went wrong with write() in %s! %s\n", __func__, strerror(errno));
				}
				
				exit(0);
			}
			else{
				/* This code will only be executed by the parent */
				cPids[d] = p; // Record the PID of the child to properly wait for it later.
				close(fds[d][1]); // The parent does not need to send data to its children.
			}
		}
	}

	if(lvl == 0){
		p_result=PyList_New(D);

		/* Listen to children on the pipes */
		for(d=0; d<D; d++){
			if((!PyList_Check(PyList_GetItem(l_s_times, d))) || (PyList_Size(PyList_GetItem(l_s_times, d))==0)){
				PyList_SetItem(p_result, d, PyList_New(0));
				continue; // Dimension d has no string time?
			}

			s_times=PyList_GetItem(l_s_times, d);
			kernel_types=PyList_GetItem(l_kernel_types, d);
			K=PyList_Size(kernel_types);

			// Determine the amount of data to read from the pipe.
			tot_count=0;
			for(k=0; k<K; k++){
				n_k=PyList_Size(PyList_GetItem(s_times, k));
				
				if(n_k>0)
					tot_count+=1 + 2*4*n_k*n_k + 2*n_k*4;
			}
			tmp=malloc(tot_count*sizeof(double));
			
			// Listen to the d-th child on the pipe.
			status=read(fds[d][0], tmp, tot_count*sizeof(double));
			close(fds[d][0]);

			// Re-pack the data received and update p_result
			ana=PyList_New(K);
			idx=0;

			for(k=0; k<K; k++){
				n_k=PyList_Size(PyList_GetItem(s_times, k));
				if(n_k>0){
					row=PyList_New(4);	
										
					dims[0]=2*n_k;
					dims[1]=2*n_k;
					p_L=PyArray_FromDims(2, dims, NPY_DOUBLE);
					PyArray_AsCArray_Safe((PyObject **)&p_L, &c_L, PyArray_DIMS((PyArrayObject *)p_L), 2, PyArray_DescrFromType(NPY_DOUBLE));
					
					p_L_I=PyArray_FromDims(2, dims, NPY_DOUBLE);
					PyArray_AsCArray_Safe((PyObject **)&p_L_I, &c_L_I, PyArray_DIMS((PyArrayObject *)p_L_I), 2, PyArray_DescrFromType(NPY_DOUBLE));
					
					dims[1]=4;
					p_M=PyArray_FromDims(2, dims, NPY_DOUBLE);
					PyArray_AsCArray_Safe((PyObject **)&p_M, &c_M, PyArray_DIMS((PyArrayObject *)p_M), 2, PyArray_DescrFromType(NPY_DOUBLE));
					
					for(u=0; u < 1 + 2*4*n_k*n_k + 2*n_k*4; u++){
						// 1. The first n_k2 x n_k2 terms correspond to L (row major).
						if(u<4*n_k*n_k){
							i=u/(2*n_k);
							j=u-(2*n_k*i);
							c_L[i][j]=tmp[u+idx];
						}
						// 2. The next n_k2 x n_k2 terms correspond to L^{-1} (row major).
						else if(u<2*4*n_k*n_k){
							i=(u-4*n_k*n_k)/(2*n_k);
							j=(u-4*n_k*n_k)-(2*n_k*i);
							c_L_I[i][j]=tmp[u+idx];
						}
						// 3. The next n_k2 x 4 terms correspond to M (row major).
						else if(u < 2*4*n_k*n_k + 2*n_k*4){
							i=(u-2*4*n_k*n_k)/4;
							j=(u-2*4*n_k*n_k)-(4*i);
							c_M[i][j]=tmp[u+idx];
						}
						// 4. The last term corresponds to the determinant.
						else{
							det=tmp[u+idx]; // log(det)
						}	
					}
					idx += 1 + 2*4*n_k*n_k + 2*n_k*4;

					PyList_SetItem(row, 0, p_L);
					PyList_SetItem(row, 1, p_L_I);
					PyList_SetItem(row, 2, p_M);
					PyList_SetItem(row, 3, (PyObject *)PyFloat_FromDouble(det));
					PyList_SetItem(ana, k, row);
				}else{
					// When the string has no string times, return an empty list
					row=PyList_New(0);
					PyList_SetItem(ana, k, row);
				}
			}

			// Append the result of the d-th dimension.
			PyList_SetItem(p_result, d, ana);

			// Cleanup
			free(tmp);
		}

		/* Wait for children to exit */
		int waitc, *statuses=NULL;
		statuses=malloc(D*sizeof(int));
		for(d=0; d<D; d++)
			statuses[d]=0;

		do {
		   waitc=0;
			for (d=0; d<D; d++) {
			   if (cPids[d]>0) {
				  	if (waitpid(cPids[d], &statuses[d], 0) != 0) {// WNOHANG -> 0: Should hang to avoid too many calls.
						/* Child is done */
						cPids[d]=0;
						if(WIFSIGNALED(statuses[d]))
							printf("Child %d was killed by signal %d", d, WTERMSIG(statuses[d]));
				  	}
				  	else {
						/* Still waiting on this child */
						waitc=1;
					}
				}
			   /* Give up timeslice and prevent hard loop */
			   sleep(0);
			}
		} while (waitc);
		free(statuses);
	}

	// Cleanup
	free(cPids);

	return p_result;
}

/*
	Truncate a double to 6 decimals digits and return it as a string.
		(Returns a new reference: should be PyDECREF in the calling function)
*/
PyObject *float_as_idx(PyObject *p_val){

	double val=PyFloat_AsDouble(p_val);

	PyObject *key=NULL;
	key = PyFloat_FromDouble(val); // DEBUG
	return key; // DEBUG


	// Get the number of character of the string output
	size_t needed = snprintf(NULL, 0, "%.6f", val);

	// Get the string output.
    char  *buff1 = malloc(needed+1);
    sprintf(buff1, "%.6f", val);

    // Remove NULL character at the end.
    char *buff2 = malloc(needed); 
    memcpy(buff2, buff1, needed);

    // Construct Python string. (New reference -- Should be freed!!)
    key = PyString_FromStringAndSize(buff2, (Py_ssize_t) needed);

    // Cleanup
    free(buff1);
    free(buff2);

    return key;
}

/*
	Sample the eigen factors L corresponding to the covariance matrices of DSGPs at a boundary time conditional on the previous.
*/
PyObject *compute_bound_ls(PyObject *l_kernel_types, PyObject *l_kernel_hypers, PyObject *l_b_times){
	Py_ssize_t D;
	D=PyList_Size(l_kernel_types);

	if(D <= 0)
		PyErr_SetString(PyExc_ValueError, "The dimension should be positive.");

	if((PyList_Size(l_b_times) != D) || (PyList_Size(l_kernel_hypers) != D))
		PyErr_SetString(PyExc_ValueError, "List lengths should be consistent.");

	pid_t *cPids =NULL;
	int lvl=0, d=0, p=0, k=0, K=0, i=0, j=0, dims[2]={0}, n_mixt=0;
	PyObject *kernel_types=NULL, *kernel_hypers=NULL, *b_times=NULL, *p_result=NULL, *p_theta_k=NULL;
	PyObject *p_L=NULL, *row=NULL, *p_M=NULL, *sub_row=NULL;
	double **c_L=NULL, **c_M=NULL, *prev_t=NULL, *t=NULL, **U=NULL, **V=NULL, *S=NULL, **cov=NULL, *theta=NULL;
	double **_tmp1=NULL, **_tmp2=NULL;
	double **cov_new_new=NULL, **cov_old_old=NULL, **cov_new_old=NULL, **cov_old_new=NULL;
	char *kernel_type=NULL;
	
	double **tmps=NULL; // Block of memory shared between parent and child.
	tmps = malloc(D*sizeof(double *));

	cPids=malloc(D*sizeof(pid_t));
	dims[0]=2;
	dims[1]=2;
	
	
	/* Special case for D=1. No need for parallelism: it actually creates a lot of computational overhead */
	if(D == 1){
		d=0;
		kernel_hypers=PyList_GetItem(l_kernel_hypers, d);
		K=PyList_Size(kernel_hypers); // Number of strings
		kernel_types=PyList_GetItem(l_kernel_types, d);				
		b_times=PyList_GetItem(l_b_times, d);
		
		U=new_zeros(2, 2);
		V=new_zeros(2, 2);
		S=malloc(2*sizeof(double));
		for(i=0; i<2; i++)
			S[i]=0.0;

		kernel_type=PyString_AsString(PyList_GetItem(kernel_types, (Py_ssize_t)0));
		n_mixt=n_spectral_comp((PyArrayObject *)PyList_GetItem(kernel_hypers, 0), kernel_type);
		p_theta_k=PyList_GetItem(kernel_hypers, 0);
		PyArray_AsCArray(&p_theta_k, &theta, PyArray_DIMS((PyArrayObject *)p_theta_k), 1, PyArray_DescrFromType(NPY_DOUBLE));

		t=malloc(sizeof(double));
		t[0]=PyFloat_AsDouble(PyList_GetItem(b_times, 0));
		cov=new_zeros(2, 2);
		deriv_cov(cov, t, 1, t, 1, theta, n_mixt, kernel_type);
		// svd(cov, U, S, V, 2); // Compute the SVD of cov in-place
		svd2(cov, U, S, V);// Compute the SVD of cov in-place

		p_L=PyArray_FromDims(2, dims, NPY_DOUBLE);
		PyArray_AsCArray_Safe((PyObject **)&p_L, &c_L, PyArray_DIMS((PyArrayObject *)p_L), 2, PyArray_DescrFromType(NPY_DOUBLE));

		p_M=PyArray_FromDims(2, dims, NPY_DOUBLE);
		PyArray_AsCArray_Safe((PyObject **)&p_M, &c_M, PyArray_DIMS((PyArrayObject *)p_M), 2, PyArray_DescrFromType(NPY_DOUBLE));

		for(i=0; i<2; i++){
			for(j=0; j<2; j++){
				c_L[i][j]=U[i][j]*sqrt(S[j]);
				c_M[i][j]=0.0;
			}
		}

		PyArray_Free(p_theta_k, theta);
		free(S);
		free_mem_matrix(U, 2);
		free_mem_matrix(V, 2);
		free_mem_matrix(cov, 2);
		free(t);
		
		// Initialise the row.
		row=PyList_New(K+1);
		
		sub_row=PyList_New(2);
		PyList_SetItem(sub_row, 0, p_L);
		PyList_SetItem(sub_row, 1, p_M);
		PyList_SetItem(row, 0, sub_row);	


		for(k=1; k<K+1; k++){
			t=malloc(sizeof(double));
			prev_t=malloc(sizeof(double));

			t[0]=PyFloat_AsDouble(PyList_GetItem(b_times, k));
			prev_t[0]=PyFloat_AsDouble(PyList_GetItem(b_times, k-1));

			cov_new_new=new_zeros(2, 2);
			cov_new_old=new_zeros(2, 2);
			cov_old_new=new_zeros(2, 2);
			cov_old_old=new_zeros(2, 2);
			_tmp1=new_zeros(2, 2);
			_tmp2=new_zeros(2, 2);

			if(k < K){
				kernel_type=PyString_AsString(PyList_GetItem(kernel_types, (Py_ssize_t)k));
				n_mixt=n_spectral_comp((PyArrayObject *)PyList_GetItem(kernel_hypers, k), kernel_type);
				p_theta_k=PyList_GetItem(kernel_hypers, k);
				PyArray_AsCArray(&p_theta_k, &theta, PyArray_DIMS((PyArrayObject *)p_theta_k), 1, PyArray_DescrFromType(NPY_DOUBLE));
			}

			deriv_cov(cov_new_new, t, 1, t, 1, theta, n_mixt, kernel_type);
			deriv_cov(cov_old_new, prev_t, 1, t, 1, theta, n_mixt, kernel_type);
			deriv_cov(cov_new_old, t, 1, prev_t, 1, theta, n_mixt, kernel_type);
			deriv_cov(cov_old_old, prev_t, 1, prev_t, 1, theta, n_mixt, kernel_type);

			invert_robust(cov_old_old, 2); // cov_old_old = cov_old_old^{-1}

			matrix_prod(_tmp1, cov_old_old, 2, 2, cov_old_new, 2);
			matrix_prod(_tmp2, cov_new_old, 2, 2, _tmp1, 2);
			matrix_sub(_tmp1, cov_new_new, _tmp2, 2, 2); // Conditional covariance matrix.

			U=new_zeros(2, 2);
			V=new_zeros(2, 2);
			S=malloc(2*sizeof(double));

			//svd(_tmp1, U, S, V, 2); // Compute the SVD of cov in-place
			svd2(_tmp1, U, S, V);// Compute the SVD of cov in-place

			p_L=PyArray_FromDims(2, dims, NPY_DOUBLE);
			PyArray_AsCArray_Safe((PyObject **)&p_L, &c_L, PyArray_DIMS((PyArrayObject *)p_L), 2, PyArray_DescrFromType(NPY_DOUBLE));

			for(i=0; i<2; i++){
				for(j=0; j<2; j++){
					c_L[i][j]=U[i][j]*sqrt(S[j]);
				}
			}

			p_M=PyArray_FromDims(2, dims, NPY_DOUBLE);
			PyArray_AsCArray_Safe((PyObject **)&p_M, &c_M, PyArray_DIMS((PyArrayObject *)p_M), 2, PyArray_DescrFromType(NPY_DOUBLE));
			matrix_prod(c_M, cov_new_old, 2, 2, cov_old_old, 2);

			if(k != K-1){
				PyArray_Free(p_theta_k, theta);
			}
			
			free(S);
			free_mem_matrix(U, 2);
			free_mem_matrix(V, 2);
			free_mem_matrix(cov_new_new, 2);
			free_mem_matrix(cov_new_old, 2);
			free_mem_matrix(cov_old_new, 2);
			free_mem_matrix(cov_old_old, 2);
			free_mem_matrix(_tmp1, 2);
			free_mem_matrix(_tmp2, 2);
			free(t);
			free(prev_t);
	
			sub_row=PyList_New(2);
			PyList_SetItem(sub_row, 0, p_L);
			PyList_SetItem(sub_row, 1, p_M);
			PyList_SetItem(row, k, sub_row);		
		}
			
		// Prepare the result
		p_result=PyList_New(D);
		PyList_SetItem(p_result, d, row);		
		
		// Cleanup
		free(cPids);
		return p_result;
	}
	
	/* Make use of process based parallelism for D > 1 */
	for(d=0; d<D; d++){
		if(lvl==0){			
			kernel_hypers=PyList_GetItem(l_kernel_hypers, d);
			K=PyList_Size(kernel_hypers); // Number of strings
			// Initialise the memory map
			tmps[d] = (double *)mmap(NULL, 8*(K+1)*sizeof(double), PROT_READ|PROT_WRITE, MAP_ANON|MAP_SHARED, -1, 0);
			if(tmps[d] == MAP_FAILED)
				PyErr_SetString(PyExc_ValueError, "mmap failed.");
			
			
			p=fork();
			if(p == 0){
				/* This code will only be executed by children */
				lvl = 1;
				kernel_types=PyList_GetItem(l_kernel_types, d);				
				b_times=PyList_GetItem(l_b_times, d);
				
				U=new_zeros(2, 2);
				V=new_zeros(2, 2);
				S=malloc(2*sizeof(double));
				for(i=0; i<2; i++)
					S[i]=0.0;

				kernel_type=PyString_AsString(PyList_GetItem(kernel_types, (Py_ssize_t)0));
				n_mixt=n_spectral_comp((PyArrayObject *)PyList_GetItem(kernel_hypers, 0), kernel_type);
				p_theta_k=PyList_GetItem(kernel_hypers, 0);
				PyArray_AsCArray(&p_theta_k, &theta, PyArray_DIMS((PyArrayObject *)p_theta_k), 1, PyArray_DescrFromType(NPY_DOUBLE));

				t=malloc(sizeof(double));
				t[0]=PyFloat_AsDouble(PyList_GetItem(b_times, 0));
				cov=new_zeros(2, 2);
				deriv_cov(cov, t, 1, t, 1, theta, n_mixt, kernel_type);
				// svd(cov, U, S, V, 2); // Compute the SVD of cov in-place
				svd2(cov, U, S, V);// Compute the SVD of cov in-place

				p_L=PyArray_FromDims(2, dims, NPY_DOUBLE);
				PyArray_AsCArray_Safe((PyObject **)&p_L, &c_L, PyArray_DIMS((PyArrayObject *)p_L), 2, PyArray_DescrFromType(NPY_DOUBLE));

				p_M=PyArray_FromDims(2, dims, NPY_DOUBLE);
				PyArray_AsCArray_Safe((PyObject **)&p_M, &c_M, PyArray_DIMS((PyArrayObject *)p_M), 2, PyArray_DescrFromType(NPY_DOUBLE));

				for(i=0; i<2; i++){
					for(j=0; j<2; j++){
						c_L[i][j]=U[i][j]*sqrt(S[j]);
						c_M[i][j]=0.0;
					}
				}

				tmps[d][0]=c_L[0][0];
				tmps[d][1]=c_L[0][1];
				tmps[d][2]=c_L[1][0];
				tmps[d][3]=c_L[1][1];
				tmps[d][4]=c_M[0][0];
				tmps[d][5]=c_M[0][1];
				tmps[d][6]=c_M[1][0];
				tmps[d][7]=c_M[1][1];

				PyArray_Free(p_theta_k, theta);
				free(S);
				free_mem_matrix(U, 2);
				free_mem_matrix(V, 2);
				free_mem_matrix(cov, 2);
				free(t);
				PyArray_Free(p_L, c_L);
				PyArray_Free(p_M, c_M);

				for(k=1; k<K+1; k++){
					t=malloc(sizeof(double));
					prev_t=malloc(sizeof(double));

					t[0]=PyFloat_AsDouble(PyList_GetItem(b_times, k));
					prev_t[0]=PyFloat_AsDouble(PyList_GetItem(b_times, k-1));

					cov_new_new=new_zeros(2, 2);
					cov_new_old=new_zeros(2, 2);
					cov_old_new=new_zeros(2, 2);
					cov_old_old=new_zeros(2, 2);
					_tmp1=new_zeros(2, 2);
					_tmp2=new_zeros(2, 2);

					if(k < K){
						kernel_type=PyString_AsString(PyList_GetItem(kernel_types, (Py_ssize_t)k));
						n_mixt=n_spectral_comp((PyArrayObject *)PyList_GetItem(kernel_hypers, k), kernel_type);
						p_theta_k=PyList_GetItem(kernel_hypers, k);
						PyArray_AsCArray(&p_theta_k, &theta, PyArray_DIMS((PyArrayObject *)p_theta_k), 1, PyArray_DescrFromType(NPY_DOUBLE));
					}

					deriv_cov(cov_new_new, t, 1, t, 1, theta, n_mixt, kernel_type);
					deriv_cov(cov_old_new, prev_t, 1, t, 1, theta, n_mixt, kernel_type);
					deriv_cov(cov_new_old, t, 1, prev_t, 1, theta, n_mixt, kernel_type);
					deriv_cov(cov_old_old, prev_t, 1, prev_t, 1, theta, n_mixt, kernel_type);

					invert_robust(cov_old_old, 2); // cov_old_old = cov_old_old^{-1}

					matrix_prod(_tmp1, cov_old_old, 2, 2, cov_old_new, 2);
					matrix_prod(_tmp2, cov_new_old, 2, 2, _tmp1, 2);
					matrix_sub(_tmp1, cov_new_new, _tmp2, 2, 2); // Conditional covariance matrix.

					U=new_zeros(2, 2);
					V=new_zeros(2, 2);
					S=malloc(2*sizeof(double));

					svd(_tmp1, U, S, V, 2); // Compute the SVD of cov in-place

					p_L=PyArray_FromDims(2, dims, NPY_DOUBLE);
					PyArray_AsCArray_Safe((PyObject **)&p_L, &c_L, PyArray_DIMS((PyArrayObject *)p_L), 2, PyArray_DescrFromType(NPY_DOUBLE));

					for(i=0; i<2; i++){
						for(j=0; j<2; j++){
							c_L[i][j]=U[i][j]*sqrt(S[j]);
						}
					}

					p_M=PyArray_FromDims(2, dims, NPY_DOUBLE);
					PyArray_AsCArray_Safe((PyObject **)&p_M, &c_M, PyArray_DIMS((PyArrayObject *)p_M), 2, PyArray_DescrFromType(NPY_DOUBLE));

					matrix_prod(c_M, cov_new_old, 2, 2, cov_old_old, 2);

					tmps[d][8*k]=c_L[0][0];
					tmps[d][1+8*k]=c_L[0][1];
					tmps[d][2+8*k]=c_L[1][0];
					tmps[d][3+8*k]=c_L[1][1];
					tmps[d][4+8*k]=c_M[0][0];
					tmps[d][5+8*k]=c_M[0][1];
					tmps[d][6+8*k]=c_M[1][0];
					tmps[d][7+8*k]=c_M[1][1];

					if(k != K-1){
						PyArray_Free(p_theta_k, theta);
					}
					
					free(S);
					free_mem_matrix(U, 2);
					free_mem_matrix(V, 2);
					free_mem_matrix(cov_new_new, 2);
					free_mem_matrix(cov_new_old, 2);
					free_mem_matrix(cov_old_new, 2);
					free_mem_matrix(cov_old_old, 2);
					free_mem_matrix(_tmp1, 2);
					free_mem_matrix(_tmp2, 2);
					free(t);
					free(prev_t);
					PyArray_Free(p_L, c_L);
					PyArray_Free(p_M, c_M);
				}

				// Cleanup
				free(cPids);		
				exit(0);
			}
			else{
				/* This code will only be executed by the parent */
				cPids[d] = p; // Record the PID of the child to properly wait for it later.
			}
		}
	}

	if(lvl == 0){
		p_result=PyList_New(D);		

		/* Wait for children to exit */
		int waitc, *statuses=NULL;
		statuses=malloc(D*sizeof(int));
		for(d=0; d<D; d++)
			statuses[d]=0;

		do {
		   waitc=0;
			for (d=0; d<D; d++) {
			   if (cPids[d]>0) {
				  	if (waitpid(cPids[d], &statuses[d], 0) != 0) { // WNOHANG -> 0: Should hang to avoid too many calls.
						/* Child is done */
						
						// Mark the child pid as done.
						cPids[d]=0;
						
						// Read the mapped memory to collect results
						kernel_types=PyList_GetItem(l_kernel_types, d);
						K=PyList_Size(kernel_types);
						row=PyList_New(K+1);

						for(k=0; k<K+1; k++){
							p_L = PyArray_FromDims(2, dims, NPY_DOUBLE);
							PyArray_AsCArray_Safe((PyObject **)&p_L, &c_L, PyArray_DIMS((PyArrayObject *)p_L), 2, PyArray_DescrFromType(NPY_DOUBLE));

							p_M = PyArray_FromDims(2, dims, NPY_DOUBLE);
							PyArray_AsCArray_Safe((PyObject **)&p_M, &c_M, PyArray_DIMS((PyArrayObject *)p_M), 2, PyArray_DescrFromType(NPY_DOUBLE));

							c_L[0][0]=tmps[d][8*k];
							c_L[0][1]=tmps[d][1+8*k];
							c_L[1][0]=tmps[d][2+8*k];
							c_L[1][1]=tmps[d][3+8*k];

							c_M[0][0]=tmps[d][4+8*k];
							c_M[0][1]=tmps[d][5+8*k];
							c_M[1][0]=tmps[d][6+8*k];
							c_M[1][1]=tmps[d][7+8*k];

							sub_row=PyList_New(2);
							PyList_SetItem(sub_row, 0, p_L);
							PyList_SetItem(sub_row, 1, p_M);
							PyList_SetItem(row, k, sub_row);
						}
						PyList_SetItem(p_result, d, row);
						munmap(tmps[d], 8*(1+K)*sizeof(double));
						
						// Notify if the child terminated abnormally.
						if(WIFSIGNALED(statuses[d])){
							PyErr_SetString(PyExc_ValueError, "Child was killed by signal!");
							printf("Child %d was killed by signal %d\n", d, WTERMSIG(statuses[d]));
						}
				  	}
				  	else {
						/* Still waiting on this child */
						waitc=1;
					}
				}
			   /* Give up timeslice and prevent hard loop */
			   //sleep(0);
			}
		} while (waitc);
		free(statuses);
		free(tmps);
	}

	// Cleanup
	free(cPids);
	return p_result;
}

/*
	Sample i.i.d standard normals for boundary conditions.
*/
PyObject *sample_whtn_bound_conds(PyObject *l_b_times){
	Py_ssize_t D;
	D=PyList_Size(l_b_times);

	if(D <= 0)
		PyErr_SetString(PyExc_ValueError, "The dimension should be positive.");

	int fds[D][2];
	pid_t *cPids =NULL;
	int lvl=0, d=0, p=0, status=0, k=0, K=0, i=0, dims[1]={0};
	PyObject *b_times=NULL, *p_result=NULL, *p_n=NULL;
	PyObject *row=NULL;
	double *tmp=NULL, **_tmp=NULL, *c_n=NULL;


	cPids=malloc(D*sizeof(pid_t));
	dims[0]=2;

	for(d=0; d<D; d++){
		if(lvl==0){
			status=pipe(fds[d]);

			if(status==-1){
				PyErr_SetString(PyExc_ValueError, "Pipe failed."); // Pipe failed
			}
			
			p=fork();
			if(p == 0){
				/* This code will only be executed by children */
				lvl = 1;
				close(fds[d][0]); // The child does not need to receive data from the parent
				b_times=PyList_GetItem(l_b_times, d);
				K=PyList_Size(b_times); // Number of boundary conditions
				tmp=malloc(2*K*sizeof(double));

				// Initialise the random seed.
				srand((unsigned)random_seed());
				// Sample the whitened values
				_tmp = randn(2*K);

				for(i=0; i<2*K; i++)
					tmp[i]=_tmp[i][0];

				// Write to the pipe
				status=write(fds[d][1], tmp, 2*K*sizeof(double));

				// Cleanup
				free(tmp);
				free_mem_matrix(_tmp, 2*K);
				free(cPids);

				if(status<0){
					PyErr_SetString(PyExc_ValueError, "An error occured while sending data to the parent.");
					printf("Oh dear, something went wrong with write() in %s! %s\n", __func__, strerror(errno));
				}
				
				exit(0);
			}
			else{
				/* This code will only be executed by the parent */
				cPids[d] = p; // Record the PID of the child to properly wait for it later.
				close(fds[d][1]); // The parent does not need to send data to its children.
			}
		}
	}

	if(lvl == 0){
		p_result=PyList_New(D);

		/* Listen to children on the pipes */
		for(d=0; d<D; d++){
			b_times=PyList_GetItem(l_b_times, d);
			K=PyList_Size(b_times);
			tmp=malloc(2*K*sizeof(double));
			
			// Listen to the d-th child on the pipe.
			status=read(fds[d][0], tmp, 2*K*sizeof(double));
			close(fds[d][0]);

			row=PyList_New(K);

			for(k=0; k<K; k++){
				p_n = PyArray_FromDims(1, dims, NPY_DOUBLE);
				PyArray_AsCArray_Safe((PyObject **)&p_n, &c_n, PyArray_DIMS((PyArrayObject *)p_n), 1, PyArray_DescrFromType(NPY_DOUBLE));

				c_n[0]=tmp[2*k];
				c_n[1]=tmp[1+2*k];

				PyList_SetItem(row, k, p_n);
			}
			PyList_SetItem(p_result, d, row);
			free(tmp);
		}
		

		/* Wait for children to exit */
		int waitc, *statuses=NULL;
		statuses=malloc(D*sizeof(int));
		for(d=0; d<D; d++)
			statuses[d]=0;

		do {
		   waitc=0;
			for (d=0; d<D; d++) {
			   if (cPids[d]>0) {
				  	if (waitpid(cPids[d], &statuses[d], 0) != 0) {// WNOHANG -> 0: Should hang to avoid too many calls.
						/* Child is done */
						cPids[d]=0;
						if(WIFSIGNALED(statuses[d]))
							printf("Child %d was killed by signal %d\n", d, WTERMSIG(statuses[d]));
				  	}
				  	else {
						/* Still waiting on this child */
						waitc=1;
					}
				}
			   /* Give up timeslice and prevent hard loop */
			   sleep(0);
			}
		} while (waitc);
		free(statuses);	
	}

	// Cleanup
	free(cPids);

	return p_result;
}


/*
	Sample i.i.d standard normals for inner (conditional) inner string values.
*/
PyObject *sample_whtn_string(PyObject *l_s_times){
	Py_ssize_t D;
	D=PyList_Size(l_s_times);

	if(D <= 0)
		PyErr_SetString(PyExc_ValueError, "The dimension should be positive.");

	int fds[D][2];
	pid_t *cPids =NULL;
	int lvl=0, d=0, p=0, status=0, k=0, K=0, i=0, dims[1]={0};
	PyObject *s_times=NULL, *p_result=NULL, *p_n=NULL;
	PyObject *row=NULL;
	double *tmp=NULL, **_tmp=NULL, *c_n=NULL;
	int tot_count=0, n_k=0, idx=0;


	cPids=malloc(D*sizeof(pid_t));
	dims[0]=2;

	for(d=0; d<D; d++){
		if(lvl==0){
			status=pipe(fds[d]);

			if(status==-1){
				PyErr_SetString(PyExc_ValueError, "Pipe failed."); // Pipe failed
			}
			
			p=fork();
			if(p == 0){
				/* This code will only be executed by children */
				lvl = 1;
				close(fds[d][0]); // The child does not need to receive data from the parent
				s_times=PyList_GetItem(l_s_times, d);
				K=PyList_Size(s_times); // Number of strings

				// Determine the total number of times at which to evaluate the SDGP
				tot_count=0;
				for(k=0; k<K; k++){
					n_k=PyList_Size(PyList_GetItem(s_times, k));
					if(PyList_Check(PyList_GetItem(s_times, k)) & (n_k>0)){
						tot_count+=n_k;
					}
				}

				tmp=malloc(2*tot_count*sizeof(double));

				// Initialise the random seed.
				srand((unsigned)random_seed());
				// Sample the whitened values
				_tmp = randn(2*tot_count);

				for(i=0; i<2*tot_count; i++)
					tmp[i]=_tmp[i][0];

				// Write to the pipe
				status=write(fds[d][1], tmp, 2*tot_count*sizeof(double));

				// Cleanup
				free(tmp);
				free_mem_matrix(_tmp, 2*tot_count);
				free(cPids);

				if(status<0){
					PyErr_SetString(PyExc_ValueError, "An error occured while sending data to the parent.");
					printf("Oh dear, something went wrong with write() in %s! %s\n", __func__, strerror(errno));
				}
				
				exit(0);
			}
			else{
				/* This code will only be executed by the parent */
				cPids[d] = p; // Record the PID of the child to properly wait for it later.
				close(fds[d][1]); // The parent does not need to send data to its children.
			}
		}
	}

	if(lvl == 0){
		p_result=PyList_New(D);

		/* Listen to children on the pipes */
		for(d=0; d<D; d++){
			s_times=PyList_GetItem(l_s_times, d);
			K=PyList_Size(s_times);

			// Determine the total number of times at which to evaluate the SDGP
			tot_count=0;
			for(k=0; k<K; k++){
				n_k=PyList_Size(PyList_GetItem(s_times, k));
				if(PyList_Check(PyList_GetItem(s_times, k)) & (n_k>0)){
					tot_count+=n_k;
				}
			}

			tmp=malloc(2*tot_count*sizeof(double));
			
			// Listen to the d-th child on the pipe.
			status=read(fds[d][0], tmp, 2*tot_count*sizeof(double));
			close(fds[d][0]);

			row=PyList_New(K);
			idx=0;

			for(k=0; k<K; k++){
				n_k=PyList_Size(PyList_GetItem(s_times, k));
				dims[0]=2*n_k;
				p_n = PyArray_FromDims(1, dims, NPY_DOUBLE);
				PyArray_AsCArray_Safe((PyObject **)&p_n, &c_n, PyArray_DIMS((PyArrayObject *)p_n), 1, PyArray_DescrFromType(NPY_DOUBLE));

				if(n_k>0){
					for(i=0; i<2*n_k; i++){
						c_n[i]=tmp[idx+i];
					}
					idx += 2*n_k;
				}
				
				PyList_SetItem(row, k, p_n);
			}

			PyList_SetItem(p_result, d, row);
			free(tmp);
		}
		

		/* Wait for children to exit */
		int waitc, *statuses=NULL;
		statuses=malloc(D*sizeof(int));
		for(d=0; d<D; d++)
			statuses[d]=0;

		do {
		   waitc=0;
			for (d=0; d<D; d++) {
			   if (cPids[d]>0) {
				  	if (waitpid(cPids[d], &statuses[d], 0) != 0) {// WNOHANG -> 0: Should hang to avoid too many calls.
						/* Child is done */
						cPids[d]=0;
						if(WIFSIGNALED(statuses[d]))
							printf("Child %d was killed by signal %d\n", d, WTERMSIG(statuses[d]));
				  	}
				  	else {
						/* Still waiting on this child */
						waitc=1;
					}
				}
			   /* Give up timeslice and prevent hard loop */
			   sleep(0);
			}
		} while (waitc);
		free(statuses);	
	}

	// Cleanup
	free(cPids);

	return p_result;
}

/*
	Similar to sample_sgps_from_ls except that both L factors and whithened values are provided.
*/
PyObject *compute_sgps_from_lxs(PyObject *l_kernel_types, PyObject *l_kernel_hypers, PyObject *l_b_times, PyObject *l_s_times, PyObject *l_Xb, PyObject *l_Xs, PyObject *l_bound_eig, PyObject *l_string_eig){
	PyObject *p_result=NULL, *p_l_z=NULL, *p_l_zp=NULL, *z_dict=NULL, *z_p_dict=NULL, *p_Xb=NULL, *p_Xs=NULL, *p_X=NULL, *bound_eig_d=NULL, *b_times_d=NULL, *s_times_d=NULL;
	PyObject *p_L=NULL, *p_M=NULL, *string_eig_d=NULL, *key=NULL, *value=NULL;
	int d=0, i=0, k=0, K=0, K2=0, n_k2=0, D=0;
	double **c_L=NULL, **c_M=NULL, *c_X=NULL, **tmp=NULL, **b_cond=NULL, **tmp2=NULL, **inner_values=NULL, **c_bound_cond=NULL;

	// When bound_eig or string_eig is none, recompute them.
	if((l_bound_eig == NULL) | (l_string_eig == NULL) || (l_bound_eig == Py_None) || (l_string_eig == Py_None)){
		l_bound_eig = compute_bound_ls(l_kernel_types, l_kernel_hypers, l_b_times);
		l_string_eig = cond_eigen_anals(l_kernel_types, l_kernel_hypers, l_b_times, l_s_times);
	}

	D=PyList_Size(l_kernel_types);

	if(D <= 0)
		PyErr_SetString(PyExc_ValueError, "The dimension should be positive.");

	if((PyList_Size(l_kernel_hypers) != D) | (PyList_Size(l_b_times) != D) | (PyList_Size(l_s_times) != D) | (PyList_Size(l_Xb) != D) | (PyList_Size(l_Xs) != D))
		PyErr_SetString(PyExc_ValueError, "List lengths should be consistent.");

	p_l_z = PyList_New(D);
	p_l_zp = PyList_New(D);
	p_result = PyList_New(2);

	for(d=0; d<D; d++){
		z_dict = PyDict_New();
		z_p_dict = PyDict_New();
		p_Xb = PyList_GetItem(l_Xb, d);
		bound_eig_d = PyList_GetItem(l_bound_eig, d);
		b_times_d = PyList_GetItem(l_b_times, d);

		b_cond = new_zeros(2, 1);
		tmp2 = new_zeros(2, 1);
		tmp = new_zeros(2, 1);
		K = PyList_Size(p_Xb); // Number of boundary times

		if((PyList_Size(b_times_d) != K) | (PyList_Size(bound_eig_d) != K) | (PyList_Size(p_Xb) != K))
			PyErr_SetString(PyExc_ValueError, "The number of boundary times is inconsistent among arguments.");

		c_bound_cond = new_zeros(2*K, 1);

		// Record boundary conditions
		for(k=0; k < K; k++){
			p_L = PyList_GetItem(PyList_GetItem(bound_eig_d, k), 0);
			PyArray_AsCArray((PyObject **)&p_L, &c_L, PyArray_DIMS((PyArrayObject *)p_L), 2, PyArray_DescrFromType(NPY_DOUBLE));

			p_M = PyList_GetItem(PyList_GetItem(bound_eig_d, k), 1);
			PyArray_AsCArray((PyObject **)&p_M, &c_M, PyArray_DIMS((PyArrayObject *)p_M), 2, PyArray_DescrFromType(NPY_DOUBLE));

			p_X = PyList_GetItem(p_Xb, k);
			PyArray_AsCArray((PyObject **)&p_X, &c_X, PyArray_DIMS((PyArrayObject *)p_X), 1, PyArray_DescrFromType(NPY_DOUBLE));

			tmp[0][0]=c_X[0];
			tmp[1][0]=c_X[1];

			matrix_prod(tmp2, c_L, 2, 2, tmp, 1);
			matrix_prod(tmp, c_M, 2, 2, b_cond, 1);
			matrix_add(b_cond, tmp2, tmp, 2, 1);

			// Record z_t and z_t^\prime
			key = float_as_idx(PyList_GetItem(b_times_d, k));

			value = PyFloat_FromDouble(b_cond[0][0]);
			PyDict_SetItem(z_dict, key, value);
			Py_DECREF(value);

			value = PyFloat_FromDouble(b_cond[1][0]);
			PyDict_SetItem(z_p_dict, key, value);
			Py_DECREF(value);

			Py_DECREF(key);

			c_bound_cond[2*k][0] = b_cond[0][0];
			c_bound_cond[1+2*k][0] = b_cond[1][0];

			PyArray_Free(p_L, c_L);
			PyArray_Free(p_M, c_M);
			PyArray_Free(p_X, c_X);		
		}

		free_mem_matrix(tmp, 2);
		free_mem_matrix(tmp2, 2);

		// Record inner DSGP values now
		string_eig_d = PyList_GetItem(l_string_eig, d);
		p_Xs = PyList_GetItem(l_Xs, d);
		s_times_d = PyList_GetItem(l_s_times, d);
		K2 = PyList_Size(string_eig_d); // Number of strings

		b_cond = new_zeros(4, 1);

		for(k=0; k<K2; k++){
			if((!PyList_Check(PyList_GetItem(s_times_d, k))) || ((n_k2 = 2*PyList_Size(PyList_GetItem(s_times_d, k))) <= 0))
				// The string has no inner times: move on.
				continue;

			p_L = PyList_GetItem(PyList_GetItem(string_eig_d, k), 0);
			PyArray_AsCArray((PyObject **)&p_L, &c_L, PyArray_DIMS((PyArrayObject *)p_L), 2, PyArray_DescrFromType(NPY_DOUBLE));

			p_M = PyList_GetItem(PyList_GetItem(string_eig_d, k), 2);
			PyArray_AsCArray((PyObject **)&p_M, &c_M, PyArray_DIMS((PyArrayObject *)p_M), 2, PyArray_DescrFromType(NPY_DOUBLE));

			p_X = PyList_GetItem(p_Xs, k);
			PyArray_AsCArray((PyObject **)&p_X, &c_X, PyArray_DIMS((PyArrayObject *)p_X), 1, PyArray_DescrFromType(NPY_DOUBLE));
			tmp = new_zeros(n_k2, 1);
			tmp2 = new_zeros(n_k2, 1);

			// From shape (n_k2) to (n_k2, 1)
			for(i=0; i<n_k2; i++){
				tmp[i][0]=c_X[i];
			}
			matrix_prod(tmp2, c_L, n_k2, n_k2, tmp, 1);

			b_cond[0][0]=c_bound_cond[2*k][0];
			b_cond[1][0]=c_bound_cond[1+2*k][0];
			b_cond[2][0]=c_bound_cond[2+2*k][0];
			b_cond[3][0]=c_bound_cond[3+2*k][0];

			matrix_prod(tmp, c_M, n_k2, 4, b_cond, 1);

			inner_values = new_zeros(n_k2, 1);
			matrix_add(inner_values, tmp2, tmp, n_k2, 1);

			// Record z_t and z_t^\prime at inner times
			for(i=0; i<n_k2/2; i++){
				key = float_as_idx(PyList_GetItem(PyList_GetItem(s_times_d, k), i));
				value = PyFloat_FromDouble(inner_values[2*i][0]);
				PyDict_SetItem(z_dict, key, value);
				Py_DECREF(value);

				value = PyFloat_FromDouble(inner_values[1+2*i][0]);
				PyDict_SetItem(z_p_dict, key, value);
				Py_DECREF(value);
				Py_DECREF(key);
			}

			free_mem_matrix(tmp, n_k2);
			free_mem_matrix(tmp2, n_k2);
			free_mem_matrix(inner_values, n_k2);

			PyArray_Free(p_L, c_L);
			PyArray_Free(p_M, c_M);
			PyArray_Free(p_X, c_X);			
		}

		PyList_SetItem(p_l_z, d, z_dict);
		PyList_SetItem(p_l_zp, d, z_p_dict);

		free_mem_matrix(b_cond, 4);
		free_mem_matrix(c_bound_cond, 2*K);
	}

	PyList_SetItem(p_result, 0, p_l_z);
	PyList_SetItem(p_result, 1, p_l_zp);

	return (PyObject *)p_result;
}


/*
	Returns old*cos(a) + new*sin(a) element-wise.
*/
PyObject *elliptic_tranform_lx(PyObject *old_l_x, PyObject *new_l_x, PyObject *a){
	int D=0, d=0, K=0, k=0, n=0, i=0, dims[2]={0};
	double c_a, *c_old_x=NULL, *c_new_x=NULL, *c_x=NULL;
	PyObject *p_x=NULL, *p_row=NULL, *p_result=NULL, *old_row=NULL, *new_row=NULL, *old_x=NULL, *new_x=NULL;

	D = PyList_Size(old_l_x);

	if((PyList_Size(new_l_x) != D))
			PyErr_SetString(PyExc_ValueError, "New and old objects should have the same dimension.");

	p_result = PyList_New(D);
	c_a = PyFloat_AsDouble(a);

	for(d=0; d<D; d++){
		old_row = PyList_GetItem(old_l_x, d);
		new_row = PyList_GetItem(new_l_x, d);
		K = PyList_Size(old_row);

		if((PyList_Size(new_row) != K))
			PyErr_SetString(PyExc_ValueError, "New and old objects should have the same dimension.");

		p_row = PyList_New(K);

		for(k=0; k<K; k++){
			old_x = PyList_GetItem(old_row, k);
			new_x = PyList_GetItem(new_row, k);

			if((!PyArray_Check(old_x)) || (!PyArray_Check(new_x)) || ((n = PyArray_DIMS((PyArrayObject *)old_x)[0]) != PyArray_DIMS((PyArrayObject *)new_x)[0]))
				PyErr_SetString(PyExc_ValueError, "New and old objects should have the same dimension.");

			dims[0]=PyArray_DIMS((PyArrayObject *)old_x)[0];
			dims[1]=PyArray_DIMS((PyArrayObject *)old_x)[1];

			p_x = PyArray_FromDims(1, dims, NPY_DOUBLE);
			PyArray_AsCArray_Safe(&p_x, &c_x, PyArray_DIMS((PyArrayObject *)p_x), 1, PyArray_DescrFromType(NPY_DOUBLE));
			PyArray_AsCArray(&new_x, &c_new_x, PyArray_DIMS((PyArrayObject *)new_x), 1, PyArray_DescrFromType(NPY_DOUBLE));
			PyArray_AsCArray(&old_x, &c_old_x, PyArray_DIMS((PyArrayObject *)old_x), 1, PyArray_DescrFromType(NPY_DOUBLE));

			if(n>0){
				for(i=0; i<n; i++){
					c_x[i] = c_old_x[i]*cos(c_a) + c_new_x[i]*sin(c_a);
				}
			}

			PyList_SetItem(p_row, k, p_x);
			PyArray_Free(new_x, c_new_x);
			PyArray_Free(old_x, c_old_x);
		}
		PyList_SetItem(p_result, d, p_row);
	}
	return p_result;
}

/*
	Log-likelihood in the whitened case.
*/
PyObject *log_lik_whtn(PyObject *l_xb, PyObject *l_xs){
	int D=0, d=0, K=0, k=0, n=0, i=0;
	double c_res=0.0, *c_x=NULL;
	PyObject *p_x=NULL, *p_row=NULL, *p_result=NULL;

	D = PyList_Size(l_xb);
	for(d=0; d<D; d++){
		p_row = PyList_GetItem(l_xb, d);
		K = PyList_Size(p_row);

		for(k=0; k<K; k++){
			p_x = PyList_GetItem(p_row, k);
			PyArray_AsCArray(&p_x, &c_x, PyArray_DIMS((PyArrayObject *)p_x), 1, PyArray_DescrFromType(NPY_DOUBLE));
			n = (int)(PyArray_DIMS((PyArrayObject *)p_x)[0]);

			for(i=0; i<n; i++){
				c_res += -0.5*log(2.0*M_PI)-0.5*c_x[i]*c_x[i];
			}
			PyArray_Free(p_x, c_x);
		}
	}

	D = PyList_Size(l_xs);
	for(d=0; d<D; d++){
		p_row = PyList_GetItem(l_xs, d);
		K = PyList_Size(p_row);

		for(k=0; k<K; k++){
			p_x = PyList_GetItem(p_row, k);
			PyArray_AsCArray(&p_x, &c_x, PyArray_DIMS((PyArrayObject *)p_x), 1, PyArray_DescrFromType(NPY_DOUBLE));
			n = (int)(PyArray_DIMS((PyArrayObject *)p_x)[0]);

			for(i=0; i<n; i++){
				c_res += -0.5*log(2.0*M_PI)-0.5*c_x[i]*c_x[i];
			}
			PyArray_Free(p_x, c_x);
		}
	}

	p_result = PyFloat_FromDouble(c_res);
	return p_result;
}

/*
	Sample i.i.d standard normals.
*/
PyObject *sample_norm_hypers(PyObject *l_shape, PyObject *l_means, PyObject *std){
	Py_ssize_t D;
	D=PyList_Size(l_shape);

	if(D <= 0)
		PyErr_SetString(PyExc_ValueError, "The dimension should be positive.");

	int fds[D][2];
	pid_t *cPids =NULL;
	int lvl=0, d=0, p=0, status=0, k=0, K=0, i=0, dims[1]={0};
	PyObject *hypers_shape=NULL, *hypers_mean=NULL, *p_result=NULL, *p_n=NULL, *p_hypers=NULL, *p_means=NULL;
	PyObject *row=NULL;
	double *tmp=NULL, **_tmp=NULL, *c_n=NULL, *c_hypers=NULL, *c_means=NULL, c_std=0.0;
	int tot_count=0, n_k=0, idx=0;

	cPids=malloc(D*sizeof(pid_t));
	dims[0]=2;

	for(d=0; d<D; d++){
		if(lvl==0){
			status=pipe(fds[d]);

			if(status==-1){
				PyErr_SetString(PyExc_ValueError, "Pipe failed."); // Pipe failed
			}
			
			p=fork();
			if(p == 0){
				/* This code will only be executed by children */
				lvl = 1;
				close(fds[d][0]); // The child does not need to receive data from the parent
				hypers_shape = PyList_GetItem(l_shape, d);
				K=PyList_Size(hypers_shape); // Number of strings

				// Determine the total number of times at which to evaluate the SDGP
				tot_count=0;
				for(k=0; k<K; k++){
					p_hypers = PyList_GetItem(hypers_shape, k);
					if(PyArray_Check(p_hypers) & ((n_k = (int)(PyArray_DIMS((PyArrayObject *)p_hypers)[0])) > 0)){
						tot_count += n_k;
					}
				}

				tmp=malloc(tot_count*sizeof(double));

				// Initialise the random seed.
				srand((unsigned)random_seed());
				// Sample the whitened values
				_tmp = randn(tot_count);

				for(i=0; i<tot_count; i++)
					tmp[i]=_tmp[i][0];

				// Write to the pipe
				status=write(fds[d][1], tmp, tot_count*sizeof(double));

				// Cleanup
				free(tmp);
				free_mem_matrix(_tmp, tot_count);
				free(cPids);

				if(status<0){
					PyErr_SetString(PyExc_ValueError, "An error occured while sending data to the parent.");
					printf("Oh dear, something went wrong with write() in %s! %s\n", __func__, strerror(errno));
				}
				
				exit(0);
			}
			else{
				/* This code will only be executed by the parent */
				cPids[d] = p; // Record the PID of the child to properly wait for it later.
				close(fds[d][1]); // The parent does not need to send data to its children.
			}
		}
	}

	if(lvl == 0){
		c_std = PyFloat_AsDouble(std);
		p_result=PyList_New(D);

		if((l_means != Py_None) && (PyList_Size(l_means) != D))
			PyErr_SetString(PyExc_ValueError, "Shape and mean parameters do not have the same dimension.");

		/* Listen to children on the pipes */
		for(d=0; d<D; d++){
			hypers_shape=PyList_GetItem(l_shape, d);

			if(l_means != Py_None)
				hypers_mean=PyList_GetItem(l_means, d);

			K=PyList_Size(hypers_shape);

			// Determine the total number of times at which to evaluate the SDGP
			tot_count=0;
			for(k=0; k<K; k++){
				p_hypers = PyList_GetItem(hypers_shape, k);
				if(PyArray_Check(p_hypers) & ((n_k = (int)(PyArray_DIMS((PyArrayObject *)p_hypers)[0])) > 0)){
					tot_count += n_k;
				}
			}
			tmp=malloc(tot_count*sizeof(double));
			
			// Listen to the d-th child on the pipe.
			status=read(fds[d][0], tmp, tot_count*sizeof(double));
			close(fds[d][0]);

			row=PyList_New(K);
			idx=0;

			for(k=0; k<K; k++){
				p_hypers = PyList_GetItem(hypers_shape, k);
				PyArray_AsCArray((PyObject **)&p_hypers, &c_hypers, PyArray_DIMS((PyArrayObject *)p_hypers), 1, PyArray_DescrFromType(NPY_DOUBLE));
				n_k = (int)(PyArray_DIMS((PyArrayObject *)p_hypers)[0]);

				if(hypers_mean != NULL){
					p_means=PyList_GetItem(hypers_mean, k);
					PyArray_AsCArray((PyObject **)&p_means, &c_means, PyArray_DIMS((PyArrayObject *)p_means), 1, PyArray_DescrFromType(NPY_DOUBLE));
				}

				dims[0]=n_k;
				p_n = PyArray_FromDims(1, dims, NPY_DOUBLE);
				PyArray_AsCArray_Safe((PyObject **)&p_n, &c_n, PyArray_DIMS((PyArrayObject *)p_n), 1, PyArray_DescrFromType(NPY_DOUBLE));

				if(n_k>0){
					for(i=0; i<n_k; i++){
						if(c_means == NULL){
							c_n[i] = tmp[idx+i];
						}else{
							c_n[i] = c_std*tmp[idx+i] + c_means[i];
						}
					}
					idx += n_k;
				}
				
				PyList_SetItem(row, k, p_n);
				PyArray_Free(p_hypers, c_hypers);
				if(c_means != NULL)
					PyArray_Free(p_means, c_means);
			}

			PyList_SetItem(p_result, d, row);
			free(tmp);
		}
		

		/* Wait for children to exit */
		int waitc, *statuses=NULL;
		statuses=malloc(D*sizeof(int));
		for(d=0; d<D; d++)
			statuses[d]=0;

		do {
		   waitc=0;
			for (d=0; d<D; d++) {
			   if (cPids[d]>0) {
				  	if (waitpid(cPids[d], &statuses[d], 0) != 0) {// WNOHANG -> 0: Should hang to avoid too many calls.
						/* Child is done */
						cPids[d]=0;
						if(WIFSIGNALED(statuses[d]))
							printf("Child %d was killed by signal %d\n", d, WTERMSIG(statuses[d]));
				  	}
				  	else {
						/* Still waiting on this child */
						waitc=1;
					}
				}
			   /* Give up timeslice and prevent hard loop */
			   sleep(0);
			}
		} while (waitc);
		free(statuses);	
	}
	// Cleanup
	free(cPids);

	return p_result;
}


/*
	Returns h_max/(1+exp(-h_norm)).
*/
PyObject *scaled_sigmoid(PyObject *l_hypers_max, PyObject *l_hypers_norm){
	int D=0, d=0, K=0, k=0, n=0, i=0, dims[2]={0};
	double *c_max_x=NULL, *c_norm_x=NULL, *c_x=NULL;
	PyObject *p_x=NULL, *p_row=NULL, *p_result=NULL, *max_row=NULL, *norm_row=NULL, *norm_x=NULL, *max_x=NULL;

	D = PyList_Size(l_hypers_max);

	if((PyList_Size(l_hypers_norm) != D))
			PyErr_SetString(PyExc_ValueError, "Max and normalized objects should have the same dimension.");

	p_result = PyList_New(D);

	for(d=0; d<D; d++){
		max_row = PyList_GetItem(l_hypers_max, d);
		norm_row = PyList_GetItem(l_hypers_norm, d);
		K = PyList_Size(max_row);

		if((PyList_Size(norm_row) != K))
			PyErr_SetString(PyExc_ValueError, "Max and normalized objects should have the same dimension.");

		p_row = PyList_New(K);

		for(k=0; k<K; k++){
			max_x = PyList_GetItem(max_row, k);
			norm_x = PyList_GetItem(norm_row, k);

			if((!PyArray_Check(max_x)) || (!PyArray_Check(norm_x)) || ((n = PyArray_DIMS((PyArrayObject *)max_x)[0]) != PyArray_DIMS((PyArrayObject *)norm_x)[0]))
				PyErr_SetString(PyExc_ValueError, "Max and normalized objects should have the same dimension.");

			dims[0]=PyArray_DIMS((PyArrayObject *)max_x)[0];
			dims[1]=PyArray_DIMS((PyArrayObject *)max_x)[1];

			p_x = PyArray_FromDims(1, dims, NPY_DOUBLE);
			PyArray_AsCArray_Safe(&p_x, &c_x, PyArray_DIMS((PyArrayObject *)p_x), 1, PyArray_DescrFromType(NPY_DOUBLE));
			PyArray_AsCArray(&norm_x, &c_norm_x, PyArray_DIMS((PyArrayObject *)norm_x), 1, PyArray_DescrFromType(NPY_DOUBLE));
			PyArray_AsCArray(&max_x, &c_max_x, PyArray_DIMS((PyArrayObject *)max_x), 1, PyArray_DescrFromType(NPY_DOUBLE));

			if(n>0){
				for(i=0; i<n; i++){
					c_x[i] = c_max_x[i]/(1.0 + exp(-c_norm_x[i]));
				}
			}

			PyList_SetItem(p_row, k, p_x);
			PyArray_Free(norm_x, c_norm_x);
			PyArray_Free(max_x, c_max_x);
		}
		PyList_SetItem(p_result, d, p_row);
	}
	return p_result;
}

/*
	Compute the log likelihood of a DSGP. TODO: this implementation is wrong.
*/
PyObject *model_log_lik(PyObject *data, PyObject *l_sgp, PyObject *link_f_type, PyObject *ll_type, PyObject *noise_var){
	
	if(!PyArray_Check(data)){
		PyErr_SetString(PyExc_ValueError, "The data should be a 2d ndarray.");
		Py_RETURN_NONE;
	}

	PyObject *p_result=NULL, *key=NULL;
	double **c_data=NULL, log_lik=0.0, fi=0.0, c_n_var=0.0;
	int n=0, i=0, d=0, D=0;
	char *c_ll_type=NULL, *c_link_f_type=NULL;

	n = PyArray_DIMS((PyArrayObject *)data)[0];
	D = -1 + PyArray_DIMS((PyArrayObject *)data)[1];
	if(D<=0){
		PyErr_SetString(PyExc_ValueError, "data should have at least 2 columns. The last column should contain instance classes.");
		Py_RETURN_NONE;
	}

	if((!PyList_Check(l_sgp)) || (PyList_Size(l_sgp) != D))
	{
		PyErr_SetString(PyExc_ValueError, "l_sgp has an invalid format.");
		Py_RETURN_NONE;	
	}

	PyArray_AsCArray(&data, &c_data, PyArray_DIMS((PyArrayObject *)data), 2, PyArray_DescrFromType(NPY_DOUBLE));
	c_ll_type = PyString_AsString(ll_type);
	c_link_f_type = PyString_AsString(link_f_type);

	if(noise_var != Py_None)
		c_n_var = PyFloat_AsDouble(noise_var);


	for(i=0; i<n; i++){
		// Compute the latent function value
		if(strcmp(c_link_f_type, "prod") == 0){
			fi = 1.0;
		}
		else if(strcmp(c_link_f_type, "sum") == 0){
			fi = 0.0;
		}
		else{
			PyErr_SetString(PyExc_ValueError, "sum and prod are the only link functions allowed thus far.");
			Py_RETURN_NONE;
		}
		
		for(d=0; d<D; d++){
			key = float_as_idx(PyFloat_FromDouble(c_data[i][d]));
			if(strcmp(c_link_f_type, "prod") == 0){
				fi *= PyFloat_AsDouble(PyDict_GetItem(PyList_GetItem(l_sgp, d), key));
			}

			if(strcmp(c_link_f_type, "sum") == 0){
				fi += PyFloat_AsDouble(PyDict_GetItem(PyList_GetItem(l_sgp, d), key));
			}
			Py_DECREF(key);
		}

		if((strcmp(c_ll_type, "gaussian") ==0) && (c_n_var <=0.0)){
			PyErr_SetString(PyExc_ValueError, "For Gaussian regression models, the noise variance should be strictly positive.");
			Py_RETURN_NONE;
		}
		// Increment the likelihood
		if(strcmp(c_ll_type, "gaussian") == 0){
			log_lik += -0.5*log(c_n_var) -0.5*log(2.0*M_PI) -0.5*(c_data[i][D] - fi)*(c_data[i][D] - fi)/c_n_var;
		}
		else if(strcmp(c_ll_type, "logit") == 0){
			log_lik += -c_data[i][D]*log(1.0 + exp(-fi)) - (1.0-c_data[i][D])*log(1.0 + exp(fi));
		}
		else{
			PyErr_SetString(PyExc_ValueError, "gaussian and logit are the only likelihoods supported thus far.");
			Py_RETURN_NONE;
		}
	}

	PyArray_Free(data, c_data);
	p_result = PyFloat_FromDouble(log_lik);
	return p_result;
}