'''
	Import some goodies from the C library
'''
cdef extern from "c_utilities.c":
	int c_n_spectral_comp(int n, char*  type) nogil
	
	void c_shuffle(double* inp, double* out, unsigned long* ind, unsigned long n) nogil

	void c_factors(int i, int j, int n, int d, double* l_factors, double* m_factors, double* l_hypers,
		double* X, int n_theta, char* k_type) nogil

	void c_string_cov(double *res, double *s_times, unsigned int n_s_times, double *b_times,
        unsigned int n_b_times, double *thetas, unsigned int n_theta, char *k_type) nogil

	void c_string_deriv_cov(double *res, double *s_times, unsigned int n_s_times, double *b_times,\
        unsigned int n_b_times, double *thetas, unsigned int n_theta, char *k_type) nogil