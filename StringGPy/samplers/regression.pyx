from StringGPy.samplers.base_sampler cimport BaseSGPSampler
cimport numpy as np
import numpy as np


'''
	MCMC sampler for regression tasks using string Gaussian processes.
'''
cdef class SGPRegressor(BaseSGPSampler):
	def __init__(self, np.ndarray[np.float64_t, ndim=2] X, np.ndarray[np.float64_t, ndim=1] Y_train,\
			int n_train, bytes kernel_type, np.ndarray[np.float64_t, ndim=2] hypers_max,\
			bytes link_f_type, int should_print, bytes hypers_prior_type, int plot, infer_change_pts=0):
		super(SGPRegressor, self).__init__(X, Y_train, n_train, kernel_type, hypers_max, 1, "gaussian",\
			link_f_type, should_print, hypers_prior_type, plot, infer_change_pts=infer_change_pts)

	'''
	Compute the log-likelihood under the observation model.
	'''
	def compute_log_lik(self, np.ndarray[np.float64_t, ndim=1] f,\
			np.ndarray[np.float64_t, ndim=3] dsgp):

		cdef np.ndarray[np.float64_t, ndim=1] data_f = f[:self.n_train]
		cdef np.ndarray[np.float64_t, ndim=1] data_Y = self.Y_train
		cdef np.ndarray[np.float64_t, ndim=1] errors

		# Numpy tends to be faster than StringGPu for dot products and sums
		if self.ll_type == "gaussian":
			errors = data_f-data_Y
			return -0.5*np.dot(errors, errors)/self.noise_var
		else:
			assert self.ll_type == "gaussian", "Only likelihood model implemented thus far is Gaussian"

