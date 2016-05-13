from StringGPy.samplers.base_sampler cimport BaseSGPSampler
cimport numpy as np
import numpy as np


'''
	MCMC sampler for binary classification tasks using string Gaussian processes.
		The classes should be 1/-1. P(yi=1)=1.0/(1.0 + exp(-f(xi)))
'''
cdef class SGPBinaryClassifier(BaseSGPSampler):
	def __init__(self, np.ndarray[np.float64_t, ndim=2] X, np.ndarray[np.float64_t, ndim=1] Y_train,\
			int n_train, bytes kernel_type, np.ndarray[np.float64_t, ndim=2] hypers_max,\
			bytes link_f_type, int should_print, bytes hypers_prior_type, int plot, infer_change_pts=0):
		super(SGPBinaryClassifier, self).__init__(X, Y_train, n_train, kernel_type, hypers_max, 0, "logit",\
			link_f_type, should_print, hypers_prior_type, plot, mean_f_lambda=lambda f: 1.0/(1.0+np.exp(-f)),\
			infer_change_pts=infer_change_pts)

	'''
	Compute the log-likelihood under the observation model.
	log-likelihood
	'''
	def compute_log_lik(self, np.ndarray[np.float64_t, ndim=1] f, np.ndarray[np.float64_t, ndim=3] dsgp):

		cdef np.ndarray[np.float64_t, ndim=1] data_f = f[:self.n_train]
		cdef np.ndarray[np.float64_t, ndim=1] e_data_f
		cdef np.ndarray[np.float64_t, ndim=1] log_arg1
		cdef np.ndarray[np.float64_t, ndim=1] log_arg2
		cdef np.ndarray[np.float64_t, ndim=1] data_Y = self.Y_train

		if self.ll_type == "logit":
			if self.use_GPU:
				try:
					e_data_f = self.stringgpu_worker.exp(data_f)
					log_arg1 = (1.0 + 1.0/e_data_f).astype(np.float64)
					log_arg2 = (1.0 + e_data_f).astype(np.float64)

					return -0.5*np.sum(\
						(1.0+data_Y)*self.stringgpu_worker.log(log_arg1)+\
						(1.0-data_Y)*self.stringgpu_worker.log(log_arg2))

				except Exception as details:
					print 'Disabling GPU computing'
					print details
					self.use_GPU = False

			return -0.5*np.sum((1.0+data_Y)*np.log(1.0 + np.exp(-data_f))\
				+ (1.0-data_Y)*np.log(1.0 + np.exp(data_f)))
		else:
			assert self.ll_type == "logit", "The only binary classification implemented if logistic regression"
