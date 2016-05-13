import numpy as np
cimport numpy as np
from StringGPy.utilities.goodies cimport c_string_cov, c_string_deriv_cov

from GPy.kern import Kern
from GPy.core.parameterization import Param
from GPy.core.parameterization.transformations import Logexp, Logistic
import sys
from numpy.core.umath_tests import inner1d

import warnings
warnings.filterwarnings("ignore")

LOG_LEVEL_INFO  = 0
LOG_LEVEL_DEBUG = 1

'''
Construct the global covariance matrix between the values of the string GP at
	X and its values at X2
'''
def string_cov(X, X2, np.ndarray[np.float64_t, ndim=2, mode='c'] thetas,\
		np.ndarray[np.float64_t, ndim=1] b_times, char* k_type, input_index=0):
	cdef np.ndarray[np.float64_t, ndim=2, mode='c'] cov
	cdef np.ndarray[np.float64_t, ndim=2, mode='c'] res
	cdef int n_b_times = b_times.shape[0]
	cdef int n_thetas  = thetas.shape[1]
	cdef np.ndarray[np.float64_t, ndim=1] s_times
	cdef np.ndarray[long, ndim=1] ind_x, ind_x2

	if (X2 == None) or np.array_equal(X, X2):
		X2 = X
		if len(X.shape) > 1:
			flat_X = X[:, input_index]
		else:
			flat_X = X
		s_times = np.array(sorted(set(list(flat_X))))
		ind_x = np.searchsorted(s_times, flat_X)
		ind_x2 = ind_x
	else:
		if len(X.shape) > 1:
			flat_X = X[:, input_index]
			flat_X2 = X2[:, input_index]
		else:
			flat_X = X
			flat_X2 = X2

		s_times = np.array(sorted(set(list(flat_X) + list(flat_X2))))
		ind_x = np.searchsorted(s_times, flat_X)
		ind_x2 = np.searchsorted(s_times, flat_X2)

	cov = np.empty((s_times.shape[0], s_times.shape[0]), order='C')

	cdef int n_s_times = s_times.shape[0]

	with nogil:
		c_string_cov(&cov[0,0], &s_times[0], n_s_times, &b_times[0], n_b_times, &thetas[0,0], n_thetas, k_type)

	res = cov[np.ix_(ind_x, ind_x2)]
	return res

'''
Construct the global covariance matrix between the values of the derivative string GP at
	X and its values at X2
'''
def string_deriv_cov(X, X2, np.ndarray[np.float64_t, ndim=2, mode='c'] thetas,\
		np.ndarray[np.float64_t, ndim=1] b_times, char* k_type, input_index=0):
	cdef np.ndarray[np.float64_t, ndim=2, mode='c'] cov
	cdef np.ndarray[np.float64_t, ndim=2, mode='c'] res
	cdef int n_b_times = b_times.shape[0]
	cdef int n_thetas  = thetas.shape[1]
	cdef np.ndarray[np.float64_t, ndim=1] s_times
	cdef np.ndarray[long, ndim=1] ind_x, ind_x2
	cdef list deriv_ind_x, deriv_ind_x2

	if (X2 == None) or np.array_equal(X, X2):
		X2 = X
		if len(X.shape) > 1:
			flat_X = X[:, input_index]
		else:
			flat_X = X
		s_times = np.array(sorted(set(list(flat_X))))
		ind_x = np.searchsorted(s_times, flat_X)
		ind_x2 = ind_x
	else:
		if len(X.shape) > 1:
			flat_X = X[:, input_index]
			flat_X2 = X2[:, input_index]
		else:
			flat_X = X
			flat_X2 = X2

		s_times = np.array(sorted(set(list(flat_X) + list(flat_X2))))
		ind_x = np.searchsorted(s_times, flat_X)
		ind_x2 = np.searchsorted(s_times, flat_X2)

	deriv_ind_x = []
	deriv_ind_x2 = []

	for i in range(len(ind_x)):
		deriv_ind_x += [2*ind_x[i]]
		deriv_ind_x += [2*ind_x[i]+1]

	for i in range(len(ind_x2)):
		deriv_ind_x2 += [2*ind_x2[i]]
		deriv_ind_x2 += [2*ind_x2[i]+1]

	cdef int n_s_times = s_times.shape[0]
	cov = np.empty((2*n_s_times, 2*n_s_times), order='C')

	with nogil:
		c_string_deriv_cov(&cov[0,0], &s_times[0], n_s_times, &b_times[0], n_b_times, &thetas[0,0], n_thetas, k_type)
	res = cov[np.ix_(deriv_ind_x, deriv_ind_x2)]
	return res


'''
Generic kernel for String GP indexed on undimensional input spaces.
'''
class StringGPKern(Kern):
	def __init__(self, kernel_type, min_input, max_input, name='StringGP',\
			n_inner_b_times=0, thetas=None, n_mixt=1, verbose=None,\
				 b_times=None, input_index=0):
					 
		super(StringGPKern, self).__init__(1, None, name)

		assert kernel_type in ("se", "ma32", "ma52", "rq", "sse", "sm", "sma32", "sma52", "srq", "period"),\
			"kernel_type not supported."
		assert min_input < max_input, "min_input should be smaller than max_input"
		
		# Step 0: Record what shouldn't be linked.
		self.k_type = kernel_type
		self.n_b_times = 2 + n_inner_b_times # The min and max input values are
		self.n_strings = 1 + n_inner_b_times
		self.n_mixt = n_mixt
		self.verbose = verbose
		self.min_input = min_input
		self.max_input = max_input
		self.input_index = input_index

		# Step 1: Set and link boundary times (which are also change points).
		# 	Min/Max inputs should be boundary times but shouldn't be linked,
		#	as they need not be learned.
		if self.verbose in (LOG_LEVEL_DEBUG, LOG_LEVEL_INFO):
			print 'Initializing boundary times'

		if (b_times == None) and (n_inner_b_times > 0):
			self.learn_b_times = True
			if n_inner_b_times > 0:
				p_drv = np.zeros((self.n_strings))

				for i in xrange(self.n_strings):
					setattr(self, 'b_time_pdrv_' + str(i), Param('b_time_pdrv_' + str(i), p_drv[i], Logistic(-1.0, 1.0)))
					self.link_parameter(getattr(self, 'b_time_pdrv_' + str(i)))
		elif n_inner_b_times == 0:
			self.b_times = np.array([min_input, max_input])
			self.learn_b_times = False
			self.n_b_times = 2
			self.n_strings = 1		
		else:
			self.b_times = np.sort(b_times)
			self.learn_b_times = False
			self.n_b_times = len(self.b_times)
			self.n_strings = self.n_b_times - 1

		# Step 2: Set and link hyper-parameters.
		if self.verbose in (LOG_LEVEL_DEBUG, LOG_LEVEL_INFO):
			print 'Initializing thetas'

		if thetas == None:
			if self.k_type in ("se", "ma32", "ma52"):
				shape = (self.n_strings, 2)
				thetas = np.ones(shape)
				thetas[:,1] = 0.5*(max_input-min_input) # Input scale

			if self.k_type in ("rq", "period"): 
				shape = (self.n_strings, 3)
				thetas = np.ones(shape)
				thetas[:,1] = 0.5*(max_input-min_input) # Input scale

			if self.k_type in ("sse", "sma32", "sma52", "sm"):
				shape = (self.n_strings, 3*self.n_mixt)
				thetas = np.ones(shape)
				thetas[:,1::3] = 0.5*(max_input-min_input) # Input scales
				thetas[:,2::3] = 1.0*np.arange(1, 1+self.n_mixt)/(max_input-min_input)

			if self.k_type == "srq":
				shape = (self.n_strings, 4*self.n_mixt)
				thetas = np.ones(shape) 
				thetas[:,1::4] = 0.5*(max_input-min_input)
				thetas[:,2::4] = 1.0*np.arange(1, 1+self.n_mixt)/(max_input-min_input)

		self.thetas_shape = thetas.shape

		for i in range(self.thetas_shape[0]):
			for j in range(self.thetas_shape[1]):
				setattr(self, 'theta_' + str(i) + '_' + str(j), Param('theta_' + str(i) + '_'+ str(j), thetas[i][j], Logexp()))
				self.link_parameter(getattr(self, 'theta_' + str(i) + '_'+ str(j)))    					  	
	
	'''
	Method that uses the linked parameters to form a matrix of unconditional
		string hyper-parameters. There should be as many rows as strings, and 
		the matrix should be C-contiguous.
	'''
	def _get_thetas(self):
		thetas = np.empty(self.thetas_shape)
		for i in range(self.thetas_shape[0]):
			for j in range(self.thetas_shape[1]):
				thetas[i][j] = getattr(self, 'theta_' + str(i) + '_'+ str(j))[0]
		return np.ascontiguousarray(thetas)     


	'''
	Construct the vector of boundary times from object attributes.
	'''
	def _get_b_times(self):
		if self.learn_b_times:
			p_drv = [getattr(self, 'b_time_pdrv_' + str(i))[0] for i in range(self.n_strings)]
			# Boundary times as partition of [min_input, max_input]
			part = np.cumsum(np.exp(p_drv))
			part /= part[-1]
			part = np.array([0.0] + list(part))
			b_times = self.min_input + (self.max_input-self.min_input)*part
			return b_times
		else:
			return self.b_times 


	'''
	Construct the global covariance matrix between the values of the string GP at
		X and its values at X2
	'''
	def _cov_matrix(self, X, X2, np.ndarray[np.float64_t, ndim=2, mode='c'] thetas,\
			np.ndarray[np.float64_t, ndim=1] b_times):
		cov = string_cov(X, X2, thetas, b_times, self.k_type, input_index=self.input_index)
		return cov

	'''
	Workout noise variance assignment from 
		boundary times
	'''
	def Y_metadata(self, X):       	
		flat_X = X[:, self.input_index]
		n_x = len(flat_X)
		res = {'output_index':np.arange(n_x)[:,None]}
		_tmp = zip(flat_X, range(n_x))
		_tmp = np.array(sorted(_tmp, key=lambda x:x[0], reverse=False))

		b_times = self._get_b_times()
		n_b = len(b_times)
		j = 0

		for i in range(n_x):
			while (j+1 < n_b-1) and (_tmp[i][0] > b_times[j+1]):
				j += 1

			res['output_index'][_tmp[i][1]][0] = j

		return res

	def K(self, X, X2):
		"""
		Compute the kernel function.

		:param X: the first set of inputs to the kernel
		:param X2: (optional) the second set of arguments to the kernel. If X2
				   is None, this is passed through to the 'part' object, which
				   handLes this as X2 == X.
		""" 
		return self._cov_matrix(X, X2, self._get_thetas(), self._get_b_times())

	'''
	Diagonal of the covariance matrix
	'''
	def Kdiag(self, X):
		return np.diag(self.K(X, X))

	  
	def update_gradients_full(self, dL_dK, X, X2):
		"""
		Given the derivative of the objective wrt the covariance matrix
		(dL_dK), compute the gradient wrt the parameters of this kernel,
		and store in the parameters object as e.g. self.variance.gradient
		"""
		if self.verbose == LOG_LEVEL_INFO:
			print '------------------------'
			print 'In update_gradients_full'
			print '------------------------'

		h = 1.0e-6        
		current_thetas = self._get_thetas()
		current_b_times = self._get_b_times()
		current_cov = self.K(X, X2)

		if self.learn_b_times:
			current_p_drv = [getattr(self, 'b_time_pdrv_' + str(i))[0] for i in range(self.n_strings)]
			# Derivatives with respect to boundary times
			for i in range(self.n_strings):
				p_drv = current_p_drv[:] # !Slicing is important here.
				p_drv[i] = p_drv[i] + h

				# Boundary times as partition of [min_input, max_input]
				part = np.cumsum(np.exp(p_drv))
				part /= part[-1]
				part = np.array([0.0] + list(part))
				b_times = self.min_input + (self.max_input-self.min_input)*part

				cov_h = self._cov_matrix(X, X2, current_thetas, b_times)
				dcov = (1.0/h)*(cov_h-current_cov)
				setattr(getattr(self, 'b_time_pdrv_' + str(i)), 'gradient', np.sum(inner1d(dL_dK, dcov.T)))

				if self.verbose == LOG_LEVEL_INFO:
					sys.stdout.flush() 
					print 'dLdb_time', str(1+i), getattr(self, 'b_time_pdrv_' + str(i)).gradient[0]

		# Derivatives with respect to hyper-parameters
		for i in range(self.thetas_shape[0]):
			for j in range(self.thetas_shape[1]):
				thetas = current_thetas.copy()
				thetas[i][j] = thetas[i][j] + h
				cov_h = self._cov_matrix(X, X2, thetas, current_b_times)
				dcov = (1.0/h)*(cov_h-current_cov)
				setattr(getattr(self, 'theta_' + str(i) + '_'+ str(j)), 'gradient', np.sum(inner1d(dL_dK, dcov.T)))

				if self.verbose == LOG_LEVEL_INFO:
					sys.stdout.flush() 
					print 'dLdthetas', str(i), str(j), getattr(self, 'theta_' + str(i) + '_'+ str(j)).gradient[0]
