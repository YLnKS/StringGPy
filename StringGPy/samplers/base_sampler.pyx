# -*- coding: utf-8 -*-
# cython: boundscheck=False
# cython: initializedcheck=False
# cython: wraparound=False
# cython: linetrace=True
# filename: cy_samplers.pyx
"""
Created on Thu Jun 18 16:49:14 2015

@author: ylkomsamo
"""
import numpy as np
from numpy.dual import svd
cimport numpy as np
from libc.math cimport sin, cos, acos, exp, sqrt, fabs, M_PI
from libc.stdlib cimport malloc, free
from scipy.stats import invgamma, gamma
import cython
from cython.parallel import prange, parallel, threadid
from time import time, clock
import multiprocessing
cimport openmp
from ctypes import c_ulong, c_uint
from cpython cimport bool
try:
    from StringGPu import GPUFactorsWorker
except:
    pass
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
import scipy.stats as stats
import scipy.special as sp
from StringGPy.utilities.goodies cimport c_n_spectral_comp, c_shuffle, c_factors

'''
    MCMC sampler to solve some supervised learning tasks using string Gaussian processes.
'''
cdef class BaseSGPSampler:

    def __init__(self, np.ndarray[np.float64_t, ndim=2] X, np.ndarray[np.float64_t, ndim=1] Y_train, \
            int n_train, bytes kernel_type, np.ndarray[np.float64_t, ndim=2] hypers_max, int has_noise,\
            bytes ll_type, bytes link_f_type, int should_print, bytes hypers_prior_type, int plot,\
            double y_min_lim=-2.0, double y_max_lim=2.0, double yp_min_lim=-4.0, double yp_max_lim=4.0,\
            object mean_f_lambda=None, infer_change_pts=0):

        assert hypers_prior_type in ("sigmoid", "log"), "Supported priors over hyper-parameters are sigmoid-Gaussian and log-Gaussian"
        self.hypers_prior_type = hypers_prior_type

        # Unique and sort the inputs in each dimension to ease computation
        self.X_ind = np.asfortranarray(np.empty_like(X, dtype=c_ulong))
        _uniq_sorted_inputs = [None]*X.shape[1]
        _max_n_uniq = 0

        self.domain_limits = np.zeros((2, X.shape[1]))
        for j in xrange(X.shape[1]):
            # Sorted unique inputs in the j-th dimension
            _tmp = sorted(list(set(X[:,j])), reverse=False)
            _uniq_sorted_inputs[j] = _tmp

            # Put in a value-index dict to achieve O(1) search complexity
            _idx_dict = {_tmp[i]:i for i in range(len(_tmp))}

            # Index of original inputs in uniqued inputs.
            self.X_ind[:, j] = [_idx_dict[X[i, j]] for i in xrange(X.shape[0])]

            # Maximum number of unique inputs across dimenisons
            _max_n_uniq = max(_max_n_uniq, len(_tmp))

            # Set min/max inputs
            self.domain_limits[0, j] = np.min(_tmp)
            self.domain_limits[1, j] = np.max(_tmp)

        # Construct a compressed array of uniqued inputs
        self.X = np.asfortranarray(np.empty((_max_n_uniq, X.shape[1])))
        self.is_nan = np.asfortranarray(np.empty((_max_n_uniq, X.shape[1]), dtype=bool))

        for j in xrange(X.shape[1]):
            _n = len(_uniq_sorted_inputs[j]) # Real number of uniqued entries
            self.X[:_n, j] = _uniq_sorted_inputs[j]
            self.is_nan[:_n, j] = False
            # The number of unique items varies across dimensions. 
            #   Empty slots are set to nan.
            self.X[_n:, j] = np.nan 
            self.is_nan[_n:, j] = True

        self.Y_train = np.ascontiguousarray(Y_train)
        self.n_train = n_train
        self.f = np.ascontiguousarray(np.empty((X.shape[0])))
        self.post_mean_f = np.zeros((self.f.shape[0]))
        self.mean_f_lambda = mean_f_lambda
        self.kernel_type = kernel_type

        # Kernel configuration setup
        self.change_pts = [[]]*self.X.shape[1]
        self.change_pts_samples = [[]]*self.X.shape[1]
        self.X_config_idx = np.zeros((self.X.shape[0], self.X.shape[1]), dtype=int)
        # Mean and variance of the independent Gamma priors placed on the intensity of
        #   the homogeneous point process priors on synthetic boundary times.
        mean_intensity = np.zeros((self.X.shape[1]))
        for j in xrange(X.shape[1]):
            mean_intensity[j] = 0.05*len(_uniq_sorted_inputs[j])
            #print 'Setting the number of change-points in dimension', j, 'to', '%.2f'%(mean_intensity[j])

        var_intensity = 50.0*mean_intensity
        # Parameters a and b of the Gamma above.
        self.change_pts_b = mean_intensity/var_intensity
        self.change_pts_a = mean_intensity*self.change_pts_b

        self.change_pts_lambda = mean_intensity/(self.domain_limits[1,:]-self.domain_limits[0,:])
        self.infer_change_pts = infer_change_pts
        self.n_change_pts_samples = []

        self.hypers_max = hypers_max
        self.wt_hypers = list(np.zeros((self.hypers_max.shape[1], self.hypers_max.shape[0])))

        self.has_noise = has_noise
        assert ll_type in ("logit", "gaussian", "spt"), "Supported noise models are gaussian and logit"
        self.ll_type = ll_type
        self.log_lik = -999999999.0

        if self.has_noise:
            self.noise_var = 0.05*np.var(self.Y_train)
        else:
            self.noise_var = 0.0

        assert link_f_type in ("prod", "sum", "sum_prod"), "link_f_type should be prod or sum"
        self.link_f_type = link_f_type
        self.should_print = should_print 

        self.dsgp = np.ascontiguousarray(np.zeros((self.X.shape[0], self.X.shape[1], 2)))
        self.post_mean_dsgp = self.dsgp.copy()
        self.dsgp_samples = []
        self.wt_dsgp = np.ascontiguousarray(self.dsgp.copy())
        self.l_hypers = np.asfortranarray(np.empty((self.hypers_max.shape[0], self.X.shape[0], self.X.shape[1])))

        for i in range(self.X.shape[1]):
            self.l_hypers[:,:,i] = np.repeat(0.5*self.hypers_max[:,i].reshape(1, self.hypers_max.shape[0]), self.X.shape[0], axis=0).T

        self.l_factors = np.ascontiguousarray(np.empty((2, 2*self.X.shape[0], self.X.shape[1])))
        self.m_factors = np.ascontiguousarray(np.empty((2, 2*self.X.shape[0], self.X.shape[1])))
        self.l_factors, self.m_factors = self.compute_factors(self.l_hypers)

        self.t_tot = 0.0
        self.t_dsgp = 0.0
        self.t_hypers = 0.0
        self.t_change_pts = 0.0
        self.t_sigma = 0.0
        self.t_pc = []

        try:
            from StringGPu import GPUFactorsWorker
            self.stringgpu_worker = GPUFactorsWorker(self.X, self.hypers_max.shape[0], kernel_type, 64, 256)
            self.use_GPU = True
        except Exception as details:
            print 'Cannot import StringGPu'
            print details
            self.use_GPU = False

        self.plot = plot
        if self.plot:
            self.fig, self.axes = plt.subplots(3, self.X.shape[1], sharex='col', sharey='row')

            try:
                self.axes[0][0]
            except:
                self.axes = self.axes.reshape((3, 1))

            self.lines = [None]*(3*self.X.shape[1])

            for j in xrange(self.X.shape[1]):
                self.lines[3*j], = self.axes[0][j].plot([], [], 'b-')
                self.axes[0][j].set_title(r'$z_{}$'.format('{'+str(j)+'}'))
                self.axes[0][j].grid(True)

                self.lines[3*j+1], = self.axes[1][j].plot([], [], 'b-')
                self.axes[1][j].set_title(r'$z_{}^\prime$'.format('{'+str(j)+'}'))
                self.axes[1][j].grid(True)

                self.lines[3*j+2], = self.axes[2][j].plot([], [], 'r*')
                self.axes[2][j].set_title('Change-points ' + str(j) + '-th dimension')
                self.axes[2][j].grid(True)

                self.axes[0][j].set_xlim(self.X[:,j][np.logical_not(self.is_nan[:, j])].min(), self.X[:,j][np.logical_not(self.is_nan[:, j])].max())

            self.axes[0][0].set_ylim(y_min_lim, y_max_lim)
            self.axes[1][0].set_ylim(yp_min_lim, yp_max_lim)

        self.print_period = 100

        # We will compute the effective sample size using the autocorrelations of f(x_i) for 50
        #   inputs selected uniformly at random, and average them out.
        self.ess_idx = random.sample(range(self.f.shape[0]), 50)
        self.ess_f = np.empty((2000, 50))

        self.is_post_burn_in = 0

    '''
    For each univariate string GP, the conditional distribution of (z_{t_k}, z_{t_k}^\prime) | (z_{t_{k-1}}, z_{t_{k-1}}^\prime)
    is Gaussian with mean M_{t_k}(z_{t_{k-1}}, z_{t_{k-1}}^\prime)^T with M_{t_k} = C_{t_k,t_{k-1}}C_{t_{k-1}, t_{k-1}}^{-1}
    and covariance matrix C_{t_k|t_{k-1}} = C_{t_k, t_{k-1}}C_{t_{k-1},t_{k-1}}^{-1}C_{t_k, t_{k-1}}^T.

    Let C_{t_k|t_{k-1}} = U_{t_k|t_{k-1}} D_{t_k|t_{k-1}} U_{t_k|t_{k-1}}^T be the SVD decomposition of C_{t_k|t_{k-1}}, 
    and L_{t_k|t_{k-1}} = U_{t_k|t_{k-1}} D_{t_k|t_{k-1}}^{1/2}.

    This function computes the factors M_{t_k} and L_{t_k|t_{k-1}} at all inputs points across all dimensions in parallel.

    (Note: for first times t_0, M_{t_0}:= 0 and C_{t_0|t_{-1}}) := C_{t_0})
    '''
    cdef list compute_factors(self, np.ndarray[np.float64_t, ndim=3, mode='fortran'] l_hypers):
        cdef np.ndarray[np.float64_t, ndim=3, mode='c'] _l_factors = np.empty_like(self.l_factors)
        cdef np.ndarray[np.float64_t, ndim=3, mode='c'] _m_factors = np.empty_like(self.m_factors)
        cdef np.ndarray[np.float64_t, ndim=2, mode='fortran'] _X = self.X
        cdef np.ndarray[np.uint8_t, cast=True, ndim=2, mode='fortran'] _is_nan = self.is_nan

        cdef int i, j, l
        cdef int d = self.l_factors.shape[2]
        cdef int n = self.X.shape[0]
        cdef int n_theta = self.hypers_max.shape[0]
        cdef char* k_type = self.kernel_type
        cdef int n_mixt = c_n_spectral_comp(n_theta, k_type)

        # if self.use_GPU:
        #     try:
        #         return self.stringgpu_worker.compute_factors(l_hypers)
        #     except Exception as details:
        #         print details
        #         print 'Disabling StringGPu'
        #         self.use_GPU = False
        # else:
        
        # Process each dimension in parallel
        openmp.omp_set_dynamic(0)
        with nogil, parallel():
            for l in prange(d*n, schedule='static'):
                j = l/n
                i = l-n*j
                # No need to compute factors here as there is no corresponding input
                if _is_nan[i,j]:
                    continue
                else:
                    c_factors(i, j, n, d, &_l_factors[0,0,0], &_m_factors[0,0,0], &l_hypers[0,0,0], &_X[0,0], n_theta, k_type)
        
        return [_l_factors, _m_factors]

    '''
    Same as compute_factors except that 
        i) l_factors and m_factors are updated in-place
        j) l_factors and m_factors are updated for a single input dimension j and a list of data indices i_s.
    '''
    cdef void compute_factors_j(self, np.ndarray[np.float64_t, ndim=3, mode='fortran'] l_hypers,\
        np.ndarray[np.float64_t, ndim=3, mode='c'] l_factors,\
        np.ndarray[np.float64_t, ndim=3, mode='c'] m_factors,\
        int j, np.ndarray[int, ndim=1] i_s):

        cdef np.ndarray[np.float64_t, ndim=2, mode='fortran'] _X = self.X
        cdef np.ndarray[np.uint8_t, cast=True, ndim=2, mode='fortran'] _is_nan = self.is_nan
        cdef np.ndarray[int, ndim=1] _i_s = i_s.copy()

        cdef int i, l
        cdef int n_is = len(i_s)
        cdef int d = self.l_factors.shape[2]
        cdef int n = self.X.shape[0]
        cdef int n_theta = self.hypers_max.shape[0]
        cdef char* k_type = self.kernel_type
        cdef int n_mixt = c_n_spectral_comp(n_theta, k_type)

        # Process each dimension in parallel
        # openmp.omp_set_dynamic(0)
        # with nogil, parallel():
        for l in xrange(n_is):#prange(n_is, schedule='static'):
            i = _i_s[l]
            # No need to compute factors here as there is no corresponding input
            if _is_nan[i, j]:
                continue
            else:
                c_factors(i, j, n, d, &l_factors[0,0,0], &m_factors[0,0,0], &l_hypers[0,0,0], &_X[0,0], n_theta, k_type)
        
        return


    '''
    Inference is made on whitened derivative string GP values. More precisely, if Z ~ N(M, C) is a multivariate
        Gaussian, we define as whitened GP a vector X satisfying Z = M + LX, where C = LL^T. It is easy to see that
        if X ~ N(0, I), then M + LX ~ N(M, C).

        Making inference on X rather than Z has two main advantages:
        1) It is more robust to ill-conditioning. In effect there is no need to compute a determinant or invert a matrix.
            Letting C = UDU^T be the SVD decomposition of C, a good candidate for L is L=UD^{1/2}. Thus inference on the
            posterior over Z requires no-more than SVD decompositions.

        2) It speeds up mixing. This is because kernel hyper-parameters values affect the values of the latent functions
            through L. Thus, hyper-parameters sampling directly contribute towards improving the model fit. This wouldn't
            have been the case when sampling in the 'colored' space: sampling Z would contribute towards improving model
            fit, while sampling hyper-parameters would contribute towards determining a better fit for the last sample of Z.
            If a sample for Z is far away from the posterior model, as will happen when the Markov chain is run for long enough,
            the next sample of hyper-parameters will consolidate the 'outlier-sample', making it harder for subsequent Z samples
            to revert to region of high posterior mass.
    '''
    cdef np.ndarray[np.float64_t, ndim=3] compute_dsgp(self, np.ndarray[np.float64_t, ndim=3] wt_dsgp,\
        np.ndarray[np.float64_t, ndim=3] l_factors, np.ndarray[np.float64_t, ndim=3] m_factors):

        cdef np.ndarray[np.float64_t, ndim=3] _wt_dsgp = wt_dsgp
        cdef np.ndarray[np.float64_t, ndim=3] _l_factors = l_factors
        cdef np.ndarray[np.float64_t, ndim=3] _m_factors = m_factors
        cdef np.ndarray[np.float64_t, ndim=3] dsgp = np.empty_like(self.dsgp)
        cdef np.ndarray[np.uint8_t, cast=True, ndim=2, mode='fortran'] _is_nan = self.is_nan

        cdef int i
        cdef int prev_nondup_i
        cdef int j
        cdef int d = self.l_factors.shape[2]
        cdef int n = self.X.shape[0]

        # Process each input dimension in parallel
        openmp.omp_set_dynamic(1)
        with nogil, parallel():
            for j in prange(d, schedule='static'):
                for i in xrange(n):
                    if _is_nan[i, j]:
                        # These values aren't used as they do not correspond to real inputs
                        # They are an artefact of the number of uniques inputs per dimension being different.
                        dsgp[i,j,0] = 0.0
                        dsgp[i,j,1] = 0.0
                        continue

                    dsgp[i,j,0] = _l_factors[0,2*i,j]*_wt_dsgp[i,j,0] + _l_factors[0,2*i+1,j]*_wt_dsgp[i,j,1] 
                    dsgp[i,j,1] = _l_factors[1,2*i,j]*_wt_dsgp[i,j,0] + _l_factors[1,2*i+1,j]*_wt_dsgp[i,j,1]
                    if i > 0:
                        dsgp[i,j,0] = _m_factors[0,2*i,j]*dsgp[prev_nondup_i,j,0] + _m_factors[0,2*i+1,j]*dsgp[prev_nondup_i,j,1]
                        dsgp[i,j,1] = _m_factors[1,2*i,j]*dsgp[prev_nondup_i,j,0] + _m_factors[1,2*i+1,j]*dsgp[prev_nondup_i,j,1]

                    prev_nondup_i = i
        return dsgp

    '''
    Same as compute_dsgp_j except that updates are performed in-place and only along the j-th input dimension.
    '''
    cdef void compute_dsgp_j(self, np.ndarray[np.float64_t, ndim=3] wt_dsgp,\
        np.ndarray[np.float64_t, ndim=3] l_factors, np.ndarray[np.float64_t, ndim=3] m_factors,\
        np.ndarray[np.float64_t, ndim=3] dsgp, int j):

        cdef np.ndarray[np.float64_t, ndim=3] _wt_dsgp = wt_dsgp
        cdef np.ndarray[np.float64_t, ndim=3] _l_factors = l_factors
        cdef np.ndarray[np.float64_t, ndim=3] _m_factors = m_factors
        cdef np.ndarray[np.uint8_t, cast=True, ndim=2, mode='fortran'] _is_nan = self.is_nan

        cdef int i
        cdef int prev_nondup_i
        cdef int d = self.l_factors.shape[2]
        cdef int n = self.X.shape[0]

        for i in xrange(n):
            # dsgp is updated in-place (passed by reference)
            if _is_nan[i, j]:
                # These values aren't used as they do not correspond to real inputs
                # They are an artefact of the number of uniques inputs per dimension being different.
                dsgp[i,j,0] = 0.0
                dsgp[i,j,1] = 0.0
                continue

            dsgp[i,j,0] = _l_factors[0,2*i,j]*_wt_dsgp[i,j,0] + _l_factors[0,2*i+1,j]*_wt_dsgp[i,j,1] 
            dsgp[i,j,1] = _l_factors[1,2*i,j]*_wt_dsgp[i,j,0] + _l_factors[1,2*i+1,j]*_wt_dsgp[i,j,1]
            if i > 0:
                dsgp[i,j,0] = _m_factors[0,2*i,j]*dsgp[prev_nondup_i,j,0] + _m_factors[0,2*i+1,j]*dsgp[prev_nondup_i,j,1]
                dsgp[i,j,1] = _m_factors[1,2*i,j]*dsgp[prev_nondup_i,j,0] + _m_factors[1,2*i+1,j]*dsgp[prev_nondup_i,j,1]

            prev_nondup_i = i
        return


    '''
    self.X contains inputs sorted in each dimension. Inference is made on ndarrays whose values 
        correspond to these sorted inputs (e.g. self.dsgp). This function re-orders an ndarray
        of string GP values at sorted inputs, so as to match the original inputs order.

        This allows us to apply the string GP link function rowise to recover latent function values
        at input points.
    '''
    cdef np.ndarray[np.float64_t, ndim=2] _reorder_x(self, np.ndarray[np.float64_t, ndim=2] sgp):
        cdef np.ndarray[np.float64_t, ndim=2, mode='fortran'] inp = np.asfortranarray(sgp)
        cdef np.ndarray[unsigned long, ndim=2, mode='fortran'] ind = self.X_ind
        cdef np.ndarray[np.float64_t, ndim=2, mode='fortran'] out = np.empty(\
            (ind.shape[0], ind.shape[1]), order='F')
        cdef unsigned long n = ind.shape[0]

        cdef int j
        cdef int d = inp.shape[1]
        openmp.omp_set_dynamic(1)
        with nogil, parallel():
            for j in prange(d, schedule='static'):
                c_shuffle(&inp[0, j], &out[0, j], &ind[0, j], n)
        return np.ascontiguousarray(out)

    '''
    Recover the vector of function values at training inputs from values of the corresponding multivariate
    derivative string GP.
    '''
    cdef np.ndarray[np.float64_t, ndim=1] compute_f(self, np.ndarray[np.float64_t, ndim=3] dsgp):
        cdef np.ndarray[np.float64_t, ndim=2] sgp
        sgp = self._reorder_x(dsgp[:,:,0])

        if self.link_f_type == "prod":
            return np.prod(sgp, axis=1)

        if self.link_f_type == "sum":
            return np.sum(sgp, axis=1)

        if self.link_f_type == "sum_prod":
            return 0.5*(np.sum(sgp, axis=1) + np.prod(sgp, axis=1))


    '''
    Compute the log-likelihood under the observation model.
        Should be implemented by the derived class.
    '''
    def compute_log_lik(self, np.ndarray[np.float64_t, ndim=1] f, np.ndarray[np.float64_t, ndim=3] dsgp):
        return NotImplemented


    '''
        Sample the univariate derivative string GPs jointly using Elliptical Slice Sampling.
    '''
    def sample_dsgp(self, tolerance=0.001):
        # Sample the std scaling factor of the new proposal
        cdef np.ndarray[np.float64_t, ndim=1] dsgp_in_use = self.wt_dsgp[:,:,0][np.logical_not(self.is_nan)]
        dsgp_in_use = np.concatenate([dsgp_in_use, self.wt_dsgp[:,:,1][np.logical_not(self.is_nan)]])

        cdef double s = 2.0

        # Adjust the old log likelihood accordingly
        cdef double _log_lik_old = self.log_lik - 0.5*np.dot(dsgp_in_use, dsgp_in_use)*(1.0-1.0/s)

        # Perform ESS, scaling the proposal whitened dsgp by s
        # cdef double _log_lik_old = self.log_lik
        # cdef double s = 1.0

        cdef double u
        _u = np.random.uniform(0.0, 1.0)
        _log_lik_old += np.log(_u)

        cdef double _a
        _a = np.random.uniform(0.0, 2.0*np.pi)

        cdef double _a_min
        _a_min = _a - 2.0*np.pi

        cdef double _a_max
        _a_max = _a
        
        cdef np.ndarray[np.float64_t, ndim=3] wt_dsgp_new = np.empty_like(self.wt_dsgp)
        # Adjust the variance of the proposal whitened dsgp by the generalized ESS scaling variance
        if self.use_GPU:
            wt_dsgp_new = np.sqrt(s)*self.stringgpu_worker.randn((self.wt_dsgp.shape[0], self.wt_dsgp.shape[1], self.wt_dsgp.shape[2]))           
        else:
            wt_dsgp_new = np.sqrt(s)*np.random.randn(self.wt_dsgp.shape[0], self.wt_dsgp.shape[1], self.wt_dsgp.shape[2])

        cdef np.ndarray[np.float64_t, ndim=3] _dsgp
        cdef np.ndarray[np.float64_t, ndim=3] _wt_dsgp = wt_dsgp_new.copy()
        cdef np.ndarray[np.float64_t, ndim=1] _dsgp_in_use
        cdef np.ndarray[np.float64_t, ndim=1] _f
        cdef double _log_lik_new

        while np.abs((_a_max-_a_min)) > tolerance:
            _wt_dsgp = np.cos(_a)*self.wt_dsgp + np.sin(_a)*wt_dsgp_new
            _dsgp = self.compute_dsgp(_wt_dsgp, self.l_factors, self.m_factors)

            _dsgp_in_use = _wt_dsgp[:,:,0][np.logical_not(self.is_nan)]
            _dsgp_in_use = np.concatenate([_dsgp_in_use, _wt_dsgp[:,:,1][np.logical_not(self.is_nan)]])

            _f = self.compute_f(_dsgp)
            _log_lik_new = self.compute_log_lik(_f, _dsgp) - 0.5*np.dot(_dsgp_in_use, _dsgp_in_use)*(1.0-1.0/s)

            if _log_lik_new > _log_lik_old:
                break
            else:
                _a = np.random.uniform(_a_min, _a_max)
                if _a < 0.0:
                    _a_min = _a
                else:
                    _a_max = _a

        self.wt_dsgp = _wt_dsgp.copy()
        self.dsgp = _dsgp.copy()
        self.log_lik = _log_lik_new + 0.5*np.dot(_dsgp_in_use, _dsgp_in_use)*(1.0-1.0/s)
        self.f = _f.copy()
        return


    '''
    Compute kernels hyper-parameters from their whitened values.
    '''
    cdef np.ndarray[np.float64_t, ndim=3, mode='fortran'] compute_l_hypers(self, list wt_hypers):
        cdef np.ndarray[np.float64_t, ndim=3, mode='fortran'] l_hypers
        l_hypers = np.empty_like(self.l_hypers, order='F')
        cdef np.ndarray[np.float64_t, ndim=2] hypers_mat_j
        cdef np.ndarray[np.float64_t, ndim=1] hypers_j
        cdef np.ndarray[np.float64_t, ndim=1] hypers_scale_j

        cdef int j, n_hypers_j, kj
        for j in range(self.X.shape[1]):
            # len(self.hypers_max[:, j]) is the number of kernel parameters for one string in the j-th dimension
            #   but hypers[j] contains kernel parameters for all strings in the j-th dimension in increasing order
            #   of kernel index.
        
            n_hypers_j = self.hypers_max[:, j].shape[0]  # Number of hyper-parameters per kernel conf. in the j-th dim.
            kj = wt_hypers[j].shape[0]/n_hypers_j  # Number of different kernel configurations in the j-th dim.
            hypers_scale_j = np.tile(self.hypers_max[:, j], kj)

            if self.hypers_prior_type == "sigmoid":
                hypers_j = hypers_scale_j/(1.0+np.exp(np.maximum(np.minimum(wt_hypers[j], 100.0), -100.0)))
            else:
                # Log-Gaussian with mode hypers_max/2
                hypers_j = 0.5*hypers_scale_j*np.exp(np.maximum(np.minimum(wt_hypers[j], 100.0), -100.0))

            # hypers_mat_j is of the form 
            # | <Vector of hyper-parameters of the kernel configuration 1 in the j-th dimension>  |
            # | <Vector of hyper-parameters of the kernel configuration 2 in the j-th dimension>  |
            # |                                     ...                                           |
            # | <Vector of hyper-parameters of the kernel configuration kj in the j-th dimension> |
            hypers_mat_j = hypers_j.reshape((kj, n_hypers_j))

            # l_hypers[:, :, j] is that matrix whose i-th column corresponds to the vector of 
            #   hyper-parameters of the kernel configuration driving the i-th input in the j-th dimension.
            try:
                l_hypers[:, :, j] = (hypers_mat_j[self.X_config_idx[:, j]].copy()).T
            except:
                print 'Mismatch between kernel configuration and X_config_idx'
                print 'X_config min/max', self.X_config_idx[:, j].min(), self.X_config_idx[:, j].max()
                print 'Length hypers_mat_j', len(hypers_mat_j)

        return l_hypers


    '''
        Jointly sample the hyper-parameters
    '''
    def sample_hypers(self, tolerance=0.001):
        cdef double _log_lik_old
        _log_lik_old = self.log_lik

        cdef double u
        _u = np.random.uniform(0.0, 1.0)
        _log_lik_old += np.log(_u)

        cdef double _a
        _a = np.random.uniform(0.0, 2.0*np.pi)

        cdef double _a_min
        _a_min = _a - 2.0*np.pi

        cdef double _a_max
        _a_max = _a
        
        cdef list  wt_hypers_new = [None]*len(self.wt_hypers)
        cdef list _wt_hypers = [None]*len(self.wt_hypers)
        cdef int j
        for j in xrange(len(self.wt_hypers)):
            wt_hypers_new[j] =  np.random.randn(self.wt_hypers[j].shape[0])

        cdef np.ndarray[np.float64_t, ndim=3, mode='fortran'] _l_hypers
        cdef np.ndarray[np.float64_t, ndim=3] _dsgp
        cdef np.ndarray[np.float64_t, ndim=3] _l_factors
        cdef np.ndarray[np.float64_t, ndim=3] _m_factors
        cdef np.ndarray[np.float64_t, ndim=1] _f
        cdef double _log_lik_new

        while np.abs((_a_max-_a_min)) > tolerance:
            _wt_hypers = [None]*len(self.wt_hypers)
            for j in xrange(len(self.wt_hypers)):
                _wt_hypers[j] = np.cos(_a)*self.wt_hypers[j] + np.sin(_a)*wt_hypers_new[j]

            _l_hypers = self.compute_l_hypers(_wt_hypers)
            _l_factors, _m_factors = self.compute_factors(_l_hypers)
            _dsgp = self.compute_dsgp(self.wt_dsgp, _l_factors, _m_factors)
            _f = self.compute_f(_dsgp)
            _log_lik_new = self.compute_log_lik(_f, _dsgp)
                
            if (_log_lik_new > _log_lik_old):
                break
            else:
                _a = np.random.uniform(_a_min, _a_max)
                if _a < 0.0:
                    _a_min = _a
                else:
                    _a_max = _a

        if (not np.isnan(_log_lik_new)) and (not np.isinf(_log_lik_new)):
            self.wt_hypers = _wt_hypers
            self.l_factors = _l_factors.copy()
            self.m_factors = _m_factors.copy()
            self.dsgp = _dsgp.copy()
            self.log_lik = _log_lik_new
            self.l_hypers = _l_hypers.copy()
            self.f = _f.copy()

        return

    '''
    Sample change-points guiding synthetic kernel configurations in each dimension.
    '''
    def sample_change_points(self):
        cdef np.ndarray[np.float64_t, ndim=3, mode='fortran'] _l_hypers
        cdef np.ndarray[np.float64_t, ndim=3] _dsgp
        cdef np.ndarray[np.float64_t, ndim=3] _l_factors
        cdef np.ndarray[np.float64_t, ndim=3] _m_factors
        cdef np.ndarray[np.float64_t, ndim=1] _f

        for j in xrange(self.X.shape[1]):
            # Decide whether to consider adding or removing a change point.
            nj = len(self.change_pts[j])
            should_add = False
            if nj == 0:
                # There is no change-point in the j-th dimension:
                #    choose between adding a new change-point and doing 
                #    nothing each with probability 1/2.

                if np.random.uniform() < 0.5:
                    continue

                should_add = True
            else:
                # There is at least a chane-point in the j-th dimension:
                #    We choose between doing nothing, adding a new change-point
                #    and deleting a change-point, each with probability 1/3.
                
                if np.random.uniform() < 1.0/3.0:
                    continue

                should_add = np.random.uniform() < 0.5

            # At this point our proposal is to either add a new change-point or to delete 
            #   an existing one.
            n_hypers_j = self.hypers_max[:, j].shape[0]
            hypers_scale = self.hypers_max[:, j]

            if should_add:
                # Sample a candidate new change-point uniformly at random on the j-th domain
                c_new = np.random.uniform(self.domain_limits[0, j], self.domain_limits[1, j])

                # Find out the index of the kernel configuration (synthetic string) that this new
                #   change-point would break.
                if (len(self.change_pts[j]) == 0) or (c_new < self.change_pts[j][0]):
                    n_insert = 0
                elif c_new > self.change_pts[j][nj-1]:
                    n_insert = nj
                else:
                    for i in range(1, nj):
                        if c_new < self.change_pts[j][i] and c_new > self.change_pts[j][i-1]:
                            n_insert = i
                            break

                # n_insert is the position at which the new change point should be inserted.

                # Construct the proposal hyper-parameters for the left and right hand-side of the
                #   split synthetic string.
                current_wt_hypers = self.wt_hypers[j][n_hypers_j*n_insert:n_hypers_j*(n_insert+1)].copy()
                new_wt_hypers = np.random.randn(len(current_wt_hypers))
                proposal_wt_hypers_left = (1.0/np.sqrt(2.0))*(current_wt_hypers - new_wt_hypers)
                proposal_wt_hypers_right = (1.0/np.sqrt(2.0))*(current_wt_hypers + new_wt_hypers)

                # Adjust state variables accordingly.
                _change_pts_j = np.sort(list(self.change_pts[j]) + [c_new])
                _change_pts = list(self.change_pts)
                _change_pts[j] = _change_pts_j.copy()

                _wt_hypers_j = np.concatenate([self.wt_hypers[j][:n_hypers_j*n_insert].copy(),\
                    proposal_wt_hypers_left,\
                    proposal_wt_hypers_right,\
                    self.wt_hypers[j][n_hypers_j*(n_insert+1):].copy()])
                _wt_hypers = list(self.wt_hypers)
                _wt_hypers[j] = _wt_hypers_j.copy()

                if self.hypers_prior_type == "sigmoid":
                    proposal_hypers_left = hypers_scale/(1.0+np.exp(np.maximum(np.minimum(proposal_wt_hypers_left, 100.0), -100.0)))
                    proposal_hypers_right = hypers_scale/(1.0+np.exp(np.maximum(np.minimum(proposal_wt_hypers_right, 100.0), -100.0)))
                else:
                    # Log-Gaussian with mode hypers_max/2
                    proposal_hypers_left = 0.5*hypers_scale*np.exp(np.maximum(np.minimum(proposal_wt_hypers_left, 100.0), -100.0))
                    proposal_hypers_right = 0.5*hypers_scale*np.exp(np.maximum(np.minimum(proposal_wt_hypers_right, 100.0), -100.0))

                # Increment kernel configuration indices of input on the right of the new change-point.
                _X_config_idx = self.X_config_idx.copy()
                _selector = np.logical_and(np.logical_not(self.is_nan[:, j]), self.X[:, j] > c_new) # Inputs whose config idx should be updated.
                _X_config_idx[:, j][_selector] = 1 + _X_config_idx[:, j][_selector]

                # Update kernel configuration hyper-parameters.
                _l_hypers = np.asfortranarray(self.l_hypers.copy())
                _p = np.min(_X_config_idx[:, j][_selector])
                _p_1 = np.max(_X_config_idx[:, j][np.logical_not(_selector)])

                if np.abs(_p-_p_1) > 1.0:
                    # This would only happen if either ]c_{p-1}, c_new[ or ]c_new, c_p[ does not contain any
                    #   training or test data, in which case the proposal split is of no use.
                    continue

                # Indices of inputs that belong the synthetic string immediately to the left of c_new.
                _hypers_left_idx = np.arange(_X_config_idx.shape[0], dtype=int)[_X_config_idx[:, j]==_p_1]

                # Update their kernel configurations accordingly.
                _l_hypers[:, _hypers_left_idx, j] = np.repeat(proposal_hypers_left.reshape(1, proposal_hypers_left.shape[0]),\
                    len(_hypers_left_idx), axis=0).T

                # Indices of inputs that belong the synthetic string immediately to the right of c_new.
                _hypers_right_idx = np.arange(_X_config_idx.shape[0], dtype=int)[_X_config_idx[:, j]==_p]

                # Update their kernel configurations accordingly.
                _l_hypers[:, _hypers_right_idx, j] = np.repeat(proposal_hypers_right.reshape(1, proposal_hypers_right.shape[0]),\
                    len(_hypers_right_idx), axis=0).T

                # Update l_factors and m_factors along the j-th dimension in-place.
                _l_factors = self.l_factors.copy()
                _m_factors = self.m_factors.copy()
                # Indices of inputs whose kernel configurations would change if the proposal is accepted.
                i_s = np.concatenate([_hypers_left_idx, _hypers_right_idx])
                # Update the factors in-place as needed.
                self.compute_factors_j(_l_hypers, _l_factors, _m_factors, j, np.array(i_s).astype(np.intc))

                # Update _dsgp along the j-th dimension in-place.
                _dsgp = self.dsgp.copy()
                self.compute_dsgp_j(self.wt_dsgp, _l_factors, _m_factors, _dsgp, j)

                # Compute latent function values
                _f = np.empty_like(self.f)
                _f = self.compute_f(_dsgp)

                # Compute the new observations log-likelihood
                _log_lik = self.compute_log_lik(_f, _dsgp)

                # Randomly accept/reject the proposal.
                lr = 0.0
                lr += _log_lik - self.log_lik
                lr += np.log(self.change_pts_lambda[j]) + np.log(self.domain_limits[1, j]-self.domain_limits[0, j])
                lr -= np.log(1.0+len(self.change_pts[j]))
                lr += -0.5*np.log(2.0*np.pi)*len(proposal_wt_hypers_left) - 0.5*np.dot(proposal_wt_hypers_left, proposal_wt_hypers_left)
                lr += -0.5*np.log(2.0*np.pi)*len(proposal_wt_hypers_right) - 0.5*np.dot(proposal_wt_hypers_right, proposal_wt_hypers_right)
                lr -= -0.5*np.log(2.0*np.pi)*len(current_wt_hypers) - 0.5*np.dot(current_wt_hypers, current_wt_hypers)
                lr -= -0.5*np.log(2.0*np.pi)*len(new_wt_hypers) - 0.5*np.dot(new_wt_hypers, new_wt_hypers)
                lr = np.min([lr, 0.0])


                if np.random.uniform() <= np.exp(lr):
                    # Accepting the proposal to add the change-point c_new in dimension j.
                    self.f = _f.copy()
                    self.dsgp = _dsgp.copy()
                    self.l_factors = _l_factors.copy()
                    self.m_factors = _m_factors.copy()
                    self.l_hypers = _l_hypers.copy()
                    self.wt_hypers = list(_wt_hypers)
                    self.X_config_idx = _X_config_idx.copy()
                    self.change_pts = list(_change_pts)
                    self.log_lik = _log_lik

            else:
                # Sample a candidate change-point to delete uniformly at random between change-points
                #   in the j-th dimension
                p = np.random.randint(0, nj)
                cp = np.array(self.change_pts[j])[p]

                # Adjust state variables accordingly.
                _change_pts_j = np.delete(np.array(self.change_pts[j]), p)
                _change_pts = list(self.change_pts)
                _change_pts[j] = _change_pts_j

                # The p-th change-point corresponds to the (p+1)-th kernel configuration.
                q = p + 1           

                wt_hypers_left = self.wt_hypers[j][n_hypers_j*(q-1):n_hypers_j*q]
                wt_hypers_right = self.wt_hypers[j][n_hypers_j*q:n_hypers_j*(q+1)]
                proposal_wt_hypers = (1.0/np.sqrt(2.0))*(wt_hypers_left + wt_hypers_right)
                proposal_new_wt_hypers = (1.0/np.sqrt(2.0))*(-wt_hypers_left + wt_hypers_right)

                _wt_hypers_j = np.concatenate([self.wt_hypers[j][:n_hypers_j*(q-1)].copy(),\
                    proposal_wt_hypers,\
                    self.wt_hypers[j][n_hypers_j*(q+1):].copy()])

                _wt_hypers = list(self.wt_hypers)
                _wt_hypers[j] = _wt_hypers_j.copy()

                if self.hypers_prior_type == "sigmoid":
                    proposal_hypers = hypers_scale/(1.0+np.exp(np.maximum(np.minimum(proposal_wt_hypers, 100.0), -100.0)))
                else:
                    # Log-Gaussian with mode hypers_max/2
                    proposal_hypers = 0.5*hypers_scale*np.exp(np.maximum(np.minimum(proposal_wt_hypers, 100.0), -100.0))

                # Decrement kernel configuration indices of input on the right hand-side of the deleted change-point.
                _X_config_idx = self.X_config_idx.copy()
                _selector = (_X_config_idx[:, j] >= q) # Inputs whose config idx should be updated.
                _X_config_idx[:, j][_selector] = _X_config_idx[:, j][_selector]-1

                # Update kernel configuration hyper-parameters.
                #   The inputs whose kernel configurations (hyper-parameters) should change 
                #   if the p-th change-point is deleted are the ones whose current kernel 
                #   configuration is either p or p+1
                _l_hypers = np.asfortranarray(self.l_hypers.copy())
                # Indices of inputs that belong the synthetic string immediately to the left of c_p.
                _hypers_left_idx = np.arange(self.X_config_idx.shape[0], dtype=int)[self.X_config_idx[:, j]==p]
                # Indices of inputs that belong the synthetic string immediately to the right of c_p.
                _hypers_right_idx = np.arange(self.X_config_idx.shape[0], dtype=int)[self.X_config_idx[:, j]==p+1]
                # Merge (Indices of inputs whose kernel configurations would change if the proposal is accepted.)
                _hypers_idx = list(set(list(np.concatenate([_hypers_left_idx, _hypers_right_idx]))))

                # Update their kernel configurations accordingly.
                _l_hypers[:, _hypers_idx, j] = np.repeat(proposal_hypers.reshape(1, proposal_hypers.shape[0]),\
                    len(_hypers_idx), axis=0).T

                # Update l_factors and m_factors along the j-th dimension in-place.
                _l_factors = self.l_factors.copy()
                _m_factors = self.m_factors.copy()

                # Update the factors in-place as needed.
                self.compute_factors_j(_l_hypers, _l_factors, _m_factors, j, np.array(_hypers_idx).astype(np.intc))

                # Update _dsgp along the j-th dimension in-place.
                _dsgp = self.dsgp.copy()
                self.compute_dsgp_j(self.wt_dsgp, _l_factors, _m_factors, _dsgp, j)

                # Compute latent function values
                _f = np.empty_like(self.f)
                _f = self.compute_f(_dsgp)

                # Compute the new observations log-likelihood
                _log_lik = self.compute_log_lik(_f, _dsgp)

                # Randomly accept/reject the proposal.
                lr = 0.0
                lr += _log_lik - self.log_lik
                lr += np.log(len(self.change_pts[j]))
                lr -= np.log(self.change_pts_lambda[j]) + np.log(self.domain_limits[1, j] - self.domain_limits[0, j])
                lr += -0.5*np.log(2.0*np.pi)*len(proposal_wt_hypers) - 0.5*np.dot(proposal_wt_hypers, proposal_wt_hypers)
                lr += -0.5*np.log(2.0*np.pi)*len(proposal_new_wt_hypers) - 0.5*np.dot(proposal_new_wt_hypers, proposal_new_wt_hypers)
                lr -= -0.5*np.log(2.0*np.pi)*len(wt_hypers_left) - 0.5*np.dot(wt_hypers_left, wt_hypers_left)
                lr -= -0.5*np.log(2.0*np.pi)*len(wt_hypers_right) - 0.5*np.dot(wt_hypers_right, wt_hypers_right)
                lr = np.min([lr, 0.0])

                if np.random.uniform() <= np.exp(lr):
                    # Accepting the proposal to delete the change-point c_p^j in dimension j.
                    self.f = _f.copy()
                    self.dsgp = _dsgp.copy()
                    self.l_factors = _l_factors.copy()
                    self.m_factors = _m_factors.copy()
                    self.l_hypers = _l_hypers.copy()
                    self.wt_hypers = list(_wt_hypers)
                    self.X_config_idx = _X_config_idx.copy()
                    self.change_pts = list(_change_pts)
                    self.log_lik = _log_lik

        return

    def sample_change_points_intensities(self):
        '''
        A homogeneous Poisson point process with intensity $\lambda_j$ prior is placed on 
            the set of change-points in the j-th dimension. A (conjuguate) gamma prior 
            with parameters $a_j$ and $b_j$ (i.e. mean $\frac{a_j}{b_j}$) is then placed on 
            $\lambda_j$. 

        This method samples from the posterior over $\lambda_j$ conditional on all other
            state variables, which we recall is also gamma distributed with parameters 
            $a_j + n_j$ and $b_j + 1$ where $n_j$ is the total number of change-points
            in the j-th input dimension.
        '''

        for j in xrange(self.X.shape[1]):
            a_j = self.change_pts_a[j] + len(self.change_pts[j])
            b_j = self.change_pts_b[j] + 1.0
            self.change_pts_lambda[j] = np.random.gamma(a_j, 1.0/b_j)/(self.domain_limits[1, j]-self.domain_limits[0, j])
        return



    def sample_change_points_positions(self):
        '''
        Update the positions of change-points sequentially using Metropolis-Hastings. 
            In the j-th dimension, the proposal to update c_p^j is uniform on
            [c_{p-1}^j, c_{p+1}^j], where c_{p-1}^j can be a^j and c_{p+1}^j
            can be b^j.

            The acceptance probability is simply the ratio of likelihoods.
        '''
        for j in xrange(self.X.shape[1]):
            nj = len(self.change_pts[j])
            if nj == 0:
                continue

            n_hypers_j = self.hypers_max[:, j].shape[0]
            hypers_scale = self.hypers_max[:, j]

            for p in xrange(nj):
                current_change_pts = [self.domain_limits[0, j]] + list(self.change_pts[j])\
                    + [self.domain_limits[1, j]]
                q = p + 1

                _cp = np.random.uniform(current_change_pts[q-1], current_change_pts[q+1])
                assert _cp <= self.domain_limits[1, j] and _cp >= self.domain_limits[0, j],\
                    "New change-point should be on the support of the domain."
                cp = current_change_pts[q] # Current change-point position.

                # New set of changa-points
                _change_pts = list(self.change_pts)
                _change_pts[j][p] = _cp

                # Determine the indices of inputs whose kernel configuration would change if the proposal
                #   was accepted.
                _X_config_idx = self.X_config_idx.copy()

                if _cp > cp:
                    # The kernel configuration of points in ]c_p^j, c_new^j[ should change from 
                    #   p+1 to p.
                    _selector = (_X_config_idx[:, j] == p+1)
                    _selector = np.logical_and(np.logical_not(self.is_nan[:, j]), _selector)
                    _selector = np.logical_and(_selector, self.X[:, j] < _cp)
                    _X_config_idx[:, j][_selector] = p
                    _wt_hypers_to_copy = self.wt_hypers[j][n_hypers_j*p:n_hypers_j*(p+1)]
                else:
                    # The kernel configuration of points in ]c_new^j, c_p^j[ should change from 
                    #   p to p+1.
                    _selector = (_X_config_idx[:, j] == p)
                    _selector = np.logical_and(np.logical_not(self.is_nan[:, j]), _selector)
                    _selector = np.logical_and(_selector, self.X[:, j] > _cp)
                    _X_config_idx[:, j][_selector] = p+1
                    _wt_hypers_to_copy = self.wt_hypers[j][n_hypers_j*(p+1):n_hypers_j*(p+2)]

                # Indices of inputs whose kernel configuration would change on acceptance of the move.
                _hypers_idx = np.arange(self.X_config_idx.shape[0], dtype=int)[_selector]

                # Update kernel configuration hyper-parameters.
                if self.hypers_prior_type == "sigmoid":
                    hypers_to_copy = hypers_scale/(1.0+np.exp(np.maximum(np.minimum(_wt_hypers_to_copy, 100.0), -100.0)))
                else:
                    # Log-Gaussian with mode hypers_max/2
                    hypers_to_copy = 0.5*hypers_scale*np.exp(np.maximum(np.minimum(_wt_hypers_to_copy, 100.0), -100.0))

                _l_hypers = np.asfortranarray(self.l_hypers.copy())
                _l_hypers[:, _hypers_idx, j] = np.repeat(hypers_to_copy.reshape(1, hypers_to_copy.shape[0]),\
                    len(_hypers_idx), axis=0).T

                # Update l_factors and m_factors along the j-th dimension in-place.
                _l_factors = self.l_factors.copy()
                _m_factors = self.m_factors.copy()

                # Update the factors in-place as needed.
                self.compute_factors_j(_l_hypers, _l_factors, _m_factors, j, np.array(_hypers_idx).astype(np.intc))

                # Update _dsgp along the j-th dimension in-place.
                _dsgp = self.dsgp.copy()
                self.compute_dsgp_j(self.wt_dsgp, _l_factors, _m_factors, _dsgp, j)

                # Compute latent function values
                _f = np.empty_like(self.f)
                _f = self.compute_f(_dsgp)

                # Compute the new observations log-likelihood
                _log_lik = self.compute_log_lik(_f, _dsgp)

                # Accept/Reject
                lr = _log_lik - self.log_lik
                lr = np.min([lr, 0.0])

                if np.random.uniform() <= np.exp(lr):
                    # Accept the move to update the p-th change-point in the j-th dimension
                    self.f = _f.copy()
                    self.dsgp = _dsgp.copy()
                    self.l_factors = _l_factors.copy()
                    self.m_factors = _m_factors.copy()
                    self.l_hypers = _l_hypers.copy()
                    self.X_config_idx = _X_config_idx.copy()
                    self.change_pts = list(_change_pts)
                    self.log_lik = _log_lik

        return

    '''
    When required, we put an inverse gamma prior on the variance of the noise
        parameter.

    (Note: Only relevant for regression problems. In this implementation we assume an i.i.d Gaussian noise.)
    '''
    def sample_noise(self):
        if not self.has_noise:
            return 
        # The noise parameter is only used for regression problems.
        #   We use an (conjuguate) inverse-gamma prior with alpha = 1, beta = 1.

        cdef np.ndarray[np.float64_t, ndim=1] errors
        errors = self.f[:self.n_train]-self.Y_train

        cdef double alpha = 1.0
        cdef double beta = 1.0
        alpha += errors.shape[0]/2.0
        beta += np.sum(errors*errors)/2.0

        cdef double noise_var
        noise_var = invgamma.rvs(alpha)
        noise_var *= beta
        if not np.isnan(noise_var):
            self.noise_var = noise_var
        return


    def compute_kernel_confs(self, wt_hypers):
        hypers = [None]*self.X.shape[1]
        for j in range(self.X.shape[1]):        
            n_hypers_j = self.hypers_max[:, j].shape[0]
            kj = wt_hypers[j].shape[0]/n_hypers_j
            hypers_scale_j = np.tile(self.hypers_max[:, j], kj)

            if self.hypers_prior_type == "sigmoid":
                hypers_j = hypers_scale_j/(1.0+np.exp(np.maximum(np.minimum(wt_hypers[j], 100.0), -100.0)))
            else:
                # Log-Gaussian with mode hypers_max/2
                hypers_j = 0.5*hypers_scale_j*np.exp(np.maximum(np.minimum(wt_hypers[j], 100.0), -100.0))
            hypers[j] = hypers_j.reshape((kj, n_hypers_j))

        return hypers


    '''
    Utility function: initializes the animation.
    '''
    def init_plots(self):
        cdef int i
        if self.plot:
            for i in xrange(3*self.X.shape[1]):
                self.lines[i].set_data([], [])
            return self.lines
        else:
            return []

    '''
    Record samples
    '''
    def record(self, i, n):
        # Record samples of dsgp every now and then    
        #   For memory and decorrelation purposes we aim to 
        #   keep 100 samples post-burn-in
        if ((i-n/2) % n/100) == 0:
            self.dsgp_samples += [self.dsgp]
            self.n_change_pts_samples += [[len(self.change_pts[j]) for j in xrange(self.X.shape[1])]]
            # Record actual change-points
            for j in xrange(self.X.shape[1]):
                self.change_pts_samples[j] = list(self.change_pts_samples[j]) + list(self.change_pts[j])


    def get_lines(self):
        y_min_lim = 0.0
        y_max_lim = 0.0
        yp_min_lim = 0.0
        yp_max_lim = 0.0

        if self.plot:
            for j in xrange(self.X.shape[1]):
                self.lines[3*j].set_data(self.X[:,j][np.logical_not(self.is_nan[:, j])], self.dsgp[:,j,0][np.logical_not(self.is_nan[:, j])])
                y_min_lim = min(y_min_lim, self.dsgp[:,j,0][np.logical_not(self.is_nan[:, j])].min())
                y_max_lim = max(y_max_lim, self.dsgp[:,j,0][np.logical_not(self.is_nan[:, j])].max())

                self.lines[3*j+1].set_data(self.X[:,j][np.logical_not(self.is_nan[:, j])], self.dsgp[:,j,1][np.logical_not(self.is_nan[:, j])])   
                yp_min_lim = min(yp_min_lim, self.dsgp[:,j,1][np.logical_not(self.is_nan[:, j])].min())
                yp_max_lim = max(yp_max_lim, self.dsgp[:,j,1][np.logical_not(self.is_nan[:, j])].max())

                self.lines[3*j+2].set_data(self.change_pts[j], np.zeros((len(self.change_pts[j]))))

            self.axes[0][0].set_ylim(y_min_lim, y_max_lim)
            self.axes[1][0].set_ylim(yp_min_lim, yp_max_lim)
            
            return self.lines
        else:
            return []

    '''
    Performs once iteration of the blocked Gibbs sampler.
    '''
    def sample_once(self, int i):
        cdef int n = self.n
        cdef int j
        cdef int post_n = 0
        cdef object line

        t = clock()
        t_sgp = clock()
        self.sample_dsgp()
        self.t_dsgp += clock()-t_sgp

        # Sample change-points on an opt-in basis.
        if self.infer_change_pts:
            t_cp = clock()
            if i%20 == 0:
                self.sample_change_points()
                self.sample_change_points_intensities()
                self.sample_change_points_positions()
            self.t_change_pts += clock()-t_cp

        
        if i%20 == 0:
            t_hypers = clock()
            self.sample_hypers()
            self.t_hypers += clock()-t_hypers

        t_sigma = clock()
        self.sample_noise()
        self.t_sigma = clock()-t_sigma

        self.t_tot += clock()-t
        self.t_pc += [clock()-t]

        # Record some stats post-burn-in (we choose n/2 as burn-in cutoff)
        if i > n/2:
            # Flag that burn-in is over.
            if self.is_post_burn_in == 0:
                self.is_post_burn_in = 1
                self.ess_f = np.empty((n/2, 50))

            # Compute the posterior mean f post-burn-in
            self.f = self.compute_f(self.dsgp) # No need to fully compute f until now
            if self.mean_f_lambda == None:
                self.post_mean_f = (post_n*self.post_mean_f + self.f)/(1.0+post_n)
            else:
                self.post_mean_f = (post_n*self.post_mean_f + self.mean_f_lambda(self.f))/(1.0+post_n)
                
            self.post_mean_dsgp = (post_n*self.post_mean_dsgp + self.dsgp)/(1.0+post_n)
            self.ess_f[i-n-1+self.ess_f.shape[0], :] = self.f[self.ess_idx]
            post_n += 1

        self.record(i, n)


        if (i%self.print_period == 0) and self.should_print:
            print '################'
            print 'Cycle i=', i
            print '################'
            print 'Data size:'
            print 'Original   ', self.X_ind.shape[0], ',', self.X_ind.shape[1]  
            print 'Uniqued    ', self.X.shape[0], ',', self.X.shape[1]      
            print 'Profile:'
            print 't per 100c ', '%.2f'%(100.0*self.t_tot/(i+1))
            print 'DSGP       ', '%.2f'%(self.t_dsgp/self.t_tot)
            print 'Hypers     ', '%.2f'%(self.t_hypers/self.t_tot)
            print 'Change Pts.', '%.2f'%(self.t_change_pts/self.t_tot)
            print 'Sigma      ', '%.2f'%(self.t_sigma/self.t_tot)

            print 'State:     '
            print 'Log. Lik.  ', '%.2f'%(self.log_lik)
            print 'f:         ', '%.2f'%(np.max(self.f)), '%.2f'%(np.min(self.f))
            print 'dsgp:      ', '%.2f'%(np.max(self.dsgp)), '%.2f'%(np.min(self.dsgp))
            #print 'wt_dsgp:   ', '%.2f'%(np.max(self.wt_dsgp)), '%.2f'%(np.min(self.wt_dsgp))
            print 'Noise Var. ', '%.2f'%(self.noise_var)
            #print 'Wt-hypers', self.wt_hypers
            #print 'Hypers'
            hypers = self.compute_kernel_confs(self.wt_hypers)
            for j in xrange(self.X.shape[1]):
                print 'Dim', j
                if self.infer_change_pts:
                    print '#Change-points -- Mean:', '%.2f'%(self.change_pts_lambda[j]*(self.domain_limits[1, j]-self.domain_limits[0, j])),\
                        'Actual:', len(self.change_pts[j])
                    print 'a_j', '%.2f'%(self.domain_limits[0, j]), 'Change-points', self.change_pts[j], 'b_j', '%.2f'%(self.domain_limits[1, j])
                print 'Hypers', hypers[j]
                
            print ''

        return self.get_lines()

        
    '''
    Perform n iterations of the blocked Gibbs sampler.
    '''
    def sample(self, int n=5000):
        cdef int i
        self.n = n
        for i in xrange(n):
            self.sample_once(i)
        return 

    '''
    Perform n iterations of the blocked Gibbs sampler, and illustrate the sampler in action
        through a real-time animation.
    '''
    def run(self, int n=5000):
        cdef object anim
        self.n = n
        if self.plot:
            anim = FuncAnimation(self.fig, self.sample_once, frames=np.arange(n),\
                init_func=self.init_plots, interval=1, blit=False, repeat=False)
            plt.show()
        else:
            self.sample(n=n)