# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 16:49:14 2015

@author: ylkomsamo
"""
import numpy as np
from time import time
from scipy.stats import invgamma
import pyximport
pyximport.install()
import cy_utilities as cy_ut
import cStringGPy as csgp
from guppy import hpy
import gc

h=hpy()

'''
    MCMC sampler to solve some supervised learning tasks using string Gaussian processes.
'''
class BaseSGPSampler(object):
    def __init__(self, data, kernel_types, hypers_max, domain_partition, has_noise, ll_type, noise_var, link_f_type,\
            should_print=True, sgp_bridge=False):
        d = len(domain_partition)
        assert len(kernel_types) == d and len(hypers_max) == d, "Input dimensions should be consistent." 
        assert data.shape[1] == (d+1), "The data parameter should have d+1 columns."
        
        self.ll_type = ll_type
        self.link_f_type = link_f_type

        # Dimension of the input space
        self.dim = d
        s_times = [None]*d # Inner sting times
        b_times = [None]*d # Boundary times
        
        for j in range(d):
            # Partition of the j-th String GP.
            partition_j = domain_partition[j] # Boundary times in the j-th dimension
            b_times[j] = list(partition_j)

            # Number of elements in the partition of the j-th input String.
            K_j = len(partition_j)-1 # Number of strings in the j-th dimension.
            
            # Some basic consistency checks
            assert len(kernel_types[j]) == K_j and len(hypers_max[j]) == K_j, "Parameters should be consistent in the %r-th dimension." % j
 
            # j-th coordinate of all data points (i.e. all times in the j-th dimension)
            times_j = data[:,j].copy()
            
            # Unique and sort all times
            times_j = np.sort(list(times_j) + list(partition_j))
            
            # Initialise string variables
            if not sgp_bridge:
                partition_j = np.sort(partition_j)
                s_times[j] = [None]*K_j
                for k in range(0, K_j):
                    s_times[j][k] = list(set([t for t in times_j if t > partition_j[k] and t < partition_j[k+1]]))
            else:
                s_times[j] = [[]]*K_j
        
        self.data = data # N x (d+1) ndarray, the first d columns representing input coordinates and the last coordinate representing the correspondign label.
        # Set string variables
        self.sgp_bridge=sgp_bridge

        if not self.sgp_bridge:
            self.hypers_max = hypers_max # Same shape as hypers
            self.kernel_hypers_norm = csgp.sample_norm_hypers(self.hypers_max, None, 1.0)
        else:
            for j in xrange(len(hypers_max)):
                hypers_max[j] = [hypers_max[j][0]]*len(hypers_max[j])
            self.hypers_max = hypers_max

            kernel_hypers_norm = csgp.sample_norm_hypers(self.hypers_max, None, 1.0)
            for j in xrange(len(kernel_hypers_norm)):
                kernel_hypers_norm[j] = [kernel_hypers_norm[j][0]]*len(kernel_hypers_norm[j])
            self.kernel_hypers_norm = kernel_hypers_norm

        self.kernel_hypers = csgp.scaled_sigmoid(self.hypers_max, self.kernel_hypers_norm) # Local kernel hyper-parameters
        self.kernel_types = kernel_types # List of lists of ekrnel types (e.g.: se, ma32, ma52, rq, sm, sma32, sma52, etc...)
        self.s_times = s_times
        self.b_times = b_times

        # Whitened values
        self.Xb = csgp.sample_whtn_bound_conds(self.b_times)
        self.Xs = csgp.sample_whtn_string(self.s_times)

        # Eigen value analyses
        self.bound_eig = csgp.compute_bound_ls(self.kernel_types, self.kernel_hypers, self.b_times)
        self.string_eig = csgp.cond_eigen_anals(self.kernel_types, self.kernel_hypers, self.b_times, self.s_times)

        # Set the string GP and derivative values
        self.sgp_values, self.dsgp_values = csgp.compute_sgps_from_lxs(self.kernel_types, self.kernel_hypers, self.b_times,\
            self.s_times, self.Xb, self.Xs, self.bound_eig, self.string_eig)
              
        # Whether or not the supervised learning model has a noise term (classification problems might not).
        self.has_noise = has_noise
        self.noise_var = noise_var
        
        # Store latent function samples
        self.f_samples = []
        self.gp_samples = []
        self.dgp_samples = []
        
        # Record the link function
        if link_f_type == "prod":
            self.link_f = lambda z: np.prod(z)
        else:
            self.link_f = lambda z: np.sum(z)
        
        # Inital log-likelihoods
        #self.ll_old = sum([self.model_log_likelihood(self.current_latent_f(_[:-1]), _[-1], self.noise_var) for _ in self.data])
        self.ll_old = csgp.model_log_lik(self.data, self.sgp_values, self.link_f_type, self.ll_type, self.noise_var)

        self.should_print = should_print
        self.mh_accepts = 0


        self.tt = 0.0
        self.t1 = 0.0
        self.t21 = 0.0
        self.t22 = 0.0
        self.t3 = 0.0
        self.t4 = 0.0
        self.t5 = 0.0
    
        self.ftt = 0.0
        self.f0 = 0.0
        self.f1 = 0.0
        self.f2 = 0.0
        self.f3 = 0.0
        self.f4 = 0.0

    def latent_f(self, x, sgp_values):
        z=[sgp_values[j][csgp.float_as_idx(x[j])] for j in xrange(len(x))]
        return self.link_f(z)

    '''
        Compute latent function value at input x using latest samples.
    '''        
    def current_latent_f(self, x):
        return self.latent_f(x, self.sgp_values)

    
    def latent_gradient(self, x, dsgp_values):
        return [dsgp_values[j][csgp.float_as_idx(x[j])] for j in xrange(len(x))]
        
    '''
        Compute latent gradient value at input x using latest samples.
    '''        
    def current_latent_gradient(self, x):
        return self.latent_gradient(x, self.dsgp_values)
        
    '''
        Record latest samples.
    '''        
    def record(self):
        f_samples = np.array([self.current_latent_f(x[:-1]) for x in self.data])
        self.f_samples += [f_samples]
        # self.gp_samples += [self.sgp_values]
        # self.dgp_samples += [self.dsgp_values]
        return
        
    '''
        Sample the univariate derivative string GPs jointly using Elliptical Slice Sampling.
    '''
    def sample_dsgp(self, tolerance=0.001):
        t0 = time()
        t = time()
        _ll_old = self.ll_old
        _u = np.random.uniform(0.0, 1.0)
        _ll_old += np.log(_u)

        self.f0 += time()-t

        t = time()
        new_Xb = csgp.sample_whtn_bound_conds(self.b_times)
        new_Xs = csgp.sample_whtn_string(self.s_times)
        self.f1 += time()-t

        _a = np.random.uniform(0.0, 2.0*np.pi)
        _a_min = _a - 2.0*np.pi
        _a_max = _a
        
        while np.abs((_a_max-_a_min)) > tolerance:
            t = time()
            _Xb = csgp.elliptic_tranform_lx(self.Xb, new_Xb, _a)
            _Xs = csgp.elliptic_tranform_lx(self.Xs, new_Xs, _a)
            self.f2 += time()-t

            t = time()
            _sgp_values, _dsgp_values = csgp.compute_sgps_from_lxs(self.kernel_types, self.kernel_hypers, self.b_times,\
                self.s_times, _Xb, _Xs, self.bound_eig, self.string_eig)
            self.f3 += time()-t

            t = time()
            _ll_new = csgp.model_log_lik(self.data, _sgp_values, self.link_f_type, self.ll_type, self.noise_var)
            self.f4 += time()-t
                
            if _ll_new > _ll_old:
                break
            else:
                _a = np.random.uniform(_a_min, _a_max)
                if _a < 0.0:
                    _a_min = _a
                else:
                    _a_max = _a

        self.sgp_values, self.dsgp_values = _sgp_values, _dsgp_values
        self.Xb, self.Xs = _Xb, _Xs
        self.ll_old = _ll_new
        self.ftt += time() - t0
        return
    
    '''
        Sample the string hyper-parameters in parallel using Elliptical Slice Sampling.
    '''
    def sample_kernel_parameters_ess(self, tolerance=0.001):
        # Current and new normalized hyper-parameters
        new_kernel_hypers_norm = csgp.sample_norm_hypers(self.hypers_max, None, 1.0)
        if self.sgp_bridge:
            for j in xrange(len(new_kernel_hypers_norm)):
                new_kernel_hypers_norm[j] = [new_kernel_hypers_norm[j][0]]*len(new_kernel_hypers_norm[j])
        
        _u = np.random.uniform()        
        _ll_old = self.ll_old
        _ll_old += np.log(_u)

        _a = np.random.uniform(0.0, 2.0*np.pi)
        _a_min = _a - 2.0*np.pi
        _a_max = _a
            
        while np.abs((_a_max-_a_min)) > tolerance: 
            # ELliptic transform in the whitened space followed by a scaled sigmoid transform
            _kernel_hypers_norm = csgp.elliptic_tranform_lx(self.kernel_hypers_norm, new_kernel_hypers_norm, _a)
            _kernel_hypers = csgp.scaled_sigmoid(self.hypers_max, _kernel_hypers_norm)
            
            # Eigen value analyses
            _bound_eig = csgp.compute_bound_ls(self.kernel_types, _kernel_hypers, self.b_times)
            _string_eig = csgp.cond_eigen_anals(self.kernel_types, _kernel_hypers, self.b_times, self.s_times)

            # Compute the string GP and derivative values
            _sgp_values, _dsgp_values = csgp.compute_sgps_from_lxs(self.kernel_types, _kernel_hypers, self.b_times,\
                self.s_times, self.Xb, self.Xs, _bound_eig, _string_eig)

            # Compute the log-likelihood factor.
            _ll_new = csgp.model_log_lik(self.data, _sgp_values, self.link_f_type, self.ll_type, self.noise_var)

                
            if _ll_new > _ll_old:
                break
            else:
                _a = np.random.uniform(_a_min, _a_max)
                if _a < 0.0:
                    _a_min = _a
                else:
                    _a_max = _a
    
        self.kernel_hypers_norm = _kernel_hypers_norm
        self.kernel_hypers = _kernel_hypers
        self.ll_old = _ll_new
        self.bound_eig = _bound_eig
        self.string_eig = _string_eig
        self.sgp_values = _sgp_values
        self.dsgp_values = _dsgp_values
        return


    '''
        Sample the string hyper-parameters using Metropolis-Hastings.
    '''
    def sample_kernel_parameters_mh(self):
        t0 = time()
        t = time()
        # Sigmoid normal proposal
        _kernel_hypers_norm = csgp.sample_norm_hypers(self.hypers_max, self.kernel_hypers_norm, 0.05)
        if self.sgp_bridge:
            for j in xrange(len(_kernel_hypers_norm)):
                _kernel_hypers_norm[j] = [_kernel_hypers_norm[j][0]]*len(_kernel_hypers_norm[j])
        _kernel_hypers = csgp.scaled_sigmoid(self.hypers_max, _kernel_hypers_norm)
        self.t1 += time()-t

        t = time()
        # Eigen value analyses
        _bound_eig = csgp.compute_bound_ls(self.kernel_types, _kernel_hypers, self.b_times)
        self.t21 += time()-t

        t = time()
        _string_eig = csgp.cond_eigen_anals(self.kernel_types, _kernel_hypers, self.b_times, self.s_times)
        self.t22 += time()-t
        
        t = time()
        # Compute the string GP and derivative values
        _sgp_values, _dsgp_values = csgp.compute_sgps_from_lxs(self.kernel_types, _kernel_hypers, self.b_times,\
            self.s_times, self.Xb, self.Xs, _bound_eig, _string_eig)
        self.t3 += time()-t

        t = time()
        # Compute the log-likelihood factor.
        _ll_new = csgp.model_log_lik(self.data, _sgp_values, self.link_f_type, self.ll_type, self.noise_var)
        self.t4 += time()-t

        t = time()
        _norm_ll_old = csgp.log_lik_whtn(self.kernel_hypers_norm, [])
        _norm_ll_new = csgp.log_lik_whtn(_kernel_hypers_norm, [])

        lr = np.log(np.random.uniform())
        self.t5 += time()-t
        self.tt += time()-t0

        if lr < (_ll_new - self.ll_old + _norm_ll_new - _norm_ll_old):
            self.kernel_hypers_norm = _kernel_hypers_norm
            self.kernel_hypers = _kernel_hypers
            self.ll_old = _ll_new
            self.bound_eig = _bound_eig
            self.string_eig = _string_eig
            self.sgp_values = _sgp_values
            self.dsgp_values = _dsgp_values
            self.mh_accepts += 1


    '''
    Only relevant for regression problem.
    When required, we put an inverse gamma prior on the variance of the noise
        parameter.
    In this implementation we assume an iid Gaussian noise.
    '''
    def sample_noise(self):
        if not self.has_noise:
            return 
        # The noise parameter is only used for regression problems.
        #   We use an (conjuguate) inverse-gamma prior with alpha = 1, beta = 1.
        alpha = 1.0
        beta = 1.0
        errors = np.array([self.current_latent_f(x[:-1])-x[-1] for x in self.data])
        alpha += len(errors)/2.0
        beta += sum(errors**2)/2.0
        noise_var = invgamma.rvs(alpha)
        noise_var *= beta
        if not np.isnan(noise_var):
            self.noise_var = noise_var
        return
        
    '''
    Performs one block Gibbs cycle update
    '''
    def sample(self, n=100):
        for _ in xrange(n):
            #print 'Total number of GCed objects', len(gc.get_objects())
            should_print = (len(self.f_samples)%50 == 0)
            if should_print:
                print 'Cycle #', len(self.f_samples)
                
            td = time()
            self.sample_dsgp()
            ta = time()

            if should_print:
                print '###########'
                print 'Sampling function values'
                print 'Took', ta-td, 's'
                
            td = time()
            if _ < 100:
                self.sample_kernel_parameters_ess()
                self.sample_kernel_parameters_mh()
            else:
                self.sample_kernel_parameters_mh()
            ta = time()

            if should_print:
                print 'Sampling kernel hyper-parameters'
                print 'Took', ta-td, 's'
                
            td = time()        
            self.sample_noise()
            ta = time()

            if should_print:
                print 'Sampling noise parameters'
                print 'Took', ta-td, 's'
                print 'Results:'
                print 'LL', self.ll_old
                print 'Noise var', self.noise_var
                print 'MH diagnosis'
                print 'Acceptance Ratio:', '%.3f'%(1.0*self.mh_accepts/n)
                print 'T1', str(self.t1/self.tt)
                print 'T21', str(self.t21/self.tt)
                print 'T22', str(self.t22/self.tt)
                print 'T3', str(self.t3/self.tt)
                print 'T4', str(self.t4/self.tt)
                print 'T5', str(self.t5/self.tt)

                print 'Function sampling diagnosis'
                print 'F0', str(self.f0/self.ftt)
                print 'F1', str(self.f1/self.ftt)
                print 'F2', str(self.f2/self.ftt)
                print 'F3', str(self.f3/self.ftt)
                print 'F4', str(self.t4/self.ftt)
                print 'Heap diagnosis'
                print h.heap()
                print ''
                print ''
                print ''

            self.record()
            
        
'''
    MCMC sampler for regression tasks using string Gaussian processes.
'''
class SGPRegressor(BaseSGPSampler):
    def __init__(self, data, kernel_types, hypers_max, domain_partition, noise_var, link_f_type, should_print=True, sgp_bridge=False):
        super(SGPRegressor, self).__init__(data, kernel_types, hypers_max, domain_partition, True, "gaussian",\
            noise_var, link_f_type, should_print=should_print, sgp_bridge=sgp_bridge)

'''
    MCMC sampler for binary classification tasks using string Gaussian processes.
        The classes should be 0/1. P(yi=1)=1.0/(1.0 + exp(-f(xi)))
'''
class SGPBinaryClassifier(BaseSGPSampler):
    def __init__(self, data, kernel_types, hypers_max, domain_partition, link_f_type, should_print=True, sgp_bridge=False):
        super(SGPBinaryClassifier, self).__init__(data, kernel_types, hypers_max, domain_partition, False, "logit",\
            0.0, link_f_type, should_print=should_print, sgp_bridge=sgp_bridge)