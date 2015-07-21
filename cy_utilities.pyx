
import pyximport
pyximport.install()
import numpy as np
cimport numpy as np
import cython
import cStringGPy as csgp

'''
    Takes in an ndarray with indices, z_t, z_t^{\prime}
        and returns the processes z_t and z_t^{\prime} as two dictionaries.
'''
@cython.boundscheck(False)
def dsgp_as_dict(np.ndarray[double, ndim=2] data):
    cdef int i
    cdef dict z = {}
    cdef dict zp = {}

    for i in range(data.shape[0]):
        z[float_as_idx(data[i, 0])] = data[i, 1]
        zp[float_as_idx(data[i, 0])] = data[i, 2]

    return (z, zp)



'''
    Takes in a list of ndarray with indices, z_t, z_t^{\prime}
        and returns the processes z_t and z_t^{\prime} as two lists of dictionaries.
'''
@cython.boundscheck(False)
def dsgp_as_list_dict(l_data):
    d=len(l_data)
    z=[None]*d
    zp=[None]*d

    cdef int i
    for i in range(d):
        data=l_data[i]
        z[i], zp[i]=dsgp_as_dict(np.array(data).reshape(len(data), 3))

    return (z, zp)
    
    

'''
    Takes in z_t and z_t^{\prime} as dictionaries and returns the processes as
        ndarray with rows [t, z_t, z_t^{\prime}].
'''
@cython.boundscheck(False)
def dsgp_from_dict(np.ndarray[double, ndim=1] time, dict data_z, dict data_zp):
    cdef int i
    res = np.zeros((time.shape[0], 3))

    for i in range(time.shape[0]):
        res[i][0]=time[i]
        res[i][1]=data_z[float_as_idx(time[i])]
        res[i][2]=data_zp[float_as_idx(time[i])]

    return res
    
'''
    Applies an elliptic transformation the values of two dictionaries.
'''
@cython.boundscheck(False) 
def elliptic_t(dict old, dict new, double a):
    cdef dict res={}
    cdef str key

    res= {key: np.cos(a)*old[key] + np.sin(a)*new[key] for key in old.keys()}
    return res

'''
    Applies an elliptic transformation the values of two list of dictionaries.
'''
@cython.boundscheck(False) 
def elliptic_ts(old, new, double a):
    cdef int d
    res=[None]*len(old)
    
    for d in range(len(old)):
        res[d]=elliptic_t(old[d], new[d], a)
        
    return res
    
'''
    Returns a rounded version of a double as a string.
'''
@cython.boundscheck(False) 
def float_as_idx(double val):
    return "%.4f" % round(val,4)
    
    
'''
    Converts normalized kernel hyper-parameters into full hyper-parameters.
        The hyper-parameters are assumed to be scaled sigmoid transformation of 
            the normalized hyper-parameters (on which i.i.d. Gaussian priors are typically put):
            theta = theta_max/(1.0+exp(-theta_norm))
'''
@cython.boundscheck(False)     
def n_hypers_to_hypers(int dim, template, hypers_max, n_hypers):
    hypers = [None]*dim
    cdef int j
    cdef int k
    cdef int K_j
    cdef int s=0
    cdef int e=0
    
    for j in xrange(dim):
        K_j = len(template[j])
        hypers[j]=[None]*K_j
        for k in xrange(K_j):
            s=e
            e=e+len(hypers_max[j][k])
            hypers[j][k] = np.multiply(hypers_max[j][k], 1.0/(1.0 + np.exp(-n_hypers[s:e])))
            
    return hypers
    
    
'''
    Converts full hyper-parameters into normalized hyper-parameters (see-above).
'''
@cython.boundscheck(False)     
def hypers_to_n_hypers(int dim, hypers, hypers_max):
    cdef np.ndarray[double, ndim=1] res
    cdef int j
    cdef int k
    cdef int K_j
    cdef int s=0
    cdef int e=0
    tmp_res = []
    
    for j in xrange(dim):
        K_j = len(hypers[j])
        for k in xrange(K_j):
            tmp_res += list(-np.log(-1.0+np.divide(hypers_max[j][k], hypers[j][k])))
            
    res=np.array(tmp_res)
    return res
        

'''
Inverts a positive-definite matrix taking care of conditioning
'''
def inv_cov(cov):
    cdef double eps = 0.0
    cdef double oc = 0.0
    cdef double nc = 0.0
    cdef double log_det = 0.0
    cdef double max_cn = 1e6
    
    U, S, V = np.dual.svd(cov)
    oc = np.max(S)/np.min(S)
    if oc > max_cn:
        nc = np.min([oc, max_cn])
        eps = np.min(S)*(oc-nc)/(nc-1.0)
    
    LI = np.dot(np.diag(1.0/(np.sqrt(np.absolute(S) + eps))), U.T)
    covI= np.dot(LI.T, LI)
    log_det=sum(np.log(np.absolute(S)))
    
    return (covI, log_det)   
    
    
'''
    Compute the log-likelihood of a collection of independent centered SDGP
'''
@cython.boundscheck(False)
def sgp_ll(z, z_p, int dim, kernel_types, kernel_hypers, b_times, s_times, LIs, Ms, log_dets):
    cdef int j=0
    cdef int K_j=0
    cdef int k=0
    cdef double res=0
    
    # The unidimensional derivative string GPs are assumed to be independent
    for j in xrange(dim):
        # Step 0: Likelihood of the leftmost boundary conditions
        t=b_times[j][0]
        cov=csgp.deriv_cov(np.array([t]), np.array([t]), kernel_hypers[j][0], kernel_types[j][0])
        cov_i, log_det=inv_cov(cov)

        b_cond=np.array([z[j][float_as_idx(t)], z_p[j][float_as_idx(t)]])
        res += -np.log(2.0*np.pi)-0.5*log_det-0.5*np.dot(b_cond, np.dot(cov_i, b_cond))
        K_j = len(kernel_types[j])
        
        for k in xrange(K_j):
            # Likelihood of the boundary conditions
            prev_b_cond=b_cond.copy()
            prev_t=t
            
            t=b_times[j][1+k]
            b_cond=np.array([z[j][float_as_idx(t)], z_p[j][float_as_idx(t)]])
            
            cov_new_old=csgp.deriv_cov(np.array([t]), np.array([prev_t]), kernel_hypers[j][k], kernel_types[j][k])
            cov_new_new=csgp.deriv_cov(np.array([t]), np.array([t]), kernel_hypers[j][k], kernel_types[j][k])
            cov_old_old=csgp.deriv_cov(np.array([prev_t]), np.array([prev_t]), kernel_hypers[j][k], kernel_types[j][k])
            cov_old_old_i, _=inv_cov(cov_old_old)

            cov=cov_new_new-np.dot(cov_new_old, np.dot(cov_old_old_i, cov_new_old.T))
            cov_i, log_det=inv_cov(cov)

            m = np.dot(np.dot(cov_new_old, cov_old_old_i), prev_b_cond)
            res += -np.log(2.0*np.pi) - 0.5*log_det - 0.5*np.dot(b_cond-m, np.dot(cov_i, b_cond-m))
            # print 'log_det', log_det
            # print 'quad_form', np.dot(b_cond-m, np.dot(cov_i, b_cond-m))
            # print 'bond_cond', b_cond
            # print 'm', m
            # print 'cov', cov
            # print 'cov_i', cov_i
            # print 'test I', np.dot(cov, cov_i)
            
            # Likelihood of the inner string times
            if len(s_times[j][k]) > 0:
                n_j_k = len(s_times[j][k])
                
                # Covariance matrix (+ inverse and log-det) of the DSGP at inner times conditional on the boundary conditions.
                s_log_det = log_dets[j][k]
                s_L_i = LIs[j][k]
                s_cov_i = np.dot(s_L_i.T, s_L_i)
                
                # Mean of the DSGP at innner times conditional on the boundary conditions.
                s_M = Ms[j][k]
                s_b_cond = np.array([prev_b_cond[0], prev_b_cond[1], b_cond[0], b_cond[1]])
                s_m = np.dot(s_M, s_b_cond)
                
                # DSGP values at inner strings
                s_v= np.zeros(2*n_j_k)
                s_v[0::2]= [z[j][float_as_idx(_)] for _ in s_times[j][k]]
                s_v[1::2]= [z_p[j][float_as_idx(_)] for _ in s_times[j][k]]
                
                res += -n_j_k*np.log(2.0*np.pi) - 0.5*s_log_det - 0.5*np.dot(s_v-s_m, np.dot(s_cov_i, s_v-s_m))
                
    return res

