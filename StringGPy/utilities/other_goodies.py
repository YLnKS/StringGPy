import numpy as np
from numpy.dual import svd
from scipy.spatial.distance import pdist, squareform, cdist
from StringGPy.utilities.gpy_kernels import StringGPKern, string_cov
import sys
from multiprocessing import Pool, cpu_count

'''
Computes the (unconditional) covariance matrix between two vectors.
'''
def covMatrix(X, Y, theta, symmetric = True, kernel = lambda u, theta: theta[0]*theta[0]*np.exp(-0.5*u*u/(theta[1]*theta[1])), \
        dist_f=None):
    if len(np.array(X).shape) == 1:
        _X = np.array([X]).T
    else:
        _X = np.array(X)
        
    if len(np.array(Y).shape) == 1:
        _Y = np.array([Y]).T
    else:
        _Y = np.array(Y)
        
    if dist_f == None:
        if symmetric:
            cM = pdist(_X)
            M = squareform(cM)
            M = kernel(M, theta)
            return M
        else:
            cM = cdist(_X, _Y)
            M = kernel(cM, theta)
            return M
    else:
        if symmetric:
            cM = pdist(_X, dist_f)
            M = squareform(cM)
            M = kernel(M, theta)
            return M
        else:
            cM = cdist(_X, _Y, dist_f)
            M = kernel(cM, theta)
            return M
    return


def get_kernel_lambda(k_type):
    '''
    From string to functional form of the kernel as a lambda.
    '''
    if k_type == "se":
        kernel = lambda u, theta: theta[0]*theta[0]*np.exp(-0.5*u*u/(theta[1]*theta[1]))

    if k_type == "ma32":
        kernel = lambda u, theta: theta[0]*theta[0]*(1+(np.sqrt(3.0)/theta[1])*\
            np.abs(u))*np.exp(-(np.sqrt(3.0)/theta[1])*np.abs(u))

    if k_type == "ma52":
        kernel = lambda u, theta: theta[0]*theta[0]*(1.0 + (np.sqrt(5.0)/theta[1])*np.abs(u) +\
            (5.0/(3.0*theta[1]*theta[1]))*u*u)*np.exp(-(np.sqrt(5.0)/theta[1])*np.abs(u))

    if k_type == "rq":
        self.kernel = lambda u, theta: theta[0]*theta[0]*((1.0+u*u/(2*theta[2]*theta[1]*theta[1]))**(-theta[2]))
        
    if k_type == "sse":
        kernel = lambda u, theta: sum([theta[3*i]*theta[3*i]*\
            np.exp(-0.5*u*u/(theta[3*i+1]*theta[3*i+1]))*\
            np.cos(2.0*np.pi*theta[3*i+2]) for i in xrange(len(theta)/3)]) 

    if k_type == "sma32":
        kernel = lambda u, theta: sum([theta[3*i]*theta[3*i]*\
            (1+(np.sqrt(3.0)/theta[3*i+1])*np.abs(u))*np.exp(-(np.sqrt(3.0)/theta[3*i+1])*\
            np.abs(u))*np.cos(2.0*np.pi*theta[3*i+2]) for i in xrange(len(theta)/3)])

    if k_type == "sma52":
        kernel = lambda u, theta: sum([theta[3*i]*theta[3*i]*\
            (1.0 + (np.sqrt(5.0)/theta[3*i+1])*np.abs(u) + (5.0/(3.0*theta[3*i+1]*theta[3*i+1]))*u*u)*\
            np.exp(-(np.sqrt(5.0)/theta[3*i+1])*np.abs(u))*\
            np.cos(2.0*np.pi*theta[3*i+2]) for i in xrange(len(theta)/3)]) 

    return kernel

'''
Inverts a positive-definite matrix taking care of conditioning
'''
def inv_cov(cov):
    U, S, V = svd(cov)
    eps = 0.0
    oc = np.max(S)/np.min(S)
    if oc > 1e8:
        nc = np.min([oc, 1e8])
        eps = np.min(S)*(oc-nc)/(nc-1.0)
    
    LI = np.dot(np.diag(1.0/(np.sqrt(np.absolute(S) + eps))), U.T)
    covI= np.dot(LI.T, LI)
    return covI

'''
Computes the inverse and the determinant of a covariance matrix in one go, using
    SVD.
    Returns a structure containing the following keys:
        inv: the inverse of the covariance matrix,
        L: the pseudo-cholesky factor US^0.5,
        det: the determinant of the covariance matrix.
'''
def SVDFactorise(cov, max_cn=1e8):
    U, S, V = svd(cov)
    eps = 0.0
    oc = np.max(S)/np.min(S)
    if oc > max_cn:
        nc = np.min([oc, max_cn])
        eps = np.min(S)*(oc-nc)/(nc-1.0)

    L = np.dot(U, np.diag(np.sqrt(S+eps)))        
    LI = np.dot(np.diag(1.0/(np.sqrt(np.absolute(S) + eps))), U.T)
    covI= np.dot(LI.T, LI)
    
    res = {}
    res['inv'] = covI.copy()
    res['L'] = L.copy()    
    res['det'] = np.prod(S+eps)
    res['log_det'] = np.sum(np.log(S+eps))
    res['LI'] = LI.copy()
    res['eigen_vals'] = S+eps
    res['u'] = U.copy()
    res['v'] = V.copy()
    return res 


'''
Computes the hyper-parameters and the noise variance of the GP regression model
    under i.i.d Gaussian noise.
'''
def gp_regression_calibrate(X, Y, hyper_type = 'SE', x_0 = np.array([1.0, 1.0, 1.0 ]),\
    penalty_center=0.0):
        
    from numpy.core.umath_tests import inner1d
    
    if hyper_type.lower() == 'ma32':
        kernel = lambda u, theta: theta[0]*theta[0]*(1+(np.sqrt(3.0)/theta[1])*\
            np.abs(u))*np.exp(-(np.sqrt(3.0)/theta[1])*np.abs(u))
        # Derivative of the kernel with respect to the input length scale
        kernel_d2 = lambda u, theta: theta[0]*theta[0]*(3.0/(theta[1]**3)*u*u)*\
            np.exp(-(np.sqrt(3.0)/theta[1])*np.abs(u))
       
    else:
        kernel = lambda u, theta: theta[0]*theta[0]*np.exp(-0.5*u*u/(theta[1]*theta[1]))
        # Derivative of the kernel with respect to the input length scale
        kernel_d2 = lambda u, theta: kernel(u, theta)*u*u/(theta[1]*theta[1]*theta[1])
        

    def log_marginal(x):
        noise_var = x[0]*x[0]
        theta = np.abs(x[1:])
    
        cov = covMatrix(X, X, theta, symmetric=True, kernel=kernel) + noise_var*np.eye(len(X))
        try:
            svd_factor = SVDFactorise(cov, max_cn=1e6)
        except:
            print theta, x
            raise ValueError

        cov_i = svd_factor['inv']
        cov_det = svd_factor['det']
        res = np.log(cov_det)+np.dot(Y, np.dot(cov_i, Y))
        
        if penalty_center != None:
            res += 0.5*((theta[1]-np.array([penalty_center]))/1.0)**2
        return res
        
        
    from scipy.optimize import minimize
    # Attempt 1: warm-up/smart initialisation
    res = minimize(log_marginal, x_0, method='L-BFGS-B')
    x_opt = res.x
    # Attempt 2: max from smart initialisation
    res = minimize(log_marginal, x_0, method='L-BFGS-B')
    x_opt = res.x
    
    return (x_opt[0]*x_opt[0], np.abs(x_opt[1:]))

'''
Computes the hyper-parameters and the noise variance of the GP regression model
    under i.i.d Gaussian noise.
'''
def string_gp_regression_calibrate(X, Y, n_string, min_t, max_t, x_0, hyper_type = 'SE', ):
        
    from scipy.optimize import fmin_bfgs        

    K = n_string # Number of strings
    
    # Create the array of input string gp indices (X might not be sorted)
    X_couples = [(X[i], i) for i in xrange(len(X))]
    from operator import itemgetter
    X_couples.sort(key=itemgetter(0))
    X_sorted = [elt[0] for elt in X_couples]
    
    def log_marginal(x):
        noise_vars = x[:K]**2 # The first K terms are string noise variances
        thetas = []
        for _ in xrange(K):
            thetas += [np.abs([x[K+2*_], x[K+1+2*_]])] # The next 2K are thetas
        
        thetas = np.array(thetas)
        drvs = x[-n_string:] # The last K are used to determine boundary times

        b_X_sorted = boundaries_from_drivers(drvs, min_t, max_t)
        
        if n_string > 1:
            X_sorted_string_ids = []
            idx = 1
            for x in X_sorted:
                while x > b_X_sorted[idx]:
                    idx += 1
                X_sorted_string_ids  += [idx]
        else:
            X_sorted_string_ids = [1]*len(X_sorted)

        X_sorted_string_ids_couples = [(X_sorted_string_ids[i], X_couples[i][1]) for i in xrange(len(X_couples))]
        X_sorted_string_ids_couples.sort(key=itemgetter(1))
        X_string_ids = np.array([elt[0] for elt in X_sorted_string_ids_couples])-1 #String indexed from 0 here

        cov = string_cov(X, X, thetas, b_X_sorted, hyper_type.lower()) + np.diag(noise_vars[X_string_ids])
        try:
            svd_factor = SVDFactorise(cov)
        except:
            print thetas
            print b_X_sorted
            raise ValueError
        cov_i = svd_factor['inv']
        cov_det = svd_factor['det']
        
        res = np.log(cov_det)+np.dot(Y, np.dot(cov_i, Y))
        return res
        
        
    # Attempt 1: warm-up/smart initialisation
    x_opt = fmin_bfgs(log_marginal, x_0, disp=False)
    # Attempt 2: max from smart initialisation
    x_opt = fmin_bfgs(log_marginal, np.abs(x_opt), disp=False)
    
    return np.abs(x_opt)

'''
Utility function that maps K real numbers (drvs) to a partition 
    of the interval [min_t, max_t] in K.
'''
def boundaries_from_drivers(drvs, min_t, max_t):
    const_drivers = 1.0 + 9.0/(1.0+np.exp(-drvs))
    probas = np.cumsum(const_drivers)/sum(const_drivers)
    return np.array([min_t] + list(min_t + (max_t-min_t)*probas))


###################################
#       LOG TO STD-OUT AND FILE   #
###################################
class Tee(object):
    def __init__(self, fl_name, mode):
        """
        :type mode: str
        :type fl_name: str
        """
        self.file = open(fl_name, mode)
        self.stdout = sys.stdout
        sys.stdout = self

    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def release(self):
        sys.stdout = self.stdout
        self.file.close()

    def flush(self):
        self.file.flush()
        self.stdout.flush()


def print_compiler_options():
    import distutils.sysconfig
    import distutils.ccompiler
    compiler = distutils.ccompiler.new_compiler()
    distutils.sysconfig.customize_compiler(compiler)
    print compiler.compiler_so


def robust_invert_noisy_cov(args):
    '''
    Computes (robustly) the invert of a 
        noisy auto-covariance matrix.
    '''
    Xs = args[0]
    hypers = args[1]
    k_type = args[2]
    noise_var = args[3]
    kernel = get_kernel_lambda(k_type)

    cov_train_train = covMatrix(Xs, Xs, hypers, symmetric=True, kernel=kernel)\
        + noise_var*np.eye(len(Xs))
    cov_train_train_inv = inv_cov(cov_train_train)

    return cov_train_train_inv

def parallel_invert_noisy_cov(args_list, M):
    '''
    '''
    p = Pool(min(cpu_count()-1, 30, M))
    cov_invs = p.map(robust_invert_noisy_cov, args_list)
    p.close()
    p.join()

    return cov_invs

def robust_neg_log_lik(args):
    '''
    Computes (robustly) the invert of a 
        noisy auto-covariance matrix.
    '''
    Xs = args[0]
    hypers = args[1]
    k_type = args[2]
    noise_var = args[3]
    kernel = get_kernel_lambda(k_type)
    Ys = args[4]

    cov_train_train = covMatrix(Xs, Xs, hypers, symmetric=True, kernel=kernel)\
        + noise_var*np.eye(len(Xs))

    try:
        svd_factor = SVDFactorise(cov_train_train, max_cn=1e5)
    except:
        print "Error in robust_neg_log_lik", hypers
        raise ValueError

    cov_inv = svd_factor['inv']
    log_cov_det = svd_factor['log_det']
    ll = 0.5*(log_cov_det + np.dot(Ys, np.dot(cov_inv, Ys)) + len(Xs)*np.log(2.0*np.pi))

    return ll


def parallel_neg_log_lik(args_list, M):
    '''
    '''
    p = Pool(min(cpu_count()-1, 30, M))
    lls = map(robust_neg_log_lik, args_list)
    p.close()
    p.join()

    return np.sum(lls)

