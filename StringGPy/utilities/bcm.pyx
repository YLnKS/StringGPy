import numpy as np
import GPy
from GPy.models.gp_regression import GPRegression
from StringGPy.utilities.other_goodies import covMatrix, SVDFactorise, inv_cov, get_kernel_lambda,\
    parallel_invert_noisy_cov, parallel_neg_log_lik
from scipy.optimize import minimize
from scipy.linalg import solve_triangular, cholesky
from time import clock

"""
Implements Bayesian Committee Machine Regression
"""
class BCM(object):
    def __init__(self, Xs,Ys, k_type, n_spectral_mixt=2):
        assert len(Xs) == len(Ys), "Xs and Ys should have the same len"
        self.M = len(Xs)
        self.Xs = Xs
        self.Ys = Ys
        self.k_type = k_type
        self.models = [None]*self.M
        assert k_type in ("se", "ma32", "ma52", "rq", "sse", "sma32", "sma52"), "Only vanilla kernels allowed for now"

        flatten_Xs = [val for sublist in Xs for val in sublist]
        flatten_Ys = [val for sublist in Ys for val in sublist]

        self.noise_var = 0.05*np.std(flatten_Ys)
        self.n_spectral_mixt = n_spectral_mixt

        if self.k_type in ("se", "ma32", "ma52"):
            self.hypers = np.ones((2))
            self.hypers[0] = np.std(flatten_Ys) # Output variance
            self.hypers[1] = 0.5*(np.max(flatten_Xs)-np.min(flatten_Xs)) # Input scale
        elif self.k_type == "rq":
            self.hypers = np.ones((3))
            self.hypers[0] = np.std(flatten_Ys) # Output variance
            self.hypers[1] = 0.5*(np.max(flatten_Xs)-np.min(flatten_Xs)) # Input scale
        else:
            self.hypers = np.ones((3*self.n_spectral_mixt))
            self.hypers[0::3] = np.std(flatten_Ys) # Output variance
            self.hypers[1::3] = 0.5*(np.max(flatten_Xs)-np.min(flatten_Xs)) # Input scale
            self.hypers[2::3] = self.hypers[1::3]*np.random.uniform(0.5, 1.0, size=len(self.hypers[1::3]))

        self.kernel = get_kernel_lambda(k_type)
        self.cov_train_train_inv = [None]*self.M
        self.log_lik_count = 0
        self.log_lik_ts = []

        print "Initial hypers", self.hypers

    def neg_log_lik(self, x):
        self.log_lik_count += 1
        
        t = clock()
        noise_var = x[0]*x[0]
        theta = np.abs(x[1:])
        args_list = [(self.Xs[i], theta, self.k_type, noise_var, self.Ys[i]) for i in xrange(self.M)]
        ll = parallel_neg_log_lik(args_list, self.M)
        self.log_lik_ts += [clock()-t] 

        return ll


    def training_log_lik(self):
        x = np.array([np.sqrt(self.noise_var)] + list(self.hypers))
        return -self.neg_log_lik(x)

    def optimize(self):
        print 'Beginning optimization'
        # Form optimization parameters
        x_0 = np.array([np.sqrt(self.noise_var)] + list(self.hypers))

        # Minimize the negative log-likelihood
        res = minimize(self.neg_log_lik, x_0, method='L-BFGS-B')

        # Update object with consistently with learned hyper-parameters.
        x_opt = np.abs(res.x)
        self.noise_var = x_opt[0]*x_opt[0]
        self.hypers = x_opt[1:]
        args_list = [(self.Xs[i], self.hypers, self.k_type, self.noise_var) for i in xrange(self.M)]
        self.cov_train_train_inv = parallel_invert_noisy_cov(args_list, self.M)
        print 'End of optimization'

    def predict_bcm_full(self, x_test):
        # Prediction as per BCM (Tresp, 2000), jointly.
        S_qq = covMatrix(x_test, x_test, self.hypers, symmetric=True, kernel=self.kernel)
        C = -(self.M-1.0)*inv_cov(S_qq)
        inv_cond_covs = [None]*self.M
        cond_means = [None]*self.M
        predictive_mean = np.zeros(x_test.shape[0])

        cdef int i
        for i in xrange(self.M):
            cov_test_train = covMatrix(x_test, self.Xs[i], self.hypers, symmetric=False, kernel=self.kernel)
            cond_cov = S_qq - np.dot(cov_test_train, np.dot(self.cov_train_train_inv[i], cov_test_train.T))
            inv_cond_covs[i] = inv_cov(cond_cov)
            C = C + inv_cond_covs[i]

            cond_mean = np.dot(cov_test_train, np.dot(self.cov_train_train_inv[i], self.Ys[i]))
            cond_means[i] = cond_mean.copy()

            predictive_mean = predictive_mean + np.dot(inv_cond_covs[i], cond_means[i])

        predictive_cov = inv_cov(C)
        predictive_mean = np.dot(predictive_cov, predictive_mean)

        return (predictive_mean.flatten(), np.diag(predictive_cov).flatten())

    def predict_bcm(self, x_test):
        if len(x_test.shape) == 1:
            x_test = x_test.reshape(len(x_test), 1)

        mean = []
        var = []
        # Prediction as per BCM (Tresp, 2000), one test point at a time.
        cdef int i
        for i in xrange(x_test.shape[0]):
            predictive_mean, predictive_var = self.predict_bcm_full(x_test[i, :].reshape(1, x_test.shape[1]))
            mean += [predictive_mean]
            var += [predictive_var]

        return (np.array(mean).flatten(), np.array(var).flatten())

    def predict_rbcm(self, x_test):
        if len(x_test.shape) == 1:
            x_test = x_test.reshape(len(x_test), 1)

        mean = []
        var = []
        cdef int i
        cdef int j
        for j in xrange(x_test.shape[0]):
            cond_means = [None]*self.M
            betas = [None]*self.M
            betas_inv_cond_vars = [None]*self.M
            cov_test_test = covMatrix(x_test[j, :].reshape(1, x_test.shape[1]), x_test[j, :].reshape(1, x_test.shape[1]),
                                      self.hypers, symmetric=True, kernel=self.kernel)
            prior_var = cov_test_test[0,0]

            for i in range(self.M):
                cov_test_train = covMatrix(x_test[j, :].reshape(1, x_test.shape[1]), self.Xs[i], self.hypers,
                                           symmetric=False, kernel=self.kernel)
                cond_cov = cov_test_test - np.dot(cov_test_train, np.dot(self.cov_train_train_inv[i], cov_test_train.T))
                cond_means[i] = np.dot(cov_test_train, np.dot(self.cov_train_train_inv[i], self.Ys[i])).flatten()
                betas[i] = 0.5*(np.log(cov_test_test[0,0])-np.log(cond_cov[0,0]))
                betas_inv_cond_vars[i] = betas[i]/cond_cov[0,0]

            betas_inv_cond_vars = np.array(betas_inv_cond_vars)
            cond_means = np.array(cond_means)

            predictive_var = 1.0/(np.sum(betas_inv_cond_vars) + (1.0-np.sum(betas))/prior_var)
            predictive_mean = predictive_var*np.dot(betas_inv_cond_vars, cond_means)

            mean += [predictive_mean]
            var += [predictive_var]

        return (np.array(mean).flatten(), np.array(var).flatten())


class GBCM(BCM):
    '''
    Implements the Generalized Bayesian Committee Machine for GP Classification with a logistic sigmoid.
        The approach consists of training indepedent experts by marginal likelihood maximization using Laplace approxi-
        mation. Prediction is then performed using the BCM aggregation on the approximate (Gaussian) predictive distri-
        bution coming from the experts.
    '''

    def __init__(self, Xs,Ys, k_type, n_spectral_mixt=2):
        super(GBCM, self).__init__(Xs, Ys, k_type,n_spectral_mixt=n_spectral_mixt)
        self.grads = [None]*self.M
        self.marginal_log_lik = None

    def neg_hessian(self, f):
        '''
        Computes the negative Hessian (as a function of latent function values) of the log-likelihood of GP classifi-
            cation with logistic link function.
        :param f: latent function values.
        :return:
        '''
        pi = 1.0/(1.0 + np.exp(-f))
        return pi*(1.0-pi)

    def grad(self, f, y):
        '''
        Computes the gradient (as a function of latent function values) of the log-likelihood of GP classification with
            logistic link function.
        :param f: latent function values.
        :param y: labels (+/-1)
        :return:
        '''
        pi = 1.0/(1.0 + np.exp(-f))
        t = 0.5*(1.0+y)
        return t - pi

    def posterior_mode_fs(self, theta):
        '''
        Iteratively determines the posterior mode in GP classification with logistic link.
            This implementation follows Algorithm 3.1 in Rassmussen and Williams
        :param theta: Vector of hyper-parameters.
        :return: (fs, marg_log_lik) the vector of M posterior modes (corresponding to the M experts), and the overall
            marginal log likelihood.
        '''
        fs = [None]*len(self.Ys)
        marg_log_lik = 0.0
        cdef int i
        for i in xrange(len(self.Ys)):
            f = np.zeros(self.Ys[i].shape)
            K = covMatrix(self.Xs[i], self.Xs[i], theta, symmetric=True, kernel=self.kernel)

            err = 1.0
            while err > 0.005:
                w = self.neg_hessian(f)
                s_w = np.sqrt(w)
                B = np.eye(f.shape[0]) + (((K*s_w).T)*s_w).T

                try:
                    svd_factor = SVDFactorise(B, max_cn=1e5)
                except:
                    print "Error in GBCM.posterior_mode_fs", theta
                    raise ValueError

                BI = svd_factor['inv']  # B^{-1}
                log_det = svd_factor['log_det']

                b = w*f + self.grad(f, self.Ys[i])
                a = b - s_w*np.dot(BI, s_w*np.dot(K, b))

                f_new = np.dot(K, a)
                err = np.dot(f-f_new, f-f_new)/np.dot(f_new, f_new)
                f = f_new.copy()

            fs[i] = f.copy()
            pi = 1.0/(1.0 + np.exp(-f))
            obs_log_lik = np.dot(0.5*(1.0+self.Ys[i]), np.log(pi)) + np.dot(0.5*(1.0-self.Ys[i]), np.log(1.0-pi))
            marg_log_lik += -0.5*np.dot(a, f) + obs_log_lik - log_det

        return fs, marg_log_lik


    def neg_log_lik(self, x):
        '''
        Negative log-likehood corresponding to a set of hyper-parameters.
        :param x: vector of kernel hyper-parameters.
        :return: negative log-likelihood.
        '''
        self.log_lik_count += 1
        t = clock()
        theta = np.abs(x)
        fs, ll = self.posterior_mode_fs(theta)
        self.log_lik_ts += [clock()-t] 
        return -ll

    def optimize(self):
        print 'Beginning optimization'
        x_0 = self.hypers

        # Attempt 1: warm-up/smart initialisation
        res = minimize(self.neg_log_lik, x_0, method='L-BFGS-B')
        x_opt = np.abs(res.x)
        self.hypers = x_opt.copy()
        self.fs, self.marginal_log_lik = self.posterior_mode_fs(self.hypers)

        cdef int i
        for i in range(self.M):
            cov_train_train = covMatrix(self.Xs[i], self.Xs[i], self.hypers, symmetric=True, kernel=self.kernel)
            wi = self.neg_hessian(self.fs[i])
            self.cov_train_train_inv[i] = inv_cov(cov_train_train + np.diag(1.0/wi))
            self.grads[i] = self.grad(self.fs[i], self.Ys[i])

        print 'End of optimization'


    def predict(self, x_test):
        if len(x_test.shape) == 1:
            x_test = x_test.reshape(len(x_test), 1)

        posterior_mean_f = []
        posterior_var_f = []
        predictions = []
        probas = []  # Probability of class 1

        cdef int i
        cdef int j
        for j in xrange(x_test.shape[0]):
            _x_test = x_test[j, :].reshape(1, x_test.shape[1])
            cov_test_test = covMatrix(_x_test, _x_test, self.hypers, symmetric=True, kernel=self.kernel)

            C = -(self.M-1.0)/cov_test_test[0,0]
            inv_cond_covs = [None]*self.M
            cond_means = [None]*self.M

            for i in range(self.M):
                cov_test_train = covMatrix(_x_test, self.Xs[i], self.hypers, symmetric=False, kernel=self.kernel)
                cond_cov = cov_test_test - np.dot(cov_test_train, np.dot(self.cov_train_train_inv[i], cov_test_train.T))
                inv_cond_covs[i] = 1.0/cond_cov[0,0]
                C = C + inv_cond_covs[i]
                cond_mean = np.dot(cov_test_train, self.grads[i])
                cond_means[i] = cond_mean

            predictive_mean_f = 0.0
            for i in range(self.M):
                predictive_mean_f += inv_cond_covs[i]*cond_means[i]

            predictive_var_f = 1.0/C
            predictive_mean_f = predictive_var_f*predictive_mean_f

            posterior_mean_f += [predictive_mean_f]
            posterior_var_f += [predictive_var_f]

            predictions += [np.sign(predictive_mean_f)]
            probas += [1.0/(1.0+np.exp(-predictive_mean_f))]  # MAP approach

        return np.array(predictions).flatten(), np.array(probas).flatten(), np.array(posterior_mean_f).flatten(),\
               np.array(posterior_var_f).flatten()