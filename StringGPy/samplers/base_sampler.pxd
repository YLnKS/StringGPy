cimport numpy as np

'''
    Type definition.
'''
cdef class BaseSGPSampler:
    cdef public np.ndarray X
    cdef public np.ndarray is_nan
    cdef public np.ndarray X_ind
    cdef public np.ndarray Y_train
    cdef public np.ndarray f
    cdef public np.ndarray post_mean_f
    cdef public object mean_f_lambda    # Used to compute the posterior mean over a function of f rather than f.
    cdef public int n_train
    cdef public bytes kernel_type
    cdef public np.ndarray hypers_max
    cdef public int has_noise
    cdef public bytes ll_type
    cdef public double noise_var
    cdef public bytes link_f_type
    cdef public bytes hypers_prior_type
    cdef public int should_print
    cdef public int n

    cdef public np.ndarray dsgp         # Values of the Derivative String Gaussian Process (ordered as in self.X not as in X)
    cdef public np.ndarray post_mean_dsgp
    cdef public list dsgp_samples
    cdef public np.ndarray wt_dsgp      # Whitened version of dsgp
    cdef public np.ndarray l_factors    # l factors that 'colour' wt_dsgp into dsgp
    cdef public np.ndarray m_factors
    cdef public np.ndarray l_hypers     # Mapping input -> 
    cdef public double log_lik

    # GPU variables
    cdef public int use_GPU
    cdef public object stringgpu_worker

    # Variables pertaining to infering change-points (synthetic boundary times)
    cdef public list change_pts              # List of arrays of change points. Each inner array should be sorted
    cdef public np.ndarray X_config_idx       # Maps inputs to kernel configurations.
    cdef public np.ndarray change_pts_lambda  # Array of change-point intensities.
    cdef public np.ndarray change_pts_a       # Vector of 'a' parameters of the Gamma prior on the intensities. 
    cdef public np.ndarray change_pts_b       # Vector of 'b' parameters of the Gamma prior on the intensities. 

    cdef public list wt_hypers              
    cdef public list change_pts_samples     # List of lists containing change-points sampled post burn-in.
    cdef public list n_change_pts_samples   # Samples of number of change-points in each input dim.
    cdef public int is_post_burn_in         # True if chain has already burned-in.
    cdef public np.ndarray domain_limits    # 2D array, each column containing min/max input values.
    cdef public int infer_change_pts

    # Profiling variables
    cdef public double t_tot
    cdef public double t_dsgp
    cdef public double t_hypers
    cdef public double t_change_pts
    cdef public double t_sigma
    cdef public list t_pc

    cdef public int plot
    cdef public object fig
    cdef public np.ndarray axes
    cdef public list lines
    cdef public int print_period

    cdef public list ess_idx
    cdef public np.ndarray ess_f


    cdef list compute_factors(self, np.ndarray[np.float64_t, ndim=3, mode='fortran'] l_hypers)

    cdef np.ndarray[np.float64_t, ndim=3] compute_dsgp(self, np.ndarray[np.float64_t, ndim=3] wt_dsgp,\
        np.ndarray[np.float64_t, ndim=3] l_factors, np.ndarray[np.float64_t, ndim=3] m_factors)

    cdef np.ndarray[np.float64_t, ndim=2] _reorder_x(self, np.ndarray[np.float64_t, ndim=2] sgp)

    cdef np.ndarray[np.float64_t, ndim=1] compute_f(self, np.ndarray[np.float64_t, ndim=3] dsgp)

    cdef np.ndarray[np.float64_t, ndim=3, mode='fortran'] compute_l_hypers(self, list wt_hypers)

    cdef void compute_factors_j(self, np.ndarray[np.float64_t, ndim=3, mode='fortran'] l_hypers,\
        np.ndarray[np.float64_t, ndim=3, mode='c'] l_factors, np.ndarray[np.float64_t, ndim=3, mode='c'] m_factors, int j, np.ndarray[int, ndim=1] i_s)

    cdef void compute_dsgp_j(self, np.ndarray[np.float64_t, ndim=3] wt_dsgp,\
        np.ndarray[np.float64_t, ndim=3] l_factors, np.ndarray[np.float64_t, ndim=3] m_factors,\
        np.ndarray[np.float64_t, ndim=3] dsgp, int j)