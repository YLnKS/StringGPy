# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 02:52:03 2015

@author: ylks
"""


'''
Regression Example
'''
from StringGPy.samplers.regression import SGPRegressor
# Create the model.
kernel_type = "se"          # Many kernels are available including Matern 1/2, 3/2, 5/2, RQ, and stationary spectral kernels.
all_x = None                # Your inputs here 
all_y = None                # Your outputs here
n_train = None              # Your number of training samples here.
y_train = all_y[:n_train]   # Trainging labels.
link_f_type = "sum"         # Type of link function.
hyper_prior_type = "log"    # Log-Gaussian or Sigmoid-Gaussian prior on hyper-parameters.
scaling_hypers = None       # A scaling vector of hyper-parameters prior, which corresponds to half the mode.
plot = False                # Whether to display real-time plots of learned functions.
infer_change_pts = True     # Whether to infer the location of change-points in string kernel memberships.

m = SGPRegressor(all_x, y_train, n_train, kernel_type, scaling_hypers, link_f_type, True, hyper_prior_type, plot,\
    infer_change_pts=infer_change_pts)
    
# Run the Markov Chain / Train
m.run(n=n) # n is the total number of Gibbs cycles

# Predict
predictive_means = m.post_mean_f # m.post_mean_f is the posterior mean function at all input points (all_x) obtained by using sample post burn-in.

'''
Binary Classification Example
'''
from StringGPy.samplers.binary_classification import SGPBinaryClassifier

# Create the model.
m = SGPBinaryClassifier(all_x, y_train, n_train, kernel_type, scaling_hypers, link_f_type, True, hyper_prior_type, plot,\
    infer_change_pts=infer_change_pts)

# Run the Markov Chain / Train  
m.run(n=n)

# Predict
probas = m.post_mean_f  # In the case of StringGP classification this is in fact the posterior probability that the input is of class 1.
predictions = np.sign(probas-0.5)
