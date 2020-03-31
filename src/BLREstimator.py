import numpy as np
import copy
from scipy import special


class BLREstimator:
    # This is an implementation of an Bayesian Linear Regression estimator with variance estimation.
    # The general algorithm is based upon the formulation in chapter 7.6.3. in *Machine Learning -
    # A Probabilistic Persepctive* by Kevin P. Murphy. The initilization and update procedure for recursive application
    # is taken from "C. McKinnon and A. Schoellig, Learning Probabilistic Models for Safe Predictive Control in
    # Unknown Environments"

    def __init__(self, hyperparam):
        self.type = 'BLR'
        self.hyperparam = hyperparam
        self.model = copy.copy(hyperparam["model"])
        self.model.weights = np.zeros_like(self.model.weights)
        self.V = self.hyperparam['tau']**2/self.hyperparam['sigma']**2*np.eye(len(self.model.weights))
        self.blr_a = self.hyperparam['a0']
        self.blr_b = np.array([[self.hyperparam['sigma']**2*self.blr_a]])

        # calculate confidence interval required for given target percentage
        self.conf_interval = np.sqrt(2)*special.erfinv(self.hyperparam['target_percentage'])

    def update(self, x, y, debug=False):
        # calculate feature matrix
        x_mat = self.model.basis_fun(x)
        # update model
        V0 = self.V
        w0 = self.model.weights
        self.V = np.linalg.inv(np.linalg.inv(self.V) + x_mat.T@x_mat)
        self.model.weights = self.V@(np.linalg.inv(V0)@self.model.weights + x_mat.T@y)
        self.blr_a = self.blr_a + self.hyperparam['N']/2
        self.blr_b = self.blr_b + 0.5*(w0.T@np.linalg.inv(V0)@w0 + y.T@y
                                       - self.model.weights.T@np.linalg.inv(self.V)@self.model.weights)
        # prepare prior for next step
        self.V = (self.hyperparam['n0']+self.hyperparam['N'])/self.hyperparam['n0']*self.V
        self.blr_a = self.hyperparam['n0']/(self.hyperparam['n0']+self.hyperparam['N'])*self.blr_a
        self.blr_b = self.hyperparam['n0']/(self.hyperparam['n0']+self.hyperparam['N'])*self.blr_b

    def sample(self, x_eval):
        mean_predict = self.model.sample(x_eval)
        x_mat = self.model.basis_fun(x_eval)
        std_predict = np.sqrt(self.blr_b/self.blr_a*np.diag((np.eye(len(x_eval)) + x_mat@self.V@x_mat.T))).T
        upper_bound = mean_predict + self.conf_interval*std_predict
        lower_bound = mean_predict - self.conf_interval*std_predict

        return mean_predict, upper_bound, lower_bound
