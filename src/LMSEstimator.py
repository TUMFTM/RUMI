import numpy as np
import copy


class LMSEstimator:

    def __init__(self, hyperparam):
        # initialize hyperparameters of algorithm
        self.hyperparam = hyperparam
        self.model = copy.copy(hyperparam["model"])
        self.model.weights = np.zeros_like(self.model.weights)

        # monitor the maximum occured change in the batch L2 optimal solution (debug only)
        self.delta_opt_norm_max = 0
        self.weight_opt_old = None
        self.weight_opt_log = []
        self.diff_weight_log = []
        # monitor singular values of persistent excitation matrix (debug only)
        self.sigma_p_upper = None
        self.sigma_p_lower = None

    def learn(self, x_data, y_data, debug=False):
        # calculate prediction residuals
        residuals = y_data - self.model.sample(x_data)
        # calculate feature matrix
        x_mat = self.model.basis_fun(x_data)
        # define function handle for construction of weights for regression
        m_construct = lambda v: self.hyperparam['mu'] / (1 + self.hyperparam['mu']*np.dot(v, v))
        m_weights = np.apply_along_axis(m_construct, 1, x_mat)
        m_mat = np.diag(np.apply_along_axis(m_construct, 1, x_mat))

        # calculate additional debug information
        if debug:
            # calculate l2 optimal solution for batch
            Aw = np.diag(np.sqrt(m_weights))@x_mat
            Bw = np.diag(np.sqrt(m_weights))@y_data
            weight_opt, _, _, _ = np.linalg.lstsq(Aw, Bw, rcond=None)
            # only adopt value if already an old value was present
            if self.weight_opt_old is not None:
                if np.linalg.norm(weight_opt-self.weight_opt_old) > self.delta_opt_norm_max:
                    self.delta_opt_norm_max = np.linalg.norm(weight_opt-self.weight_opt_old)
            self.weight_opt_old = weight_opt
            self.weight_opt_log.append(weight_opt[0][0])

            # calculate singular values for PE matrix for batch
            _, p_s_loc, _ = np.linalg.svd(self.estimate_pe(x_mat, self.hyperparam))
            if self.sigma_p_upper is None:
                self.sigma_p_upper = np.max(p_s_loc)
                self.sigma_p_lower = np.min(p_s_loc)
            else:
                self.sigma_p_upper = np.maximum(self.sigma_p_upper, np.max(p_s_loc))
                self.sigma_p_lower = np.minimum(self.sigma_p_upper, np.min(p_s_loc))

            # store difference in log
            self.diff_weight_log.append(np.linalg.norm(weight_opt - self.model.weights))

        # update weights
        self.model.weights = self.model.weights +\
                             self.hyperparam['gamma'] * np.matmul(np.matmul(x_mat.T, m_mat), residuals)

        return residuals

    @staticmethod
    def estimate_pe(x_mat, hyperparam):
        pe_est = np.zeros(shape=(x_mat.shape[1], x_mat.shape[1]))
        for v in x_mat:
            v2 = np.expand_dims(v, axis=1)
            pe_est = \
                pe_est + np.matmul(hyperparam['mu']*v2, v2.T) / (1 + hyperparam['mu']*np.matmul(v2.T, v2))
        return pe_est

    def get_debug_results(self):
        debug_results = {
            'sigma_upper': self.sigma_p_upper,
            'sigma_lower': self.sigma_p_upper,
            'delta_opt_solution': self.delta_opt_norm_max,
        }
        return debug_results
