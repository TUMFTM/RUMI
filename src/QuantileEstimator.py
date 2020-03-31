import numpy as np


class QuantileEstimator:

    def __init__(self, hyperparam):
        # initialize hyperparameter
        self.hyperparam = hyperparam
        # initialize estimate
        self.r_est = self.hyperparam['r0_estimate']

    def learn(self, y_data):
        # calculate alpha estimate
        alpha_est = np.sum(np.squeeze(y_data) < self.r_est) / self.hyperparam['N']
        self.r_est -= self.hyperparam['lambda']*(alpha_est - self.hyperparam['q_target'])
