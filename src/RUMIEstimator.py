import numpy as np

from src.LMSEstimator import LMSEstimator
from src.QuantileEstimator import QuantileEstimator


class RUMIEstimator:
    # This is a basic implementation of the RUMI estimator presented in:
    # A. Wischnewski, J. Betz, and B. Lohmann, *Real-Time Learning of Non-Gaussian Uncertainty Models for Robust Control
    # in Autonomous Racing* (2020)

    def __init__(self, hyperparam):
        self.type = 'RUMI'
        self.lmse = LMSEstimator(hyperparam['lms'])
        self.qe_upper = QuantileEstimator(hyperparam['qe_upper'])
        self.qe_lower = QuantileEstimator(hyperparam['qe_lower'])
        self.empiric_perc_log = None

    def update(self, x_data, y_data, debug=False):
        # learn LMS
        residuals = self.lmse.learn(x_data, y_data, debug=debug)
        # learn quantile
        self.qe_upper.learn(residuals)
        self.qe_lower.learn(residuals)

    def sample(self, x_eval):
        mean_predict = self.lmse.model.sample(x_eval)
        upper_bound = mean_predict + np.full_like(mean_predict, self.qe_upper.r_est)
        lower_bound = mean_predict + np.full_like(mean_predict, self.qe_lower.r_est)
        return mean_predict, upper_bound, lower_bound
