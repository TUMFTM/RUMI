import numpy as np
import matplotlib.pyplot as plt
import os, sys

# add top folder to pyton path
top_folder = os.path.abspath(__file__)
sys.path.append(os.path.join(top_folder, '..', '..'))

from src.DataGeneration import DataGeneration
from src.RUMIEstimator import RUMIEstimator
from src.UncertaintyEstimator import UncertaintyEstimator


def check_bounds(hyperparam, debug_data, n_batches, diff_weight_log, show=False):

    # check assumptions
    if not (0 < hyperparam['gamma'] < 1/debug_data['sigma_upper']):
        print('Gamma does not meet the conditions')
        return False

    # store polynomial coefficients without disturbance energy
    p2 = -(2*hyperparam['gamma']*debug_data['sigma_lower'] - (hyperparam['gamma']*debug_data['sigma_lower'])**2)
    p1 = 2*debug_data['delta_opt_solution']*(1-hyperparam['gamma']*debug_data['sigma_lower'])
    p0 = debug_data['delta_opt_solution']**2
    lyap_poly = np.poly1d([p2, p1, p0])
    # stretch RPI to achieve exponential convergence
    debug_data['rpi_norm'] = np.max(lyap_poly.roots)*hyperparam['alpha_factor']
    # estimate exponential convergence factor
    debug_data['beta'] = -(p2*debug_data['rpi_norm']**2 + p1*debug_data['rpi_norm'] + p0)/debug_data['rpi_norm']**2

    steps = np.arange(n_batches)
    error_bound = np.maximum(diff_weight_log[0]*np.sqrt(1-debug_data['beta'])**steps, np.full_like(steps, debug_data['rpi_norm'], dtype=float))

    # check if bound holds
    if not (error_bound - np.array(diff_weight_log) >= 0).all():
        print('Bound not satisfied')
        plt.plot(steps, error_bound)
        plt.plot(steps, diff_weight_log)
        plt.ylim([0, 2.5])
        plt.grid()
        plt.show()
        return False

    if show:
        plt.plot(steps, error_bound)
        plt.plot(steps, diff_weight_log)
        plt.grid()
        plt.show()

    return True

# create a test dataset
w0 = np.expand_dims(np.array([1]), axis=1)
datasets = []
for n in range(50):
    datasets.append(DataGeneration.linear_model(n_samples=300000, feature_type='random', noise_type='gaussian',
                                                x_spread=4, w0=w0))

param_UE = {
    'N': 3000,
    'verbose': False
}

hyperparam_LMS = {
        'N': param_UE['N'],
        'mu': 10,
        'gamma': 0.001,
        'model': datasets[0].gt_model,
        'alpha_factor': 1.1
    }

hyperparam_QE_upper = {
    'N': param_UE['N'],
    'q_target': 0.9,
    'lambda': 0.5,
    'r0_estimate': 1
}

hyperparam_QE_lower = {
    'N': param_UE['N'],
    'q_target': 0.1,
    'lambda': 0.5,
    'r0_estimate': -1
}

hyperparam_RUMI = {
    'lms': hyperparam_LMS,
    'qe_upper': hyperparam_QE_upper,
    'qe_lower': hyperparam_QE_lower
}

counter = 0

for dataset in datasets:
    # train estimator on dataset
    uncertainty_est = UncertaintyEstimator(RUMIEstimator(hyperparam_RUMI), dataset, param_UE)
    uncertainty_est.learn(debug=True)

    # evaluate bounds derived in paper on this example
    # set show to false to run all testcases automatically
    if not check_bounds(hyperparam_LMS, uncertainty_est.estimator.lmse.get_debug_results(),
                 uncertainty_est.n_batches, uncertainty_est.estimator.lmse.diff_weight_log, show=True):
        counter += 1

    # comment out to run all testcases automatically
    # uncertainty_est.visualize_model(dataset.x_data, dataset.y_data)
    # plt.show()

print('Final number of violations is: ' + str(counter))
