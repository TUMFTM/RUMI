import numpy as np
import matplotlib.pyplot as plt

from src.DataGeneration import DataGeneration
from src.GPREstimator import GPREstimator
from src.BLREstimator import BLREstimator
from src.RUMIEstimator import RUMIEstimator
from src.UncertaintyEstimator import UncertaintyEstimator

print('Create dataset ...')
w0 = np.expand_dims(np.array(np.random.rand(5)-0.5), axis=1)
dataset = DataGeneration.linear_model(n_samples=80000, feature_type='random', noise_type='gaussian', x_spread=5, w0=w0)

# parametrize all example estimators
param_UE = {
    'N': 500,
    'verbose': False
}
hyperparam_LMS = {
        'N': param_UE['N'],
        'mu': 10,
        'gamma': 0.002,
        'model': dataset.gt_model,
        'alpha_factor': 1.2
    }
hyperparam_QE_upper = {
    'N': param_UE['N'],
    'q_target': 0.9,
    'lambda': 0.5,
    'r0_estimate': 5
}
hyperparam_QE_lower = {
    'N': param_UE['N'],
    'q_target': 0.1,
    'lambda': 0.5,
    'r0_estimate': -5
}
hyperparam_RUMI = {
    'lms': hyperparam_LMS,
    'qe_upper': hyperparam_QE_upper,
    'qe_lower': hyperparam_QE_lower
}
hyperparam_BLR = {
    'N': param_UE['N'],
    'sigma': 5,
    'tau': 0.1,
    'a0': 8000,
    'n0': 4000,
    'model': dataset.gt_model,
    'target_percentage': 0.8
}
hyperparam_GPR = {
    'N': param_UE['N'],
    'target_percentage': 0.8,
    'length_scale': 1
}

print('Initialize estimators ...')
myRUMIestimator = UncertaintyEstimator(RUMIEstimator(hyperparam_RUMI), dataset, param_UE)
myGPRestimator = UncertaintyEstimator(GPREstimator(hyperparam_GPR), dataset, param_UE)
myBLRestimator = UncertaintyEstimator(BLREstimator(hyperparam_BLR), dataset, param_UE)

print('Learn uncertainty models (might take some time)...')
myRUMIestimator.learn()
myGPRestimator.learn()
myBLRestimator.learn()

print('Show results...')
myRUMIestimator.visualize_model(dataset.x_data, dataset.y_data)
myGPRestimator.visualize_model(dataset.x_data, dataset.y_data)
myBLRestimator.visualize_model(dataset.x_data, dataset.y_data)

fig, ax = plt.subplots()
ax.grid()
ax.set_xlabel('Feature')
ax.set_ylabel('Data covered in %')
ax.set_title('Comparison of uncertainty model estimators')
plt.plot(myGPRestimator.perc_log, label='GPR')
plt.plot(myBLRestimator.perc_log, label='BLR')
plt.plot(myRUMIestimator.perc_log, label='RUMI')
plt.legend()
plt.show()
