import pickle
import numpy as np
import os, sys
# add top folder to pyton path
top_folder = os.path.abspath(__file__)
sys.path.append(os.path.join(top_folder, '..', '..'))

from src.RUMIEstimator import RUMIEstimator
from src.GPREstimator import GPREstimator
from src.BLREstimator import BLREstimator
from src.DataGeneration import DataGeneration
from src.UncertaintyEstimator import UncertaintyEstimator


def dist_comparison():
    # parameters
    n_datasets = 20
    n_samples = 60000
    path = os.path.join(os.path.join(top_folder, '..', '..'), 'paper_results', 'output')

    # set disturbance models
    disturbances = ['gaussian', 'nongaussian']

    # set seed for reproducibility
    np.random.seed(0)

    # create a data sets
    datasets_all = []

    # sample ground truth parameters and normalize parameter vector
    w0 = np.expand_dims(np.array(np.random.rand(5) - 0.5), axis=1)
    w0 = w0/np.linalg.norm(w0)

    # generate appropriate datasets
    for disturbance_type in disturbances:
        dataset_batch = []
        for _ in range(n_datasets):
            dataset_batch.append(DataGeneration.linear_model(n_samples=n_samples, feature_type='random',
                                                             noise_type=disturbance_type, x_spread=10, w0=w0))
        datasets_all.append(dataset_batch)

    # specify algorithm parameters
    param_UE = {
        'N': 1000,
        'verbose': False
    }
    hyperparam_LMS = {
        'N': param_UE['N'],
        'mu': 10,
        'gamma': 0.01,
        'model': datasets_all[0][0].gt_model
    }
    hyperparam_QE_upper = {
        'N': param_UE['N'],
        'q_target': 0.9,
        'lambda': 0.5,
        'beta': 0.0,
        'r0_estimate': 2
    }
    hyperparam_QE_lower = {
        'N': param_UE['N'],
        'q_target': 0.1,
        'lambda': 0.5,
        'beta': 0.0,
        'r0_estimate': -2
    }
    hyperparam_RUMI = {
        'lms': hyperparam_LMS,
        'qe_upper': hyperparam_QE_upper,
        'qe_lower': hyperparam_QE_lower
    }
    hyperparam_GPR = {
        'N': param_UE['N'],
        'q0': 0.9,
        'target_percentage': 0.8,
        'length_scale': 3
    }
    hyperparam_BLR = {
        'N': param_UE['N'],
        'sigma': 1.41,
        'tau': 0.1,
        'a0': 8000,
        'n0': 4000,
        'target_percentage': 0.8,
        'model': datasets_all[0][0].gt_model
    }


    counter = 0
    for dataset_batch, disturbance_type in zip(datasets_all, disturbances):
        rumi_estimators = []
        gpr_estimators = []
        blr_estimators = []

        for dataset in dataset_batch:
            # create and train LMS estimator
            rumi_estimators.append(UncertaintyEstimator(RUMIEstimator(hyperparam_RUMI), dataset, param_UE))
            rumi_estimators[-1].learn()
            # create and train GP estimators
            gpr_estimators.append(UncertaintyEstimator(GPREstimator(hyperparam_GPR), dataset, param_UE))
            gpr_estimators[-1].learn()
            blr_estimators.append(UncertaintyEstimator(BLREstimator(hyperparam_BLR), dataset, param_UE))
            blr_estimators[-1].learn()
            counter += 1
            print('Done with ' + str(counter) + ' datasets...')

        full_data = [[gpr_estimators, 'GPR - ' + disturbance_type],
                     [blr_estimators, 'BLR - ' + disturbance_type],
                     [rumi_estimators, 'RUMI - ' + disturbance_type]]
        pickle.dump(full_data, open(path + "\disturbance_variation_" + disturbance_type + ".p", "wb"))


dist_comparison()
