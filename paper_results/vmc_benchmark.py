import pickle
import numpy as np
import os, sys
# add top folder to pyton path
top_folder = os.path.abspath(__file__)
sys.path.append(os.path.join(top_folder, '..', '..'))

from src.RUMIEstimator import RUMIEstimator
from src.GPREstimator import GPREstimator
from src.BLREstimator import BLREstimator
from src.UncertaintyEstimator import UncertaintyEstimator
from src.DataGeneration import DataGeneration


def vmc_benchmark():
    # parameters
    path = 'paper_results/output/'

    # The data used for training has been generated with the vehicle dynamics simulation and controller
    # of the TUM Roborace project. These are available in an open source version from https://github.com/TUMFTM.

    # load dataset to be used for training
    data = np.loadtxt('input/Modena_UncertaintyEstimation.csv', delimiter=',')

    # extract feature variables
    ay_target_mps2 = np.expand_dims(data[10000:250000:5, 6], axis=1)
    ay_dist_mps2 = np.expand_dims(data[10000:250000:5, 2], axis=1)

    dataset = DataGeneration.dataset(ay_target_mps2, ay_dist_mps2)

    param_UE = {
        'N': 500,
        'verbose': False
    }
    hyperparam_LMS = {
        'N': param_UE['N'],
        'mu': 10,
        'gamma': 0.001,
        'model': dataset.gt_model,
        'alpha_factor': 1.2,
    }
    hyperparam_QE_upper = {
        'N': param_UE['N'],
        'q_target': 0.99,
        'lambda': 6.0,
        'beta': 0.0,
        'r0_estimate': 6.0
    }
    hyperparam_QE_lower = {
        'N': param_UE['N'],
        'q_target': 0.01,
        'lambda': 6.0,
        'beta': 0.0,
        'r0_estimate': -6
    }
    hyperparam_RUMI = {
        'lms': hyperparam_LMS,
        'qe_upper': hyperparam_QE_upper,
        'qe_lower': hyperparam_QE_lower
    }
    hyperparam_GPR = {
        'N': param_UE['N'],
        'target_percentage': 0.98,
        'length_scale': 4
    }
    hyperparam_BLR = {
        'N': param_UE['N'],
        'sigma': 2.6,
        'tau': 0.01,
        'a0': 8000,
        'n0': 4000,
        'target_percentage': 0.98,
        'model': dataset.gt_model,
    }

    rumi_estimators = []
    gp_estimators = []
    blr_estimators = []

    rumi_estimators.append(UncertaintyEstimator(RUMIEstimator(hyperparam_RUMI), dataset, param_UE))
    rumi_estimators[-1].learn()
    # create and train GP estimators
    gp_estimators.append(UncertaintyEstimator(GPREstimator(hyperparam_GPR), dataset, param_UE))
    gp_estimators[-1].learn()
    # create and train BLR estimators
    blr_estimators.append(UncertaintyEstimator(BLREstimator(hyperparam_BLR), dataset, param_UE))
    blr_estimators[-1].learn()

    full_data = [[gp_estimators, 'GPR'],
                 [blr_estimators, 'BLR'],
                 [rumi_estimators, 'RUMI']]
    pickle.dump(full_data, open(path + "/VMC.p", "wb"))


vmc_benchmark()
