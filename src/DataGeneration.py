import numpy as np
from src.LinearModel import LinearModel


class DataSet:
    def __init__(self, x_data, y_data, n_data, gt_model):
        self.x_data = x_data
        self.y_data = y_data
        self.n_data = n_data
        self.gt_model = gt_model


class DataGeneration:
    @staticmethod
    def dataset(x_data, y_data):
        # sample zero ground truth model
        w0 = np.zeros(shape=(1, 1))
        gt_model = LinearModel(w0, 'Paper')

        return DataSet(x_data, y_data, np.zeros_like(y_data), gt_model)

    @staticmethod
    def linear_model(n_samples, feature_type, noise_type, x_spread, w0=None):
        if w0 is None:
            w0 = np.expand_dims(np.array(np.random.rand(5)-0.5), axis=1)

        # create a ground truth model and normalize length of weight vector
        x0 = np.expand_dims(np.arange(len(w0)) - np.floor(len(w0)/2), axis=0)
        param = {
            'x0': x0,
            'l': 1
        }
        gt_model = LinearModel(w0, 'RBF', param)

        # create feature variables
        if feature_type == 'random':
            x_data = (np.random.rand(n_samples, 1) - 0.5)*2*x_spread
        elif feature_type == "correlated":
            x_data = np.sin(np.expand_dims(np.arange(n_samples), axis=1)/50)*x_spread
        else:
            print('No valid feature represenation given')
            x_data = np.zeros(shape=(n_samples, 1))
        v_data = gt_model.basis_fun(x_data)

        # create noise
        if noise_type == "deterministic":
            n_data = np.sin(np.arange(n_samples)/200) + 0.2 + np.random.normal(0, 0.1, n_samples)
        elif noise_type == "gaussian":
            n_data = np.random.normal(0, 0.3, n_samples)
        elif noise_type == "nongaussian":
            n_sign = np.random.randn(n_samples)
            n_data = np.sign(n_sign)*np.random.normal(1.0, 0.2, n_samples)
        elif noise_type == "uniform":
            n_data = (np.random.rand(n_samples)-0.5)*2
        elif noise_type == "correlated":
            n_data = np.squeeze(np.matmul(np.array([[1, 1, 0, 0, 0]]), v_data.T))**3 \
                     + np.random.normal(0, 0.1, n_samples)
        else:
            print('No valid noise specification given ...')
            n_data = np.zeros(shape=(n_samples,))

        y_data = gt_model.sample(x_data) + np.expand_dims(n_data, axis=1)

        return DataSet(x_data, y_data, n_data, gt_model)
