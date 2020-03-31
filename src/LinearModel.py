import numpy as np


class LinearModel:

    def __init__(self, weights, basis_fun_type, param=None):
        self.param = param
        self.weights = weights
        self.basis_fun_type = basis_fun_type

    def basis_fun(self, x):
        if self.basis_fun_type == 'Paper':
            return x
        else:
            if self.param is None:
                print('No valid parameter set for RBF functions specified.')
                return None
            else:
                return np.exp(-((x-self.param['x0'])/self.param['l'])**2)

    def sample(self, X):
        # evaluates the model at the points in the matrix X [samples X dim]
        y = np.matmul(self.basis_fun(X), self.weights)
        return y
