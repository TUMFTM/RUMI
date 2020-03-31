import numpy as np
from scipy import special
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel


class GPREstimator:
    # This is an implementation of an Gaussian Process Regression framework using scikit-learn.

    def __init__(self, hyperparam):
        self.type = 'GPR'
        self.hyperparam = hyperparam
        # create standard GP regression object using scikit learn
        self.kernel = RBF(length_scale=self.hyperparam['length_scale'],
        length_scale_bounds=(self.hyperparam['length_scale'], self.hyperparam['length_scale'])) + \
            WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1))
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0.0)
        # create data dictionary for random sampling of training points
        self.data_dict_x = None
        self.data_dict_y = None

        # calculate confidence interval required for given target percentage
        self.conf_interval = np.sqrt(2)*special.erfinv(self.hyperparam['target_percentage'])

    def update(self, x_data, y_data, debug=False):
        # append data to data dictionary storing available training samples
        if self.data_dict_x is None:
            self.data_dict_x = x_data
            self.data_dict_y = y_data
        else:
            self.data_dict_x = np.concatenate((self.data_dict_x, x_data))
            self.data_dict_y = np.concatenate((self.data_dict_y, y_data))

        # select N_train samples randomly from the dataset
        idx_train = np.random.randint(0, high=len(self.data_dict_x), size=(self.hyperparam['N']))
        x_train = self.data_dict_x[idx_train,:]
        y_train = self.data_dict_y[idx_train,:]

        # train GP regressor object
        self.gp.fit(x_train, y_train)

    def sample(self, x_eval):
        mean_predict, std_predict = self.gp.predict(x_eval, return_std=True)
        # this is required as the return dimension of gp.predict is different before the first training
        if mean_predict.ndim == 1:
            mean_predict = np.expand_dims(mean_predict, axis=1)
        upper_bound = mean_predict + self.conf_interval*np.expand_dims(std_predict, axis=1)
        lower_bound = mean_predict - self.conf_interval*np.expand_dims(std_predict, axis=1)

        return mean_predict, upper_bound, lower_bound
