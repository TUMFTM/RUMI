import numpy as np
import time
import matplotlib.pyplot as plt


class UncertaintyEstimator:

    def __init__(self, estimator, dataset, param):

        # store relevant data
        self.dataset = dataset
        self.estimator = estimator
        self.param = param

        # initialize logging variables
        self.mean_log = np.zeros_like(dataset.y_data)
        self.res_log = np.zeros_like(dataset.y_data)
        self.ub_log = np.zeros_like(dataset.y_data)
        self.lb_log = np.zeros_like(dataset.y_data)
        self.perc_log = np.zeros_like(dataset.y_data)

        # transform data into batches according to batch length
        self.n_batches = int(np.floor(len(self.dataset.y_data) / self.param['N']))
        self.x_batch = np.reshape(self.dataset.x_data[:self.param['N'] * self.n_batches],
                                    newshape=(self.param['N'], self.n_batches)).T
        self.y_batch = np.reshape(self.dataset.y_data[:self.param['N'] * self.n_batches],
                                    newshape=(self.param['N'], self.n_batches)).T

        if self.param['verbose']:
            print('Uncertainty estimator of type ' + self.estimator.type + ' successfully initialized ...')

    def learn(self, debug=False):

        if self.param['verbose']:
            print('Start learning of ' + self.estimator.type + ' uncertainty model ...')

        for x, y, idx in zip(self.x_batch, self.y_batch, range(self.n_batches)):
            # evaluate estimator before updating to benchmark to prediction performance
            self.evaluate_estimator(np.expand_dims(x, axis=1), np.expand_dims(y, axis=1), idx)
            # update estimator
            self.estimator.update(np.expand_dims(x, axis=1), np.expand_dims(y, axis=1), debug=debug)
            if self.param['verbose']:
                print('Done with {} of {} batches ...'.format(idx, self.n_batches))

        if self.param['verbose']:
            print('Done with learning of ' + self.estimator.type + ' uncertainty model ...')
            print('')

    def evaluate_estimator(self, x, y, idx):

        # evaluate upper and lower bounds
        mean, ub, lb = self.estimator.sample(x)
        self.mean_log[idx*self.param['N']:(idx+1)*self.param['N'], :] = mean
        self.res_log[idx*self.param['N']:(idx+1)*self.param['N'], :] = y - mean
        self.ub_log[idx*self.param['N']:(idx+1)*self.param['N'], :] = ub
        self.lb_log[idx*self.param['N']:(idx+1)*self.param['N'], :] = lb
        # benchmark percentage of data covered
        self.perc_log[idx*self.param['N']:(idx+1)*self.param['N'], :] = \
            np.sum(np.logical_and(y < ub, lb < y))/len(y)

    def visualize_model(self, x_data, y_data, ax=None, show=False):
        if ax is None:
            fig, ax = plt.subplots()
            ax.grid()

        # visualize final model
        x_eval = np.expand_dims(np.linspace(-3, 3, 300), axis=1)
        mean, ub, lb = self.estimator.sample(x_eval)
        ax.scatter(x_data, y_data)
        ax.plot(x_eval, mean, 'r')
        ax.plot(x_eval, ub, 'r--')
        ax.plot(x_eval, lb, 'r--')
        ax.set_xlabel('Feature')
        ax.set_ylabel('Value')
        ax.set_title('Uncertainty estimate with ' + self.estimator.type + ' estimator')

        if show:
            plt.show()
