import numpy as np
import matplotlib.pyplot as plt
import random
import tikzplotlib

from src.DataGeneration import DataGeneration

# output file (set to none for plotting)
file = None

# set algorithm parameters
hyperparam_QE = {
    'N': 500,
    'q_target': 0.75,
    'lambda': 2,
    'r0_estimate': 5,
    'delta_r0': 0.2
}

# set RPI size for visualization
rpi_min = 0.5

# set seed for reproducibility
np.random.seed(0)
# create model without mean value for better visualization
w0 = np.expand_dims(np.zeros(5), axis=1)
data = DataGeneration.linear_model(n_samples=10000, feature_type='correlated', noise_type='uniform',
                                   x_spread=5, w0=w0)

# calculate empiric estimator stability and maximize upper and lower bounds
x = np.linspace(-10, 10, 801)
alpha_est_max = np.zeros_like(x)
alpha_est_min = np.ones_like(x)
alpha_est_list = []
for idx in range(0, len(data.y_data) - hyperparam_QE['N'], 10):
    alpha_est = np.zeros_like(x)
    for x_idx in np.arange(len(x)):
        alpha_est[x_idx] = np.sum(data.y_data[idx:idx + hyperparam_QE['N']] < (x[x_idx])) / hyperparam_QE['N']
    alpha_est_list.append(alpha_est)

    # store minimum and maximum values
    alpha_est_max = np.maximum(alpha_est_max, alpha_est)
    alpha_est_min = np.minimum(alpha_est_min, alpha_est)

# find best fitting uniform distribution cdf for upper and lower bound
idx_upper = np.argmax(alpha_est_min > 0.99)
idx_lower = np.argmax(alpha_est_max > 0.01)
m_est = 1 / (x[idx_upper] - x[idx_lower])
shift_left = x[-1]
shift_right = x[0]
while np.any(np.greater(alpha_est_max, np.clip(m_est * (x - shift_left), 0, 1))):
    shift_left -= 0.01
while np.any(np.less(alpha_est_min, np.clip(m_est * (x - shift_right), 0, 1))):
    shift_right += 0.01
# determine center of the two bounds
shift_center = (shift_left + shift_right) / 2
# calculate value for which alpha_tilde = 0
shift_zero = hyperparam_QE['q_target'] / m_est + shift_center
alpha_tilde_est_min = np.clip(m_est * (x - shift_left + shift_zero), 0, 1) - hyperparam_QE['q_target']
alpha_tilde_est_max = np.clip(m_est * (x - shift_right + shift_zero), 0, 1) - hyperparam_QE['q_target']
# calculate recommended lambda
lambda_recom = 1 / (m_est)

fig, ax = plt.subplots()
ax.plot(x, alpha_est_min, 'g--')
ax.plot(x, alpha_est_max, 'g--')
ax.plot(x, np.clip(m_est * (x - shift_right), 0, 1), 'r')
ax.plot(x, np.clip(m_est * (x - shift_left), 0, 1), 'r')
ax.grid()
plt.show()

# create conditions vectors
delta_x_cond_u = -x + rpi_min
delta_x_cond_u[x < -rpi_min] = \
    -2 * x[x < -rpi_min]
delta_x_cond_u[x > rpi_min] = 0
delta_x_cond_l = -x - rpi_min
delta_x_cond_l[x < -rpi_min] = 0

# visualize everything
fig, ax = plt.subplots()
ax.grid()
# draw five random samples from alpha
labeled = False
for alpha_est in random.choices(alpha_est_list, k=5):
    if not labeled:
        ax.plot(x - shift_zero, -lambda_recom * (alpha_est - hyperparam_QE['q_target']),
                color=(0.3451, 0.3451, 0.3530),
                label='$-\\lambda\\tilde{\\alpha}$ - samples')
        labeled = True
    else:
        ax.plot(x - shift_zero, -lambda_recom * (alpha_est - hyperparam_QE['q_target']),
                color=(0.3451, 0.3451, 0.3530))
ax.plot(x, -lambda_recom * alpha_tilde_est_max, color=(0, 0.3961, 0.7412),
        label='$-\\lambda\\tilde{\\alpha}^{+/-}$ - bounds')
ax.plot(x, -lambda_recom * alpha_tilde_est_min, color=(0, 0.3961, 0.7412))
ax.plot(x, -lambda_recom * alpha_tilde_est_max + hyperparam_QE['delta_r0'], color=(0.6353, 0.6784, 0),
        label='$\Delta r_\mathrm{o}^+$ - influence')
ax.plot(x, -lambda_recom * alpha_tilde_est_min - hyperparam_QE['delta_r0'], color=(0.6353, 0.6784, 0))
ax.plot(x, delta_x_cond_u, color=(0.8902, 0.44706, 0.1333), label='Convergence cond.')
ax.plot(x, delta_x_cond_l, color=(0.8902, 0.44706, 0.1333))
ax.set_xlabel('$\\tilde{r}$ - Estimation error')
ax.set_ylabel('$\Delta\\tilde{r}$ - Estimation error difference')
ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
ax.legend(loc='lower left', bbox_to_anchor=(0, 0))

if file is not None:
    tikzplotlib.clean_figure()
    extra_parameters = ['ylabel style={align=center}', 'font=\small']
    tikzplotlib.save(file, axis_height='\\figH',
                     axis_width='\\figW', extra_axis_parameters=extra_parameters)
else:
    plt.show()
