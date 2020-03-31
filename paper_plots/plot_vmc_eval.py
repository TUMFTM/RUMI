import pickle
from src.Visualizer import Visualizer

input_file = '../paper_results/output/VMC.p'
# output files (set to none for plotting)
output_file_1 = 'output/VMC_dist_prediction.tex'
output_file_2 = 'output/VMC_Residuals.tex'
output_file_3 = 'output/VMC_Percentage.tex'
label = 'Empiric percentage'
data_selector = lambda x: 100*x.perc_log

plot_options_1 = {
    'xlabel': 'Lateral acc. request in mps2',
    'ylabel': 'Disturbance \\\\ in mps2',
    'legend_loc': 'upper right',
    'legend_n_col': 3,
    'legend_bbox_to_anchor': (1, 1),
    'legend_fontsize': 'small',
    'y_lim': [-8, 8],
    'disable_x_axis': False,
    'add_options': None
}

plot_options_2 = {
    'xlabel': None,
    'ylabel': 'Uncertainty \\\\ bounds in mps2',
    'legend_loc': 'upper right',
    'legend_n_col': 3,
    'legend_bbox_to_anchor': (1, 1),
    'legend_fontsize': 'small',
    'y_lim': [-7, 7],
    'disable_x_axis': True,
    'add_options': None
}

plot_options_3 = {
    'xlabel': 'Samples in $10^3$',
    'ylabel': 'Data \\\\ covered in %',
    'legend_loc': 'upper right',
    'legend_n_col': 3,
    'legend_bbox_to_anchor': (1, 1),
    'legend_fontsize': 'small',
    'y_lim': [93, 103],
    'disable_x_axis': False,
    'add_options': None
}

full_data = pickle.load(open(input_file, 'rb'))
Visualizer.final_models(full_data, plot_options_1, output_file_1)
Visualizer.convergence(full_data, plot_options_2, output_file_2)
Visualizer.bounds(full_data, 'all', data_selector, plot_options_3, output_file_3)