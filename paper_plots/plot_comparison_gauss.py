import pickle
from src.Visualizer import Visualizer

input_file = '../paper_results/output/disturbance_variation_gaussian.p'
# output files (set to none for plotting)
output_file = 'output/disturbance_variation_gaussian.tex'
data_selector = lambda x: 100*x.perc_log

plot_options = {
    'xlabel': None,
    'ylabel': 'Data \\\\ covered in %',
    'legend_loc': 'upper right',
    'legend_n_col': 3,
    'legend_bbox_to_anchor': (1, 1),
    'legend_fontsize': 'small',
    'y_lim': [70, 115],
    'disable_x_axis': True,
    'add_options': ['scaled x ticks = false']
}

full_data = pickle.load(open(input_file, 'rb'))
Visualizer.bounds(full_data, 'var_without_bounds', data_selector, plot_options, output_file)
