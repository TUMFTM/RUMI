import matplotlib
import numpy as np
import tikzplotlib

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": [],                    # use latex default serif font
    "font.sans-serif": ["DejaVu Sans"],  # use a specific sans-serif font
})


class Visualizer:
    # define color maps
    colors = [(0, 0.3961, 0.7412), (0.8902, 0.44706, 0.1333), (0.6353, 0.6784, 0), 'k', 'm', 'c', 'y']

    @staticmethod
    def finalize_plot(ax, plot_options, file):
        # configure plot options
        ax.grid()
        if plot_options['xlabel'] is not None:
            ax.set_xlabel(plot_options['xlabel'])
        ax.set_ylabel(plot_options['ylabel'])
        ax.legend(loc=plot_options['legend_loc'], ncol=plot_options['legend_n_col'],
                  bbox_to_anchor=plot_options['legend_bbox_to_anchor'], fontsize=plot_options['legend_fontsize'],
                  edgecolor='inherit')
        ax.set_ylim(plot_options['y_lim'])

        # display or save to file
        if file is not None:
            # tikzplotlib.clean_figure()
            extra_parameters = ['ylabel style={align=center}', 'font=\small']
            if plot_options['disable_x_axis']:
                extra_parameters.append('xtick style={color=black, draw=none}')
                extra_parameters.append('xticklabels =\empty')
            if plot_options['add_options'] is not None:
                extra_parameters.extend(plot_options['add_options'])
            tikzplotlib.save(file, axis_height='\\figH',
                     axis_width='\\figW', extra_axis_parameters=extra_parameters)
        else:
            matplotlib.pyplot.show()

    @staticmethod
    def bounds(dataset_array, plot_type, plot_function, plot_options, file):
        skip_points = 100

        if len(Visualizer.colors) < len(dataset_array):
            print('Not enough plot colors given ... ')
        fig, ax = matplotlib.pyplot.subplots()

        # iterate via types of datasets
        for (batch, batch_label), c in zip(dataset_array, Visualizer.colors):
            # calculate minimum and maximum values
            maximum_values = np.squeeze(
                np.maximum.reduce([plot_function(dataset) for dataset in batch]))
            minimum_values = np.squeeze(
                np.minimum.reduce([plot_function(dataset) for dataset in batch]))
            # calculate x axis values
            x_ticks = np.arange(0, len(minimum_values), skip_points)/1000
            # check which plot type should be applied
            if plot_type == "var_without_bounds":
                ax.fill_between(x_ticks, minimum_values[::skip_points], maximum_values[::skip_points],
                                color=c, alpha=1.0, label=batch_label)
            if plot_type == 'var_bounds':
                ax.plot(x_ticks, minimum_values[::skip_points], color=c, label=batch_label)
                ax.plot(x_ticks, maximum_values[::skip_points], color=c)
                ax.fill_between(x_ticks, minimum_values[::skip_points], maximum_values[::skip_points],
                                color=c, alpha=1.0)
            if plot_type == 'all':
                for dataset in batch:
                    ax.plot(x_ticks, plot_function(dataset)[::skip_points], color=c, label=batch_label)

            if plot_type == 'rms':
                empiric_rms = np.sqrt(np.squeeze(
                    np.mean(np.array([(plot_function(dataset)-0.9)**2 for dataset in batch]), axis=0)))
                ax.plot(x_ticks, empiric_rms[::skip_points], c, label=batch_label)

        Visualizer.finalize_plot(ax, plot_options, file)

    @staticmethod
    def final_models(dataset_array, plot_options, file):

        if len(Visualizer.colors) < len(dataset_array):
            print('Not enough plot colors given ... ')
        fig, ax = matplotlib.pyplot.subplots()

        idx_random = np.random.randint(0, high=len(dataset_array[2][0][0].dataset.x_data), size=(1000, 1))
        ax.scatter(dataset_array[2][0][0].dataset.x_data[idx_random, 0], dataset_array[2][0][0].dataset.y_data[idx_random, 0],
                   color=(0.3451,0.3451,0.3530))
        x_eval = np.expand_dims(np.linspace(-20, 20, 300), axis=1)
        for (batch, batch_label), c in zip(dataset_array, Visualizer.colors):
            for dataset in batch:
                mean, upper_bound, lower_bound = dataset.estimator.sample(x_eval)
                ax.plot(x_eval, upper_bound, color=c, label=batch_label)
                ax.plot(x_eval, lower_bound, color=c)

        Visualizer.finalize_plot(ax, plot_options, file)

    @staticmethod
    def convergence(dataset_array, plot_options, file):
        skip_points = 100

        if len(Visualizer.colors) < len(dataset_array):
            print('Not enough plot colors given ... ')
        fig, ax = matplotlib.pyplot.subplots()

        for (batch, batch_label), c in zip(dataset_array, Visualizer.colors):
            evaluation_points = range(0, len(batch[0].res_log), skip_points)
            # only evaluate at specific points
            residuals_max = []
            residuals_min = []
            for idx in evaluation_points:
                residuals_max.append(np.max(batch[0].res_log[idx:idx+skip_points]))
                residuals_min.append(np.min(batch[0].res_log[idx:idx+skip_points]))

            x_ticks = np.arange(0, len(batch[0].res_log), skip_points)/1000
            # ax.fill_between(x_ticks, residuals_min, residuals_max, alpha=0.5, color='r')
            ax.plot(x_ticks, batch[0].ub_log[::skip_points]-batch[0].mean_log[::skip_points], color=c, label=batch_label)
            ax.plot(x_ticks, batch[0].lb_log[::skip_points]-batch[0].mean_log[::skip_points], color=c)

        Visualizer.finalize_plot(ax, plot_options, file)


