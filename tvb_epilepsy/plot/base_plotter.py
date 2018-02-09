import os
import numpy
import matplotlib
from matplotlib import pyplot
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tvb_epilepsy.base.constants.configurations import SAVE_FLAG, FIG_FORMAT, FOLDER_FIGURES, SHOW_FLAG, VERY_LARGE_SIZE
from tvb_epilepsy.base.utils.data_structures_utils import ensure_list


class BasePlotter(object):

    def __init__(self, save_flag=SAVE_FLAG, show_flag=SHOW_FLAG):
        self.save_flag = save_flag
        self.show_flag = show_flag

    def set_save_flag(self, value):
        self.save_flag = value

    def set_show_flag(self, value):
        self.show_flag = value

    def _check_show(self):
        if self.show_flag:
            # mp.use('TkAgg')
            pyplot.ion()
            pyplot.show()
        else:
            # mp.use('Agg')
            pyplot.ioff()
            pyplot.close()

    def _figure_filename(self, fig=None, figure_name=None):
        if fig is None:
            fig = pyplot.gcf()
        if figure_name is None:
            figure_name = fig.get_label()
        figure_name = figure_name.replace(": ", "_").replace(" ", "_").replace("\t", "_")
        return figure_name

    def _save_figure(self, fig=None, figure_name=None, figure_dir=FOLDER_FIGURES, figure_format=FIG_FORMAT):
        if self.save_flag:
            figure_name = self._figure_filename(fig, figure_name)
            figure_name = figure_name[:numpy.min([100, len(figure_name)])] + '.' + figure_format
            if not (os.path.isdir(figure_dir)):
                os.mkdir(figure_dir)
            pyplot.savefig(os.path.join(figure_dir, figure_name))

    def plot_vector(self, vector, labels, subplot, title, show_y_labels=True, indices_red=None, sharey=None):
        ax = pyplot.subplot(subplot, sharey=sharey)
        pyplot.title(title)
        n_vector = labels.shape[0]
        y_ticks = numpy.array(range(n_vector), dtype=numpy.int32)
        color = 'k'
        colors = numpy.repeat([color], n_vector)
        coldif = False
        if indices_red is not None:
            colors[indices_red] = 'r'
            coldif = True
        if len(vector.shape) == 1:
            ax.barh(y_ticks, vector, color=colors, align='center')
        else:
            ax.barh(y_ticks, vector[0, :], color=colors, align='center')
        # ax.invert_yaxis()
        ax.grid(True, color='grey')
        ax.set_yticks(y_ticks)
        if show_y_labels:
            region_labels = numpy.array(["%d. %s" % l for l in zip(range(n_vector), labels)])
            ax.set_yticklabels(region_labels)
            if coldif:
                labels = ax.yaxis.get_ticklabels()
                for ids in indices_red:
                    labels[ids].set_color('r')
                ax.yaxis.set_ticklabels(labels)
        else:
            ax.set_yticklabels([])
        ax.autoscale(tight=True)
        return ax

    def plot_vector_violin(self, vector, dataset, labels, subplot, title, colormap="YlOrRd", show_y_labels=True,
                           indices_red=None, sharey=None):
        ax = pyplot.subplot(subplot, sharey=sharey)
        # ax.hold(True)
        pyplot.title(title)
        n_vector = len(labels)
        y_ticks = numpy.array(range(n_vector), dtype=numpy.int32)
        # the vector plot
        coldif = False
        if len(vector) == n_vector:
            color = 'k'
            colors = numpy.repeat([color], n_vector)
            if indices_red is not None:
                colors[indices_red] = 'r'
                coldif = True
            for ii in range(n_vector):
                ax.plot(vector[ii], y_ticks[ii], '*', mfc=colors[ii], mec=colors[ii], ms=10)
        if indices_red is None:
            indices_red = []
        # the violin plot
        n_samples = dataset.shape[0]
        colormap = matplotlib.cm.ScalarMappable(cmap=pyplot.set_cmap(colormap))
        colormap = colormap.to_rgba(numpy.mean(dataset, axis=0), alpha=0.75)
        violin_parts = ax.violinplot(dataset, y_ticks, vert=False, widths=0.9,
                                     showmeans=True, showmedians=True, showextrema=True)
        violin_parts['cmeans'].set_color("k")
        violin_parts['cmins'].set_color("b")
        violin_parts['cmaxes'].set_color("b")
        violin_parts['cbars'].set_color("b")
        violin_parts['cmedians'].set_color("b")
        for ii in range(len(violin_parts['bodies'])):
            violin_parts['bodies'][ii].set_color(numpy.reshape(colormap[ii], (1, 4)))
            violin_parts['bodies'][ii]._alpha = 0.75
            violin_parts['bodies'][ii]._edgecolors = numpy.reshape(colormap[ii], (1, 4))
            violin_parts['bodies'][ii]._facecolors = numpy.reshape(colormap[ii], (1, 4))
        ax.grid(True, color='grey')
        ax.set_yticks(y_ticks)
        if show_y_labels:
            region_labels = numpy.array(["%d. %s" % l for l in zip(range(n_vector), labels)])
            ax.set_yticklabels(region_labels)
            if coldif:
                labels = ax.yaxis.get_ticklabels()
                for ids in indices_red:
                    labels[ids].set_color('r')
                ax.yaxis.set_ticklabels(labels)
        else:
            ax.set_yticklabels([])
        ax.invert_yaxis()
        ax.autoscale(tight=True)
        return ax

    def plot_regions2regions(self, adj, labels, subplot, title, show_y_labels=True, show_x_labels=True,
                             indices_red_x=None, sharey=None):
        ax = pyplot.subplot(subplot, sharey=sharey)
        pyplot.title(title)
        y_color = 'k'
        adj_size = adj.shape[0]
        y_ticks = numpy.array(range(adj_size), dtype=numpy.int32)
        if indices_red_x is None:
            indices_red_x = y_ticks
            x_ticks = indices_red_x
            x_color = y_color
        else:
            x_color = 'r'
            x_ticks = range(len(indices_red_x))
        region_labels = numpy.array(["%d. %s" % l for l in zip(range(adj_size), labels)])
        cmap = pyplot.set_cmap('autumn_r')
        img = ax.imshow(adj[indices_red_x, :].T, cmap=cmap, interpolation='none')
        ax.set_xticks(x_ticks)
        ax.grid(True, color='grey')
        if show_y_labels:
            region_labels = numpy.array(["%d. %s" % l for l in zip(range(adj_size), labels)])
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(region_labels)
            if not (x_color == y_color):
                labels = ax.yaxis.get_ticklabels()
                for idx in indices_red_x:
                    labels[idx].set_color('r')
                ax.yaxis.set_ticklabels(labels)
        else:
            ax.set_yticklabels([])
        if show_x_labels:
            ax.set_xticklabels(region_labels[indices_red_x], rotation=270, color=x_color)
        else:
            ax.set_xticklabels([])
        ax.autoscale(tight=True)
        # make a color bar
        divider = make_axes_locatable(ax)
        cax1 = divider.append_axes("right", size="5%", pad=0.05)
        pyplot.colorbar(img, cax=cax1)  # fraction=0.046, pad=0.04) #fraction=0.15, shrink=1.0
        return ax

    def _set_axis_labels(self, fig, sub, n_regions, region_labels, indices2emphasize, color='k', position='left'):
        y_ticks = range(n_regions)
        region_labels = numpy.array(["%d. %s" % l for l in zip(y_ticks, region_labels)])
        big_ax = fig.add_subplot(sub, frameon=False)
        if position == 'right':
            big_ax.yaxis.tick_right()
            big_ax.yaxis.set_label_position("right")
        big_ax.set_yticks(y_ticks)
        big_ax.set_yticklabels(region_labels, color='k')
        if not (color == 'k'):
            labels = big_ax.yaxis.get_ticklabels()
            for idx in indices2emphasize:
                labels[idx].set_color(color)
            big_ax.yaxis.set_ticklabels(labels)
        big_ax.invert_yaxis()
        big_ax.axes.get_xaxis().set_visible(False)
        big_ax.set_axis_bgcolor('none')

    def plot_in_columns(self, data_dict_list, labels, width_ratios=[], left_ax_focus_indices=[],
                        right_ax_focus_indices=[], description="", title="", figure_name=None,
                        figure_dir=FOLDER_FIGURES, figure_format=FIG_FORMAT, figsize=VERY_LARGE_SIZE, **kwargs):
        fig = pyplot.figure(title, frameon=False, figsize=figsize)
        fig.suptitle(description)
        n_subplots = len(data_dict_list)
        if width_ratios == []:
            width_ratios = numpy.ones((n_subplots,)).tolist()
        matplotlib.gridspec.GridSpec(1, n_subplots, width_ratios)
        if n_subplots < 10 and n_subplots > 0:
            subplot_ind0 = 100 + 10 * n_subplots
        else:
            raise ValueError("\nSubplots' number " + str(n_subplots) + "is not between 1 and 9!")
        n_regions = len(labels)
        subplot_ind = subplot_ind0
        ax = None
        ax0 = None
        for data_dict in data_dict_list:
            subplot_ind += 1
            data = data_dict["data"]
            focus_indices = data_dict.get("focus_indices")
            if subplot_ind == 0:
                if left_ax_focus_indices == []:
                    left_ax_focus_indices = focus_indices
            else:
                ax0 = ax
            if data_dict.get("plot_type") == "vector_violin":
                ax = self.plot_vector_violin(data, data_dict.get("data_samples", []), labels, subplot_ind,
                                             data_dict["name"],
                                             colormap=kwargs.get("colormap", "YlOrRd"), show_y_labels=False,
                                             indices_red=focus_indices, sharey=ax0)
            elif data_dict.get("plot_type") == "regions2regions":
                ax = self.plot_regions2regions(data, labels, subplot_ind, data_dict["name"], show_y_labels=False,
                                               show_x_labels=True, indices_red_x=focus_indices, sharey=ax0)
            else:
                ax = self.plot_vector(data, labels, subplot_ind, data_dict["name"], show_y_labels=False,
                                      indices_red=focus_indices, sharey=ax0)
        if right_ax_focus_indices == []:
            right_ax_focus_indices = focus_indices
        self._set_axis_labels(fig, 121, n_regions, labels, left_ax_focus_indices, 'r')
        self._set_axis_labels(fig, 122, n_regions, labels, right_ax_focus_indices, 'r', 'right')
        self._save_figure(pyplot.gcf(), figure_name, figure_dir, figure_format)
        self._check_show()
        return fig

    def plots(self, data_dict, shape=None, transpose=False, skip=0, xlabels={}, xscales={}, yscales={}, title='Plots',
              figure_name=None, figure_dir=FOLDER_FIGURES, figsize=VERY_LARGE_SIZE, figure_format=FIG_FORMAT):
        if shape is None:
            shape = (1, len(data_dict))
        fig, axes = pyplot.subplots(shape[0], shape[1], figsize=figsize)
        fig.set_label(title)
        for i, key in enumerate(data_dict.keys()):
            ind = numpy.unravel_index(i, shape)
            if transpose:
                axes[ind].plot(data_dict[key].T[skip:])
            else:
                axes[ind].plot(data_dict[key][skip:])
            axes[ind].set_xscale(xscales.get(key, "linear"))
            axes[ind].set_yscale(xscales.get(key, "linear"))
            axes[ind].set_xlabel(xlabels.get(key, ""))
            axes[ind].set_ylabel(key)
        fig.tight_layout()
        self._save_figure(fig, figure_name, figure_dir, figure_format)
        self._check_show()
        return fig, axes

    def pair_plots(self, data, keys, transpose=False, skip=0, title='Pair plots', subtitles=None, figure_name=None,
                   figure_dir=FOLDER_FIGURES, figsize=VERY_LARGE_SIZE, figure_format=FIG_FORMAT):
        if subtitles is None:
            subtitles = keys
        data = ensure_list(data)
        n = len(keys)
        fig, axes = pyplot.subplots(n, n, figsize=figsize)
        fig.set_label(title)
        for i, key_i in enumerate(keys):
            for j, key_j in enumerate(keys):
                for datai in data:
                    if transpose:
                        di = datai[key_i].T[skip:]
                    else:
                        di = datai[key_i][skip:]
                    if i == j:
                        axes[i, j].hist(di, int(numpy.round(numpy.sqrt(len(di)))), log=True)
                    else:
                        if transpose:
                            dj = datai[key_j].T[skip:]
                        else:
                            dj = datai[key_j][skip:]
                        axes[i, j].plot(dj, di, '.')
                if i == 0:
                    axes[i, j].set_title(subtitles[j])
                if j == 0:
                    axes[i, j].set_ylabel(key_i)
        fig.tight_layout()
        self._save_figure(fig, figure_name, figure_dir, figure_format)
        self._check_show()
        return fig, axes
