"""
Various plotting tools will be placed here.
"""
# TODO: make a plot function for sensitivity analysis results
import os
import matplotlib as mp
import numpy as np
from matplotlib import pyplot, gridspec
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats.mstats import zscore

from tvb_epilepsy.base.configurations import FOLDER_FIGURES
from tvb_epilepsy.base.constants import *
from tvb_epilepsy.base.utils import warning, sort_dict
from tvb_epilepsy.tvb_api.epileptor_models import *
from tvb_epilepsy.base.computations.analyzers_utils import time_spectral_analysis

try:
    #https://github.com/joferkington/mpldatacursor
    #pip install mpldatacursor
    #Not working with the MacosX graphic's backend
    from mpldatacursor import HighlightingDataCursor #datacursor
    MOUSEHOOVER = True
except:
    warning("\nNo mpldatacursor module found! MOUSEHOOVER will not be available.")
    MOUSEHOOVER = False


def check_show(show_flag=SHOW_FLAG):
    if show_flag:
        # mp.use('TkAgg')
        pyplot.ion()
        pyplot.show()
    else:
        # mp.use('Agg')
        pyplot.ioff()
        pyplot.close()


def save_figure(save_flag=SAVE_FLAG, figure_dir=FOLDER_FIGURES, figure_format=FIG_FORMAT, figure_name='figure'):
    if save_flag:
        if not (os.path.isdir(figure_dir)):
            os.mkdir(figure_dir)
        figure_name = figure_name.replace(" ", "_").replace("\t", "_") + '.' + figure_format
        pyplot.savefig(os.path.join(figure_dir, figure_name))


def plot_vector(vector, labels, subplot, title, show_y_labels=True, indices_red=None, sharey=None):
    ax = pyplot.subplot(subplot, sharey=sharey)
    pyplot.title(title)
    n_vector = labels.shape[0]

    y_ticks = np.array(range(n_vector), dtype=np.int32)
    color = 'k'
    colors = np.repeat([color], n_vector)
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
        region_labels = np.array(["%d. %s" % l for l in zip(range(n_vector), labels)])
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

def plot_vector_violin(vector, dataset, labels, subplot, title, colormap="YlOrRd", show_y_labels=True,
                       indices_red=None, sharey=None):
    ax = pyplot.subplot(subplot, sharey=sharey)
    #ax.hold(True)
    pyplot.title(title)
    n_vector = labels.shape[0]
    y_ticks = np.array(range(n_vector), dtype=np.int32)

    # the vector plot
    coldif = False
    if len(vector) == n_vector:
        color = 'k'
        colors = np.repeat([color], n_vector)
        if indices_red is not None:
            colors[indices_red] = 'r'
            coldif = True
        for ii in range(n_vector):
            ax.plot(vector[ii], y_ticks[ii], '*', mfc=colors[ii], mec=colors[ii], ms=5)

    # the violin plot
    n_samples = dataset.shape[0]
    colormap = mp.cm.ScalarMappable(cmap=pyplot.set_cmap(colormap))
    colormap = colormap.to_rgba(np.mean(dataset, axis=0), alpha=0.75)
    violin_parts = ax.violinplot(dataset, y_ticks, vert=False, widths=0.9,
                                 showmeans=True, showmedians=True, showextrema=True)
    violin_parts['cmeans'].set_color("k")
    violin_parts['cmins'].set_color("b")
    violin_parts['cmaxes'].set_color("b")
    violin_parts['cbars'].set_color("b")
    violin_parts['cmedians'].set_color("b")
    for ii in range(len(violin_parts['bodies'])):
        violin_parts['bodies'][ii].set_color(np.reshape(colormap[ii], (1,4)))
        violin_parts['bodies'][ii]._alpha = 0.75
        violin_parts['bodies'][ii]._edgecolors = np.reshape(colormap[ii], (1,4))
        violin_parts['bodies'][ii]._facecolors = np.reshape(colormap[ii], (1,4))

    # ax.invert_yaxis()
    ax.grid(True, color='grey')
    ax.set_yticks(y_ticks)
    if show_y_labels:
        region_labels = np.array(["%d. %s" % l for l in zip(range(n_vector), labels)])
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

def plot_regions2regions(adj, labels, subplot, title, show_y_labels=True, show_x_labels=True,
                         indices_red_x=None, sharey=None):
    ax = pyplot.subplot(subplot, sharey=sharey)
    pyplot.title(title)

    y_color = 'k'
    adj_size = adj.shape[0]
    y_ticks = np.array(range(adj_size), dtype=np.int32)
    if indices_red_x is None:
        indices_red_x = y_ticks
        x_ticks = indices_red_x
        x_color = y_color
    else:
        x_color = 'r'
        x_ticks = range(len(indices_red_x))
    region_labels = np.array(["%d. %s" % l for l in zip(range(adj_size), labels)])
    cmap = pyplot.set_cmap('autumn_r')
    img = ax.imshow(adj[indices_red_x, :].T, cmap=cmap, interpolation='none')
    ax.set_xticks(x_ticks)
    ax.grid(True, color='grey')

    if show_y_labels:
        region_labels = np.array(["%d. %s" % l for l in zip(range(adj_size), labels)])
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


def plot_in_columns(data_dict_list, labels, width_ratios=[], left_ax_focus_indices=[], right_ax_focus_indices=[],
                    description="", title="", figure_name='', show_flag=False, save_flag=True,
                    figure_dir=FOLDER_FIGURES, figure_format=FIG_FORMAT, figsize=VERY_LARGE_SIZE, **kwargs):

    fig = pyplot.figure(title, frameon=False, figsize=figsize)
    fig.suptitle(description)
    n_subplots = len(data_dict_list)
    if width_ratios == []:
        width_rations = np.ones((n_subplots, )).tolist()

    mp.gridspec.GridSpec(1, n_subplots, width_ratios)

    if n_subplots < 10 and n_subplots > 0:
        subplot_ind0 = 100 + 10*n_subplots
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
            ax = plot_vector_violin(data, data_dict.get("data_samples", []), labels, subplot_ind, data_dict["name"],
                                    colormap=kwargs.get("colormap", "YlOrRd"), show_y_labels=False,
                                    indices_red=focus_indices, sharey=ax0)

        elif data_dict.get("plot_type") == "regions2regions":
            ax = plot_regions2regions(data, labels, subplot_ind, data_dict["name"], show_y_labels=False,
                                      show_x_labels=True, indices_red_x=focus_indices, sharey=ax0)
        else:
            ax = plot_vector(data, labels, subplot_ind, data_dict["name"], show_y_labels=False,
                             indices_red=focus_indices, sharey=ax0)

    if right_ax_focus_indices == []:
        right_ax_focus_indices = focus_indices

    _set_axis_labels(fig, 121, n_regions, labels, left_ax_focus_indices, 'r')
    _set_axis_labels(fig, 122, n_regions, labels, right_ax_focus_indices, 'r', 'right')

    if figure_name == '':
        figure_name = fig.get_label().replace(": ", "_").replace(" ", "_").replace("\t", "_")

    save_figure(save_flag, figure_dir=figure_dir, figure_format=figure_format, figure_name=figure_name)
    check_show(show_flag)

    return fig


def _set_axis_labels(fig, sub, n_regions, region_labels, indices2emphasize, color='k', position='left'):
    y_ticks = range(n_regions)
    region_labels = np.array(["%d. %s" % l for l in zip(y_ticks, region_labels)])
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


def plot_timeseries(time, data_dict, time_units="ms", special_idx=None, title='Time Series', figure_name='TimeSeries',
                    labels=None, show_flag=SHOW_FLAG, save_flag=False, figure_dir=FOLDER_FIGURES,
                    figure_format=FIG_FORMAT, figsize=LARGE_SIZE):

    pyplot.figure(title, figsize=figsize)
    no_rows = len(data_dict)
    lines = []
    for i, subtitle in enumerate(data_dict):
        ax = pyplot.subplot(no_rows, 1, i + 1)
        pyplot.hold(True)
        if i == 0:
            pyplot.title(title)
        data = data_dict[subtitle]
        nTS = data.shape[1]
        if labels is None:
            labels = np.array(range(nTS)).astype(str)
        lines.append([])
        if special_idx is None:
            for iTS in range(nTS):
                line, = pyplot.plot(time, data[:, iTS], 'k', alpha=0.3, label=labels[iTS])
                lines[i].append(line)
        else:
            mask = np.array(range(nTS))
            mask = np.delete(mask,special_idx)
            for iTS in special_idx:
                line, = pyplot.plot(time, data[:, iTS], 'r', alpha=0.7, label=labels[iTS])
                lines[i].append(line)
            for iTS in mask:
                line, = pyplot.plot(time, data[:, iTS], 'k', alpha=0.3, label=labels[iTS])
                lines[i].append(line)
        pyplot.ylabel(subtitle)
        ax.set_autoscalex_on(False)
        ax.set_xlim([time[0], time[-1]])
        if MOUSEHOOVER:
            #datacursor( lines[i], formatter='{label}'.format, bbox=dict(fc='white'),
            #           arrowprops=dict(arrowstyle='simple', fc='white', alpha=0.5) )    #hover=True
            HighlightingDataCursor(lines[i], formatter='{label}'.format, bbox=dict(fc='white'),
                                   arrowprops=dict(arrowstyle='simple', fc='white', alpha=0.5) )
    pyplot.xlabel("Time (" + time_units + ")")

    fig = pyplot.gcf()
    if len(fig.get_label())==0:
        fig.set_label(figure_name)
    else:
        figure_name = fig.get_label().replace(": ", "_").replace(" ", "_").replace("\t", "_")

    save_figure(save_flag, figure_dir=figure_dir, figure_format=figure_format, figure_name=figure_name)
    check_show(show_flag)


def plot_raster(time, data_dict, time_units="ms", special_idx=None, title='Time Series', subtitles=[], offset=3.0,
                figure_name='TimeSeries', labels=None, show_flag=SHOW_FLAG, save_flag=False, figure_dir=FOLDER_FIGURES,
                figure_format=FIG_FORMAT, figsize=VERY_LARGE_SIZE):
    pyplot.figure(title, figsize=figsize)
    no_rows = len(data_dict)
    lines = []

    for i, var in enumerate(data_dict):
        ax = pyplot.subplot(1, no_rows, i + 1)
        pyplot.hold(True)
        if len(subtitles) > i:
            pyplot.title(subtitles[i])
        data = data_dict[var]
        data = zscore(data, axis=None)
        nTS = data.shape[1]
        ticks = (offset*np.array([range(nTS)])).tolist()
        if labels is None:
            labels = np.array(range(nTS)).astype(str)
        lines.append([])
        if special_idx is None:
            for iTS in range(nTS):
                line, = pyplot.plot(time, -data[:,iTS]+offset*iTS, 'k', label = labels[iTS])
                lines[i].append(line)
        else:
            mask = np.array(range(nTS))
            mask = np.delete(mask,special_idx)
            for iTS in special_idx:
                line, = pyplot.plot(time, -data[:, iTS]+offset*iTS, 'r', label = labels[iTS])
                lines[i].append(line)
            for iTS in mask:
                line, = pyplot.plot(time, -data[:, iTS]+offset*iTS, 'k', label = labels[iTS])
                lines[i].append(line)
        pyplot.ylabel(var)
        ax.set_autoscalex_on(False)
        ax.set_xlim([time[0], time[-1]])
        # ax.set_yticks(ticks)
        # ax.set_yticklabels(labels)
        ax.invert_yaxis()
        if MOUSEHOOVER:
            #datacursor( lines[i], formatter='{label}'.format, bbox=dict(fc='white'),
            #           arrowprops=dict(arrowstyle='simple', fc='white', alpha=0.5) )    #hover=True
            HighlightingDataCursor(lines[i], formatter='{label}'.format, bbox=dict(fc='white'),
                                   arrowprops=dict(arrowstyle='simple', fc='white', alpha=0.5) )
    pyplot.xlabel("Time (" + time_units + ")")
    fig = pyplot.gcf()
    fig.suptitle(title)
    if len(fig.get_label())==0:
        fig.set_label(figure_name)
    else:
        figure_name = fig.get_label().replace(": ", "_").replace(" ", "_").replace("\t", "_")
    save_figure(save_flag, figure_dir=figure_dir, figure_format=figure_format, figure_name=figure_name)
    check_show(show_flag)


def plot_trajectories(data_dict, special_idx=None, title='State space trajectories', figure_name='Trajectories',
                      labels=None, show_flag=SHOW_FLAG, save_flag=SAVE_FLAG, figure_dir=FOLDER_FIGURES,
                      figure_format=FIG_FORMAT, figsize=LARGE_SIZE):

    pyplot.figure(title, figsize=figsize)
    ax = pyplot.subplot(111)
    pyplot.hold(True)
    no_dims = len(data_dict)
    if no_dims>2:
        ax = pyplot.subplot(111,projection='3d')
    else:
        ax = pyplot.subplot(111)
    lines = []
    ax_labels=[]
    data=[]
    for i, var in enumerate(data_dict):
        if i == 0:
            pyplot.title(title)
        ax_labels.append(var)
        data.append(data_dict[var])
    nTS = data[0].shape[1]
    if labels is None:
        labels = np.array(range(nTS)).astype(str)
    lines.append([])
    if special_idx is None:
        for iTS in range(nTS):
            if no_dims>2:
                line, = pyplot.plot(data[0][:,iTS], data[1][:,iTS], data[2][:,iTS],  'k', alpha=0.3,
                                       label=labels[iTS])
            else:
                line, = pyplot.plot(data[0][:,iTS], data[1][:,iTS], 'k', alpha=0.3, label=labels[iTS])
            lines.append(line)
    else:
        mask = np.array(range(nTS))
        mask = np.delete(mask,special_idx)
        for iTS in special_idx:
            if no_dims>2:
                line, = pyplot.plot(data[0][:,iTS], data[1][:,iTS], data[2][:,iTS], 'r', alpha=0.7,
                                       label=labels[iTS])
            else:
                line, = pyplot.plot(data[0][:,iTS], data[1][:,iTS], 'r', alpha=0.7, label = labels[iTS])
            lines.append(line)
        for iTS in mask:
            if no_dims>2:
                line, = pyplot.plot(data[0][:,iTS], data[1][:,iTS], data[2][:,iTS], 'k', alpha=0.3,
                                       label=labels[iTS])
            else:
                line, = pyplot.plot(data[0][:,iTS], data[1][:,iTS], 'k', alpha=0.3, label=labels[iTS])
            lines.append(line)
    pyplot.xlabel(ax_labels[0])
    pyplot.ylabel(ax_labels[1])
    if no_dims>2:
        pyplot.ylabel(ax_labels[2])
    if MOUSEHOOVER:
        #datacursor( lines[0], formatter='{label}'.format, bbox=dict(fc='white'),
        #           arrowprops=dict(arrowstyle='simple', fc='white', alpha=0.5) )    #hover=True
        HighlightingDataCursor(lines[0], formatter='{label}'.format, bbox=dict(fc='white'),
                                   arrowprops=dict(arrowstyle='simple', fc='white', alpha=0.5) )

    fig = pyplot.gcf()
    if len(fig.get_label())==0:
        fig.set_label(figure_name)
    else:
        figure_name = fig.get_label().replace(": ", "_").replace(" ", "_").replace("\t", "_")

    save_figure(save_flag, figure_dir=figure_dir, figure_format=figure_format, figure_name=figure_name)
    check_show(show_flag)


def plot_spectral_analysis_raster(time, data, time_units="ms", freq=None, special_idx=None, title='Spectral Analysis',
                                  figure_name='Spectral Analysis', labels=None,
                                  show_flag=SHOW_FLAG, save_flag=SAVE_FLAG, figure_dir=FOLDER_FIGURES,
                                  figure_format=FIG_FORMAT, figsize=VERY_LARGE_SIZE, **kwargs):

    if time_units in ("ms", "msec"):
        fs=1000.0
    else:
        fs=1.0
    fs = fs/np.mean(np.diff(time))

    if special_idx is not None:
        data = data[:, special_idx]
        if labels is not None:
            labels = np.array(labels)[special_idx]

    nS = data.shape[1]

    if labels is None:
        labels = np.array(range(nS)).astype(str)

    log_norm = kwargs.get("log_norm", False)
    mode = kwargs.get("mode", "psd")
    psd_label = mode
    if log_norm:
        psd_label = "log" + psd_label
    stf, time, freq, psd = time_spectral_analysis(data, fs,
                                                  freq=freq,
                                                  mode=mode,
                                                  nfft=kwargs.get("nfft"),
                                                  window=kwargs.get("window", 'hanning'),
                                                  nperseg=kwargs.get("nperseg", int(np.round(fs/2))),
                                                  detrend=kwargs.get("detrend", 'constant'),
                                                  noverlap=kwargs.get("noverlap"),
                                                  f_low=kwargs.get("f_low", 10.0),
                                                  log_scale=kwargs.get("log_scale", False))

    min_val = np.min(stf.flatten())
    max_val = np.max(stf.flatten())
    if nS > 2:
        figsize = VERY_LARGE_SIZE

    fig = pyplot.figure(title, figsize=figsize)
    fig.suptitle(title)
    gs = gridspec.GridSpec(nS, 23)
    ax = np.empty((nS,2), dtype="O")
    img = np.empty((nS, ), dtype="O")
    line = np.empty((nS,), dtype="O")

    for iS in range(nS-1, -1, -1):

        if iS < nS-1:
            ax[iS, 0] = pyplot.subplot(gs[iS, :20], sharex=ax[iS, 0])
            ax[iS, 1] = pyplot.subplot(gs[iS, 20:22], sharex=ax[iS, 1], sharey=ax[iS, 0])
        else:
            ax[iS, 0] = pyplot.subplot(gs[iS, :20])
            ax[iS, 1] = pyplot.subplot(gs[iS, 20:22], sharey=ax[iS, 0])

        img[iS] = ax[iS, 0].imshow(np.squeeze(stf[:, :, iS]).T, cmap=pyplot.set_cmap('jet'), interpolation='none',
                                   norm=Normalize(vmin=min_val, vmax=max_val), aspect='auto', origin='lower',
                                   extent=(time.min(),time.max(), freq.min(), freq.max()))
        # img[iS].clim(min_val, max_val)
        ax[iS, 0].set_title(labels[iS])
        ax[iS, 0].set_ylabel("Frequency (Hz)")

        line[iS] = ax[iS, 1].plot(psd[:, iS], freq, 'k', label=labels[iS])
        pyplot.setp(ax[iS, 1].get_yticklabels(), visible=False)
        # ax[iS, 1].yaxis.tick_right()
        # ax[iS, 1].yaxis.set_ticks_position('both')

        if iS == (nS-1):
            ax[iS, 0].set_xlabel("Time (" + time_units + ")")

            ax[iS, 1].set_xlabel(psd_label)
        else:
            pyplot.setp(ax[iS, 0].get_xticklabels(), visible=False)
        pyplot.setp(ax[iS, 1].get_xticklabels(), visible=False)
        ax[iS, 0].autoscale(tight=True)
        ax[iS, 1].autoscale(tight=True)

    # make a color bar
    cax = pyplot.subplot(gs[:, 22])
    pyplot.colorbar(img[0], cax=pyplot.subplot(gs[:, 22]))  # fraction=0.046, pad=0.04) #fraction=0.15, shrink=1.0
    cax.set_title(psd_label)
    # fig = pyplot.gcf()
    if len(fig.get_label()) == 0:
        fig.set_label(figure_name)
    else:
        figure_name = fig.get_label().replace(": ", "_").replace(" ", "_").replace("\t", "_")

    save_figure(save_flag, figure_dir=figure_dir, figure_format=figure_format, figure_name=figure_name)
    check_show(show_flag)

    return fig, ax, img, line, time, freq, stf, psd


def plot_sim_results(model, seizure_indices, hyp_name, head, res, sensorsSEEG, hpf_flag=False,
                     trajectories_plot=False, spectral_raster_plot=False,
                     save_flag=SAVE_FLAG, show_flag=SHOW_FLAG, figure_dir=FOLDER_FIGURES, figure_format=FIG_FORMAT,
                     **kwargs):

    if isinstance(model, EpileptorDP2D):
        plot_timeseries(res['time'], {'x1': res['x1'], 'z(t)': res['z']}, time_units=res.get('time_units', "ms"),
                        special_idx=seizure_indices, title=hyp_name + ": Simulated TAVG",
                        save_flag=save_flag, show_flag=show_flag, figure_dir=figure_dir, figure_format=figure_format,
                        labels=head.connectivity.region_labels, figsize=VERY_LARGE_SIZE)
        plot_raster(res['time'], {'x1': res['x1']},
                    time_units=res.get('time_units', "ms"), special_idx=seizure_indices,
                    title=hyp_name + ": Simulated x1 rasterplot", offset=5.0, labels=head.connectivity.region_labels,
                    save_flag=save_flag, show_flag=show_flag, figure_dir=figure_dir, figure_format=figure_format,
                    figsize=VERY_LARGE_SIZE)

    else:
        plot_timeseries(res['time'], {'LFP(t)': res['lfp'], 'z(t)': res['z']}, time_units=res.get('time_units', "ms"),
                        special_idx=seizure_indices, title=hyp_name + ": Simulated LFP-z",
                        save_flag=save_flag, show_flag=show_flag, figure_dir=figure_dir, figure_format=figure_format,
                        labels=head.connectivity.region_labels, figsize=VERY_LARGE_SIZE)
        plot_timeseries(res['time'], {'x1(t)': res['x1'], 'y1(t)': res['y1']},time_units=res.get('time_units', "ms"),
                        special_idx=seizure_indices, title=hyp_name + ": Simulated pop1",
                        save_flag=save_flag, show_flag=show_flag, figure_dir=figure_dir, figure_format=figure_format,
                        labels=head.connectivity.region_labels, figsize=VERY_LARGE_SIZE)
        plot_timeseries(res['time'], {'x2(t)': res['x2'], 'y2(t)': res['y2'], 'g(t)': res['g']},
                        time_units=res.get('time_units', "ms"), special_idx=seizure_indices,
                        title=hyp_name + ": Simulated pop2-g",
                        save_flag=save_flag, show_flag=show_flag, figure_dir=figure_dir, figure_format=figure_format,
                        labels=head.connectivity.region_labels, figsize=VERY_LARGE_SIZE)
        start_plot = int(np.round(0.01 * res['lfp'].shape[0]))
        plot_raster(res['time'][start_plot:], {'lfp': res['lfp'][start_plot:, :]},
                    time_units=res.get('time_units', "ms"), special_idx=seizure_indices,
                    title=hyp_name + ": Simulated LFP rasterplot", offset=10.0, labels=head.connectivity.region_labels,
                    save_flag=save_flag, show_flag=show_flag, figure_dir=figure_dir, figure_format=figure_format,
                    figsize=VERY_LARGE_SIZE)

    if isinstance(model, EpileptorDPrealistic):
        plot_timeseries(res['time'], {'1/(1+exp(-10(z-3.03))': 1 / (1 + np.exp(-10 * (res['z'] - 3.03))),
                                      'slope': res['slope_t'], 'Iext2': res['Iext2_t']},
                        time_units=res.get('time_units', "ms"), special_idx=seizure_indices,
                        title=hyp_name + ": Simulated controlled parameters", labels=head.connectivity.region_labels,
                        save_flag=save_flag, show_flag=show_flag, figure_dir=figure_dir, figure_format=figure_format,
                        figsize=VERY_LARGE_SIZE)
        plot_timeseries(res['time'], {'x0_values': res['x0_t'], 'Iext1':  res['Iext1_t'], 'K': res['K_t']},
                        time_units=res.get('time_units', "ms"), special_idx=seizure_indices,
                        title=hyp_name + ": Simulated parameters", labels=head.connectivity.region_labels,
                        save_flag=save_flag, show_flag=show_flag, figure_dir=figure_dir, figure_format=figure_format,
                        figsize=VERY_LARGE_SIZE)

    if trajectories_plot:
        plot_trajectories({'x1': res['x1'], 'z(t)': res['z']}, special_idx=seizure_indices,
                          title=hyp_name + ': State space trajectories', figure_name=hyp_name+'StateSpaceTrajectories',
                          labels=head.connectivity.region_labels, show_flag=show_flag, save_flag=save_flag,
                          figure_dir=FOLDER_FIGURES, figure_format=FIG_FORMAT, figsize=LARGE_SIZE)

    if spectral_raster_plot is "lfp":
        plot_spectral_analysis_raster(res["time"], res['lfp'], time_units=res.get('time_units', "ms"),
                                      freq=None, special_idx=seizure_indices,
                                      title=hyp_name + ": Spectral Analysis",
                                      labels=head.connectivity.region_labels,
                                      show_flag=show_flag, save_flag=save_flag, figure_dir=figure_dir,
                                      figure_format=figure_format, figsize=LARGE_SIZE, **kwargs)

    for i in range(len(sensorsSEEG)):
        if hpf_flag:
            title = hyp_name + ": Simulated high pass filtered SEEG" + str(i) + " raster plot"
            start_plot = int(np.round(0.01 * res['SEEG' + str(i)].shape[0]))
        else:
            title = hyp_name + ": Simulated SEEG" + str(i) + " raster plot"
            start_plot = 0
        plot_raster(res['time'][start_plot:], {'SEEG': res['SEEG'+str(i)][start_plot:, :]},
                    time_units=res.get('time_units', "ms"), title=title,
                    offset=1.0, save_flag=save_flag, show_flag=show_flag, figure_dir=figure_dir,
                    figure_format=figure_format, labels=sensorsSEEG[i].labels, figsize=VERY_LARGE_SIZE)


def plot_fit_results(hyp_name, head, res, data, active_regions, time=None, seizure_indices=None,
                     trajectories_plot=False, save_flag=SAVE_FLAG, show_flag=SHOW_FLAG, figure_dir=FOLDER_FIGURES,
                     figure_format=FIG_FORMAT,
                     **kwargs):

    if time is None:
        time = np.array(range(data['signals'].shape[0]))

    time = time.flatten()

    plot_raster(time, sort_dict({'observation signals':  data['signals'],
                                 'observation signals fit':  res['fit_signals'],
                                 'x1': res["x"].T, 'z': res["z"].T}),
                special_idx=seizure_indices, time_units=res.get('time_units', "ms"),
                title=hyp_name + ": Observation signals vs fit rasterplot",
                subtitles=['observation signals ' +
                                '\ndynamic noise prior: sig = ' + str(data["sig_hi"]/2) +
                                '\nobservation noise prior: eps =  ' + str(data["eps_hi"]/2),
                           'observation signals fit',
                           'hidden state x1' + '\ndynamic noise fit sig = : ' + str(res["sig"]) +
                                '\nobservation noise fit eps = : ' + str(res["eps"]),
                           'hidden state z'],  offset=3.0,
                figure_name=hyp_name + 'ObservationSignals_vs_FitHiddenStates_rasterplot',
                labels=None, save_flag=save_flag, show_flag=show_flag, figure_dir=figure_dir,
                figure_format=figure_format, figsize=VERY_LARGE_SIZE)

    if trajectories_plot:
        plot_trajectories({'x1': res['x'].T, 'z(t)': res['z'].T}, special_idx=seizure_indices,
                          title=hyp_name+': Fit state space trajectories' + "\n x0 fit: " + str(res["x0"]),
                          figure_name=hyp_name+'FitHiddenStateTrajectories',
                          labels=head.connectivity.region_labels, show_flag=show_flag, save_flag=save_flag,
                          figure_dir=FOLDER_FIGURES, figure_format=FIG_FORMAT, figsize=LARGE_SIZE)

    # plot connectivity
    conn_figure_name ="Structural and Effective Connectivity"
    pyplot.figure(conn_figure_name, VERY_LARGE_SIZE)
    # plot_regions2regions(conn.weights, conn.region_labels, 121, "weights")
    plot_regions2regions(data['SC'], head.connectivity.region_labels[active_regions], 121, "Structural Connectivity" +
                         "\nglobal scaling prior: K = " + str(data["K_u"] * data["K_v"]))
    plot_regions2regions(res['FC'], head.connectivity.region_labels[active_regions], 122, "Effective Connectivity"  +
                         "\nglobal scaling fit: K = " + str(res["K"]))
    if save_flag:
        save_figure(figure_dir=figure_dir, figure_format=figure_format,
                    figure_name=conn_figure_name.replace(" ", "_").replace("\t", "_"))
    check_show(show_flag=show_flag)



    # def plot_head(head, show_flag=SHOW_FLAG, save_flag=SAVE_FLAG, figure_dir=FOLDER_FIGURES, figure_format=FIG_FORMAT,
#               figsize=LARGE_SIZE):
#     plot_connectivity(head.connectivity, show_flag=show_flag, save_flag=save_flag, figure_dir=figure_dir,
#                       figure_format=figure_format, figsize=figsize)
#
#     plot_head_stats(head.connectivity, show_flag=show_flag, save_flag=save_flag, figure_dir=figure_dir,
#                     figure_format=figure_format)
#
#     count = _show_projections_dict(head, head.sensorsEEG, 1, show_flag=show_flag,
#                                    save_flag=save_flag, figure_dir=figure_dir, figure_format=figure_format)
#     count = _show_projections_dict(head, head.sensorsSEEG, count, show_flag=show_flag,
#                                    save_flag=save_flag, figure_dir=figure_dir, figure_format=figure_format)
#     _show_projections_dict(head, head.sensorsMEG, count, show_flag=show_flag,
#                            save_flag=save_flag, figure_dir=figure_dir, figure_format=figure_format)
#
#
# def plot_connectivity(conn, show_flag=SHOW_FLAG, save_flag=SAVE_FLAG, figure_dir=FOLDER_FIGURES,
#                       figure_format=FIG_FORMAT, figure_name='Connectivity ', figsize=LARGE_SIZE):
#
#     pyplot.figure(figure_name + str(conn.number_of_regions), figsize)
#     #plot_regions2regions(conn.weights, conn.region_labels, 121, "weights")
#     plot_regions2regions(conn.normalized_weights, conn.region_labels, 121, "normalised weights")
#     plot_regions2regions(conn.tract_lengths, conn.region_labels, 122, "tract lengths")
#
#     _save_figure(save_flag, figure_dir=figure_dir, figure_format=figure_format, figure_name=figure_name)
#     _check_show(show_flag=show_flag)
#
#
# def plot_head_stats(conn, show_flag=SHOW_FLAG, save_flag=SAVE_FLAG, figure_dir=FOLDER_FIGURES, figure_format=FIG_FORMAT,
#                     figure_name='HeadStats '):
#     pyplot.figure("Head stats " + str(conn.number_of_regions), figsize=LARGE_SIZE)
#     ax = plot_vector(calculate_in_degree(conn.normalized_weights), conn.region_labels, 121, "w in-degree")
#     ax.invert_yaxis()
#     if conn.areas is not None:
#         ax = plot_vector(conn.areas, conn.region_labels, 122, "region areas")
#         ax.invert_yaxis()
#     _save_figure(save_flag, figure_dir=figure_dir, figure_format=figure_format, figure_name=figure_name)
#     _check_show(show_flag=show_flag)
#
#
# def _show_projections_dict(connectivity, sensors_dict, current_count=1, show_flag=SHOW_FLAG,
#                            save_flag=SAVE_FLAG, figure_dir=FOLDER_FIGURES, figure_format=FIG_FORMAT):
#     for sensors, projection in sensors_dict.iteritems():
#         if projection is None:
#             continue
#         _plot_projection(projection, connectivity, sensors,
#                          title=str(current_count) + " - " + sensors.s_type + " - Projection",
#                          show_flag=show_flag, save_flag=save_flag, figure_dir=figure_dir, figure_format=figure_format)
#
#         current_count += 1
#     return current_count
#
#
# def _plot_projection(proj, connectivity, sensors, figure=None, title="Projection",
#                      y_labels=1, x_labels=1, x_ticks=None, y_ticks=None, show_flag=SHOW_FLAG,
#                      save_flag=SAVE_FLAG, figure_dir=FOLDER_FIGURES, figure_format=FIG_FORMAT, figure_name=''):
#     if not (isinstance(figure, pyplot.Figure)):
#         figure = pyplot.figure(title, figsize=LARGE_SIZE)
#
#     n_sensors = sensors.number_of_sensors
#     n_regions = connectivity.number_of_regions
#     if x_ticks is None:
#         x_ticks = np.array(range(n_sensors), dtype=np.int32)
#     if y_ticks is None:
#         y_ticks = np.array(range(n_regions), dtype=np.int32)
#
#     cmap = pyplot.set_cmap('autumn_r')
#     img = pyplot.imshow(proj[x_ticks][:, y_ticks].T, cmap=cmap, interpolation='none')
#     pyplot.grid(True, color='black')
#     if y_labels > 0:
#         region_labels = np.array(["%d. %s" % l for l in zip(range(n_regions), connectivity.region_labels)])
#         pyplot.yticks(y_ticks, region_labels[y_ticks])
#     else:
#         pyplot.yticks(y_ticks)
#     if x_labels > 0:
#         sensor_labels = np.array(["%d. %s" % l for l in zip(range(n_sensors), sensors.labels)])
#         pyplot.xticks(x_ticks, sensor_labels[x_ticks], rotation=90)
#     else:
#         pyplot.xticks(x_ticks)
#
#     ax = figure.get_axes()[0]
#     ax.autoscale(tight=True)
#     pyplot.title(title)
#
#     divider = make_axes_locatable(ax)
#     cax1 = divider.append_axes("right", size="5%", pad=0.05)
#     pyplot.colorbar(img, cax=cax1)  # fraction=0.046, pad=0.04) #fraction=0.15, shrink=1.0
#
#     _save_figure(save_flag, figure_dir=figure_dir, figure_format=figure_format, figure_name=title)
#     _check_show(show_flag)
#
#     return figure


# def plot_nullclines_eq(model_configuration, region_labels, special_idx=None, model="2d", zmode=np.array("lin"),
#                        x0ne=X0_DEF, x0e=X0_CR_DEF, figure_name='Nullclines and equilibria', show_flag=SHOW_FLAG,
#                        save_flag=SAVE_FLAG, figure_dir=FOLDER_FIGURES, figure_format=FIG_FORMAT, figsize=SMALL_SIZE):
#
#     add_name = " " + "Epileptor " + model + " z-" + str(zmode)
#     figure_name = figure_name + add_name
#
#     # Fixed parameters for all regions:
#     x1eq = np.mean(model_configuration.x1EQ)
#     yc = np.mean(model_configuration.yc)
#     Iext1 = np.mean(model_configuration.Iext1)
#     x0cr = np.mean(model_configuration.x0cr)  # Critical x0_values
#     r = np.mean(model_configuration.rx0)
#     # The point of the linear approximation (1st order Taylor expansion)
#     x1LIN = def_x1lin(X1_DEF, X1_EQ_CR_DEF, len(region_labels))
#     x1SQ = X1_EQ_CR_DEF
#     x1lin0 = np.mean(x1LIN)
#     # The point of the square (parabolic) approximation (2nd order Taylor expansion)
#     x1sq0 = np.mean(x1SQ)
#     if model != "2d" or zmode != np.array("lin"):
#         x0cr, r = calc_x0cr_r(yc, Iext1, zmode=zmode, x1_rest=X1_DEF, x1_cr=X1_EQ_CR_DEF, x0def=X0_DEF,
#                               x0cr_def=X0_CR_DEF)
#
#     # Lines:
#
#     # x1 nullcline:
#     x1 = np.linspace(-2.0, 2.0 / 3.0, 100)
#     if model == "2d":
#         y1 = yc
#         b = -2.0
#     else:
#         y1 = calc_eq_y1(x1, yc, d=5.0)
#         b = 3.0
#     zX1 = calc_fx1(x1, z=0, y1=y1, Iext1=Iext1, x1_neg=None, model=model,
#                    b=b)  # yc + Iext1 - x1 ** 3 - 2.0 * x1 ** 2
#     # approximations:
#     # linear:
#     x1lin = np.linspace(-5.5 / 3.0, -3.5 / 3, 30)
#     # x1 nullcline after linear approximation
#     zX1lin = calc_fx1_2d_taylor(x1lin, x1lin0, z=0, y1=yc, Iext1=Iext1, slope=0.0, a=1.0, b=-2.0, tau1=1.0,
#                                 x1_neg=None,
#                                 order=2)  # yc + Iext1 + 2.0 * x1lin0 ** 3 + 2.0 * x1lin0 ** 2 - \
#     # (3.0 * x1lin0 ** 2 + 4.0 * x1lin0) * x1lin  # x1 nullcline after linear approximation
#     # center point without approximation:
#     # zlin0 = yc + Iext1 - x1lin0 ** 3 - 2.0 * x1lin0 ** 2
#     # square:
#     x1sq = np.linspace(-5.0 / 3, -1.0, 30)
#     # x1 nullcline after parabolic approximation
#     zX1sq = calc_fx1_2d_taylor(x1sq, x1sq0, z=0, y1=yc, Iext1=Iext1, slope=0.0, a=1.0, b=-2.0, tau1=1.0,
#                                x1_neg=None, order=3,
#                                shape=x1sq.shape)  # + 2.0 * x1sq ** 2 + 16.0 * x1sq / 3.0 + yc + Iext1 + 64.0 / 27.0
#     # center point (critical equilibrium point) without approximation:
#     # zsq0 = yc + Iext1 - x1sq0 ** 3 - 2.0 * x1sq0 ** 2
#     if model == "2d":
#         # z nullcline:
#         zZe = calc_fz(x1, z=0.0, x0_values=x0e, x0cr=x0cr, r=r, zmode=zmode)  # for epileptogenic regions
#         zZne = calc_fz(x1, z=0.0, x0_values=x0ne, x0cr=x0cr, r=r, zmode=zmode)  # for non-epileptogenic regions
#     else:
#         x0e_6d = calc_x0_val__to_model_x0(x0e, yc, Iext1, zmode=zmode)
#         x0ne_6d = calc_x0_val__to_model_x0(x0ne, yc, Iext1, zmode=zmode)
#         # z nullcline:
#         zZe = calc_fz(x1, z=0.0, x0_values=x0e_6d, zmode=zmode, model="2d")  # for epileptogenic regions
#         zZne = calc_fz(x1, z=0.0, x0_values=x0ne_6d, zmode=zmode, model="2d")  # for non-epileptogenic regions
#
#     fig = pyplot.figure(figure_name, figsize=figsize)
#     x1null, = pyplot.plot(x1, zX1, 'b-', label='x1 nullcline', linewidth=1)
#     ax = pyplot.gca()
#     ax.axes.hold(True)
#     zE1null, = pyplot.plot(x1, zZe, 'g-', label='z nullcline at critical point (e_values=1)', linewidth=1)
#     zE2null, = pyplot.plot(x1, zZne, 'g--', label='z nullcline for e_values=0', linewidth=1)
#     sq, = pyplot.plot(x1sq, zX1sq, 'm--', label='Parabolic local approximation', linewidth=2)
#     lin, = pyplot.plot(x1lin, zX1lin, 'c--', label='Linear local approximation', linewidth=2)
#     pyplot.legend(handles=[x1null, zE1null, zE2null, lin, sq])
#
#     ii = range(len(region_labels))
#     if special_idx is None:
#         ii = np.delete(ii, special_idx)
#
#     points = []
#     for i in ii:
#         point, = pyplot.plot(model_configuration.x1EQ[i], model_configuration.zEQ[i], '*', mfc='k', mec='k',
#                              ms=10, alpha=0.3,
#                              label=str(i) + '.' + region_labels[i])
#         points.append(point)
#     if special_idx is None:
#         for i in special_idx:
#             point, = pyplot.plot(model_configuration.x1EQ[i], model_configuration.zEQ[i], '*', mfc='r', mec='r',
#                                  ms=10, alpha=0.8, label=str(i) + '.' + region_labels[i])
#             points.append(point)
#     # ax.plot(x1lin0, zlin0, '*', mfc='r', mec='r', ms=10)
#     # ax.axes.text(x1lin0 - 0.1, zlin0 + 0.2, 'e_values=0.0', fontsize=10, color='r')
#     # ax.plot(x1sq0, zsq0, '*', mfc='m', mec='m', ms=10)
#     # ax.axes.text(x1sq0, zsq0 - 0.2, 'e_values=1.0', fontsize=10, color='m')
#     if model == "2d":
#         ax.set_title(
#             "Equilibria, nullclines and Taylor series approximations \n at the x1-z phase plane of the" +
#             add_name + " for x1<0")
#     else:
#         ax.set_title("Equilibria, nullclines at the x1-z phase plane of the" + add_name + " for x1<0")
#     ax.axes.autoscale(tight=True)
#     ax.axes.set_xlabel('x1')
#     ax.axes.set_ylabel('z')
#     ax.axes.set_ylim(2.0, 5.0)
#     if MOUSEHOOVER:
#         # datacursor( lines[0], formatter='{label}'.format, bbox=dict(fc='white'),
#         #           arrowprops=dict(arrowstyle='simple', fc='white', alpha=0.5) )    #hover=True
#         HighlightingDataCursor(points[0], formatter='{label}'.format, bbox=dict(fc='white'),
#                                arrowprops=dict(arrowstyle='simple', fc='white', alpha=0.5))
#
#     if len(fig.get_label()) == 0:
#         fig.set_label(figure_name)
#     else:
#         figure_name = fig.get_label().replace(": ", "_").replace(" ", "_").replace("\t", "_")
#
#     _save_figure(save_flag, figure_dir=figure_dir, figure_format=figure_format, figure_name=figure_name)
#     _check_show(show_flag)


# def plot_hypothesis_model_configuration_and_lsa(hypothesis, model_configuration, plot_equilibria=False, n_eig=None,
#                                                 weighted_eigenvector_sum=None,
#                                                 figure_name='', show_flag=False, save_flag=True,
#                                                 figure_dir=FOLDER_FIGURES, figure_format=FIG_FORMAT,
#                                                 figsize=VERY_LARGE_SIZE):
#     fig = pyplot.figure(hypothesis.name + ": Overview", frameon=False, figsize=figsize)
#
#     mp.gridspec.GridSpec(1, 5+2*plot_equilibria, width_ratios=[1, 1] + plot_equilibria*[1, 1]+[1, 2, 1])
#     subplot_ind = 150 + 20*plot_equilibria
#
#     ax0 = plot_vector(model_configuration.x0_values, hypothesis.get_region_labels(), subplot_ind+1,
#                        'Excitabilities x0_values', show_y_labels=False, indices_red=hypothesis.x0_indices)
#
#     plot_vector(model_configuration.E_values, hypothesis.get_region_labels(), subplot_ind+2, 'Epileptogenicities e_values',
#                  show_y_labels=False, indices_red=hypothesis.e_indices, sharey=ax0)
#
#     if plot_equilibria:
#         plot_vector(model_configuration.x1EQ, hypothesis.get_region_labels(), subplot_ind+3, 'x1 Equilibria',
#                      show_y_labels=False, indices_red=hypothesis.get_all_disease_indices(), sharey=ax0)
#
#         plot_vector(model_configuration.zEQ, hypothesis.get_region_labels(), subplot_ind+4, 'z Equilibria',
#                      show_y_labels=False, indices_red=hypothesis.get_all_disease_indices(), sharey=ax0)
#
#     plot_vector(model_configuration.Ceq, hypothesis.get_region_labels(), subplot_ind+3+2*plot_equilibria,
#                  'Total afferent coupling \n at equilibrium', show_y_labels=False,
#                  indices_red=hypothesis.get_all_disease_indices(), sharey=ax0)
#
#     seizure_and_propagation_indices = np.unique(np.r_[hypothesis.get_all_disease_indices(),
#                                                             hypothesis.propagation_indices])
#
#     if len(seizure_and_propagation_indices) > 0:
#         plot_regions2regions(hypothesis.get_weights(), hypothesis.get_region_labels(), subplot_ind+4+2*plot_equilibria,
#                               'Afferent connectivity \n from seizuring regions',
#                               show_y_labels=False, show_x_labels=True,
#                               indices_red_x=seizure_and_propagation_indices, sharey=ax0)
#
#     if hypothesis.propagation_strengths is not None:
#         title = "LSA Propagation Strength:\nabsolut "
#         if weighted_eigenvector_sum:
#             title += ":\nabsolut eigenvalue-weighted sum of first "
#             if n_eig is not None:
#                 title += str(n_eig) + " "
#             title += "eigenvectors"
#         plot_vector(hypothesis.propagation_strengths, hypothesis.get_region_labels(), subplot_ind+5+2*plot_equilibria,
#                      title, show_y_labels=False, indices_red=seizure_and_propagation_indices, sharey=ax0)
#
#     _set_axis_labels(fig, 121, hypothesis.get_number_of_regions(), hypothesis.get_region_labels(),
#                      hypothesis.get_regions_disease_indices(), 'r')
#     _set_axis_labels(fig, 122, hypothesis.get_number_of_regions(), hypothesis.get_region_labels(),
#                      seizure_and_propagation_indices, 'r', 'right')
#
#     if figure_name == '':
#         figure_name = fig.get_label().replace(": ", "_").replace(" ", "_").replace("\t", "_")
#
#     _save_figure(save_flag, figure_dir=figure_dir, figure_format=figure_format, figure_name=figure_name)
#     _check_show(show_flag)
#
#
# def plot_lsa_pse(hypothesis, model_configuration, pse_results, plot_equilibria=False, n_eig=None,
#                  weighted_eigenvector_sum=None, colormap="YlOrRd", figure_name='', show_flag=False, save_flag=True,
#                  figure_dir=FOLDER_FIGURES, figure_format=FIG_FORMAT, figsize=VERY_LARGE_SIZE):
#     fig = pyplot.figure(hypothesis.name + ": LSA PSE overview", frameon=False, figsize=figsize)
#
#     mp.gridspec.GridSpec(1, 5 + 2 * plot_equilibria, width_ratios=[1, 1] + plot_equilibria * [1, 1] + [1, 2, 1])
#     subplot_ind = 150 + 20 * plot_equilibria
#
#     if pse_results.get("x0_values") is not None:
#         ax0 = plot_vector_violin(model_configuration.x0_values, pse_results.get("x0_values"),
#                                   hypothesis.get_region_labels(), subplot_ind+1, 'Excitabilities x0_values', colormap=colormap,
#                                   show_y_labels=False, indices_red=hypothesis.x0_indices)
#     else:
#         ax0 = plot_vector(model_configuration.x0_values, hypothesis.get_region_labels(), subplot_ind+1,
#                            'Excitabilities x0_values', show_y_labels=False, indices_red=hypothesis.x0_indices)
#
#     if pse_results.get("E_values") is not None:
#         plot_vector_violin(model_configuration.x0_values, pse_results.get("E_values"), hypothesis.get_region_labels(),
#                             subplot_ind + 2, 'Epileptogenicities e_values', colormap=colormap, show_y_labels=False,
#                             indices_red=hypothesis.e_indices, sharey=ax0)
#     else:
#         plot_vector(model_configuration.E_values, hypothesis.get_region_labels(), subplot_ind+2,
#                      'Epileptogenicities e_values', show_y_labels=False, indices_red=hypothesis.e_indices, sharey=ax0)
#
#     if plot_equilibria:
#         if pse_results.get("x1EQ") is not None:
#             plot_vector_violin(model_configuration.x1EQ, pse_results.get("x1EQ"), hypothesis.get_region_labels(),
#                                 subplot_ind+3, 'x1 Equilibria', colormap=colormap, show_y_labels=False,
#                                 indices_red=hypothesis.get_all_disease_indices(), sharey=ax0)
#         else:
#             plot_vector(model_configuration.x1EQ, hypothesis.get_region_labels(), subplot_ind+3, 'x1 Equilibria',
#                          show_y_labels=False, indices_red=hypothesis.get_all_disease_indices(), sharey=ax0)
#
#         if pse_results.get("zEQ") is not None:
#             plot_vector_violin(model_configuration.zEQ, pse_results.get("zEQ"), hypothesis.get_region_labels(),
#                                 subplot_ind+4, 'z Equilibria', colormap=colormap+"_r", show_y_labels=False,
#                                 indices_red=hypothesis.get_all_disease_indices(), sharey=ax0)
#         else:
#             plot_vector(model_configuration.zEQ, hypothesis.get_region_labels(), subplot_ind+4, 'z Equilibria',
#                          show_y_labels=False, indices_red=hypothesis.get_all_disease_indices(), sharey=ax0)
#
#     if pse_results.get("Ceq") is not None:
#         plot_vector_violin(model_configuration.Ceq, pse_results.get("Ceq"), hypothesis.get_region_labels(),
#                             subplot_ind + 3 + 2 * plot_equilibria, 'Total afferent coupling \n at equilibrium',
#                             colormap=colormap, show_y_labels=False, indices_red=hypothesis.get_all_disease_indices(),
#                             sharey=ax0)
#     else:
#         plot_vector(model_configuration.Ceq, hypothesis.get_region_labels(), subplot_ind+3+2*plot_equilibria,
#                      'Total afferent coupling \n at equilibrium', show_y_labels=False,
#                      indices_red=hypothesis.get_regions_disease_indices(), sharey=ax0)
#
#     seizure_and_propagation_indices = np.unique(np.r_[hypothesis.get_all_disease_indices(),
#                                                             hypothesis.propagation_indices])
#
#     if len(seizure_and_propagation_indices) > 0:
#         plot_regions2regions(hypothesis.get_weights(), hypothesis.get_region_labels(), subplot_ind+4+2*plot_equilibria,
#                               'Afferent connectivity \n from seizuring regions', show_y_labels=False,
#                               show_x_labels=True, indices_red_x=seizure_and_propagation_indices, sharey=ax0)
#
#     if hypothesis.propagation_strengths is not None:
#         title = "LSA Propagation Strength:\nabsolut "
#         if weighted_eigenvector_sum:
#             title += "eigenvalue-weighted \n"
#         title += "sum of first "
#         if n_eig is not None:
#             title += str(n_eig) + " "
#         title += "eigenvectors"
#         if pse_results.get("propagation_strengths") is not None:
#             plot_vector_violin(hypothesis.propagation_strengths, pse_results.get("propagation_strengths"),
#                                 hypothesis.get_region_labels(), subplot_ind+5+2*plot_equilibria, title,
#                                 show_y_labels=False, indices_red=seizure_and_propagation_indices, sharey=ax0)
#         else:
#             plot_vector(hypothesis.propagation_strengths, hypothesis.get_region_labels(),
#                          subplot_ind+5+2*plot_equilibria, title, show_y_labels=False,
#                          indices_red=seizure_and_propagation_indices, sharey=ax0)
#
#     _set_axis_labels(fig, 121, hypothesis.get_number_of_regions(), hypothesis.get_region_labels(),
#                      hypothesis.get_regions_disease_indices(), 'r')
#     _set_axis_labels(fig, 122, hypothesis.get_number_of_regions(), hypothesis.get_region_labels(),
#                      seizure_and_propagation_indices, 'r', 'right')
#
#     if figure_name == '':
#         figure_name = fig.get_label().replace(": ", "_").replace(" ", "_").replace("\t", "_")
#
#     _save_figure(save_flag, figure_dir=figure_dir, figure_format=figure_format, figure_name=figure_name)
#     _check_show(show_flag)