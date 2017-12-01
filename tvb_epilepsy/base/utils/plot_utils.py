"""
Various plotting tools will be placed here.
"""
# TODO: make a plot function for sensitivity analysis results
import os

import numpy as np
import matplotlib as mp
from matplotlib import pyplot, gridspec
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats.mstats import zscore

from tvb_epilepsy.base.computations.analyzers_utils import time_spectral_analysis
from tvb_epilepsy.base.constants.configurations import FOLDER_FIGURES, VERY_LARGE_SIZE, LARGE_SIZE, FIG_FORMAT, \
    SAVE_FLAG, \
    SHOW_FLAG
from tvb_epilepsy.base.utils.data_structures_utils import sort_dict, ensure_list
from tvb_epilepsy.base.utils.log_error_utils import warning
from tvb_epilepsy.tvb_api.epileptor_models import *

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


def figure_filename(fig=None, figure_name=None):
    if fig is None:
        fig = pyplot.gcf()
    if figure_name is None:
        figure_name = fig.get_label()
    else:
        figure_name = figure_name.replace(": ", "_").replace(" ", "_").replace("\t", "_")
    return figure_name


def save_figure(save_flag=SAVE_FLAG, fig=None, figure_name=None, figure_dir=FOLDER_FIGURES,
                figure_format=FIG_FORMAT):
    if save_flag:
        figure_name = figure_filename(fig, figure_name)
        figure_name = figure_name[:np.min([100, len(figure_name)])] + '.' + figure_format
        if not (os.path.isdir(figure_dir)):
            os.mkdir(figure_dir)
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
                    description="", title="", figure_name=None, show_flag=False, save_flag=True,
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
    save_figure(save_flag, pyplot.gcf(), figure_name, figure_dir, figure_format)
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


def plot_timeseries(time, data_dict, time_units="ms", special_idx=None, title='Time Series', figure_name=None,
                    labels=None, show_flag=SHOW_FLAG, save_flag=False, figure_dir=FOLDER_FIGURES,
                    figure_format=FIG_FORMAT, figsize=LARGE_SIZE):
    pyplot.figure(title, figsize=figsize)
    if not(isinstance(figure_name, basestring)):
        figure_name = title.replace(".", "").replace(' ', "")
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
    save_figure(save_flag, pyplot.gcf(), figure_name, figure_dir, figure_format)
    check_show(show_flag)


def plot_raster(time, data_dict, time_units="ms", special_idx=None, title='Time Series', subtitles=[], offset=3.0,
                figure_name=None, labels=None, show_flag=SHOW_FLAG, save_flag=False, figure_dir=FOLDER_FIGURES,
                figure_format=FIG_FORMAT, figsize=VERY_LARGE_SIZE):
    pyplot.figure(title, figsize=figsize)
    no_rows = len(data_dict)
    lines = []

    def plot_line(color):
        try:
            return pyplot.plot(time, -data[:, iTS]+offset*iTS, color, label=labels[iTS])
        except:
            return pyplot.plot(time, -data[:, iTS] + offset * iTS, color, label="_".join(["SEEG", str(iTS)]))

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
                # line, = pyplot.plot(time, -data[:,iTS]+offset*iTS, 'k', label=labels[iTS])
                line, = plot_line("k")
                lines[i].append(line)
        else:
            mask = np.array(range(nTS))
            mask = np.delete(mask,special_idx)
            for iTS in special_idx:
                # line, = pyplot.plot(time, -data[:, iTS]+offset*iTS, 'r', label=labels[iTS])
                line, = plot_line('r')
                lines[i].append(line)
            for iTS in mask:
                # line, = pyplot.plot(time, -data[:, iTS]+offset*iTS, 'k', label=labels[iTS])
                line, = plot_line('k')
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
    save_figure(save_flag, pyplot.gcf(), figure_name, figure_dir, figure_format)
    check_show(show_flag)


def plot_trajectories(data_dict, special_idx=None, title='State space trajectories', figure_name=None,
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
    save_figure(save_flag, pyplot.gcf(), figure_name, figure_dir, figure_format)
    check_show(show_flag)


def plot_spectral_analysis_raster(time, data, time_units="ms", freq=None, special_idx=None, title='Spectral Analysis',
                                  figure_name=None, labels=None, show_flag=SHOW_FLAG, save_flag=SAVE_FLAG,
                                  figure_dir=FOLDER_FIGURES, figure_format=FIG_FORMAT, figsize=VERY_LARGE_SIZE,
                                  **kwargs):
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
                                                  nperseg=kwargs.get("nperseg", int(np.round(fs/4))),
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
    save_figure(save_flag, pyplot.gcf(), figure_name, figure_dir, figure_format)
    check_show(show_flag)
    return fig, ax, img, line, time, freq, stf, psd


def plot_sim_results(model, seizure_indices, hyp_name, res, sensorsSEEG=None, hpf_flag=False,
                     trajectories_plot=False, spectral_raster_plot=False, region_labels=None,
                     save_flag=SAVE_FLAG, show_flag=SHOW_FLAG, figure_dir=FOLDER_FIGURES, figure_format=FIG_FORMAT,
                     **kwargs):
    if isinstance(model, EpileptorDP2D):
        plot_timeseries(res['time'], {'x1': res['x1'], 'z(t)': res['z']}, time_units=res.get('time_units', "ms"),
                        special_idx=seizure_indices, title=hyp_name + ": Simulated TAVG",
                        save_flag=save_flag, show_flag=show_flag, figure_dir=figure_dir, figure_format=figure_format,
                        labels=region_labels, figsize=VERY_LARGE_SIZE)
        plot_raster(res['time'], {'x1': res['x1']},
                    time_units=res.get('time_units', "ms"), special_idx=seizure_indices,
                    title=hyp_name + ": Simulated x1 rasterplot", offset=5.0, labels=region_labels,
                    save_flag=save_flag, show_flag=show_flag, figure_dir=figure_dir, figure_format=figure_format,
                    figsize=VERY_LARGE_SIZE)
    else:
        plot_timeseries(res['time'], {'LFP(t)': res['lfp'], 'z(t)': res['z']}, time_units=res.get('time_units', "ms"),
                        special_idx=seizure_indices, title=hyp_name + ": Simulated LFP-z",
                        save_flag=save_flag, show_flag=show_flag, figure_dir=figure_dir, figure_format=figure_format,
                        labels=region_labels, figsize=VERY_LARGE_SIZE)
        plot_timeseries(res['time'], {'x1(t)': res['x1'], 'y1(t)': res['y1']},time_units=res.get('time_units', "ms"),
                        special_idx=seizure_indices, title=hyp_name + ": Simulated pop1",
                        save_flag=save_flag, show_flag=show_flag, figure_dir=figure_dir, figure_format=figure_format,
                        labels=region_labels, figsize=VERY_LARGE_SIZE)
        plot_timeseries(res['time'], {'x2(t)': res['x2'], 'y2(t)': res['y2'], 'g(t)': res['g']},
                        time_units=res.get('time_units', "ms"), special_idx=seizure_indices,
                        title=hyp_name + ": Simulated pop2-g",
                        save_flag=save_flag, show_flag=show_flag, figure_dir=figure_dir, figure_format=figure_format,
                        labels=region_labels, figsize=VERY_LARGE_SIZE)
        start_plot = int(np.round(0.01 * res['lfp'].shape[0]))
        plot_raster(res['time'][start_plot:], {'lfp': res['lfp'][start_plot:, :]},
                    time_units=res.get('time_units', "ms"), special_idx=seizure_indices,
                    title=hyp_name + ": Simulated LFP rasterplot", offset=10.0, labels=region_labels,
                    save_flag=save_flag, show_flag=show_flag, figure_dir=figure_dir, figure_format=figure_format,
                    figsize=VERY_LARGE_SIZE)
    if isinstance(model, EpileptorDPrealistic):
        plot_timeseries(res['time'], {'1/(1+exp(-10(z-3.03))': 1 / (1 + np.exp(-10 * (res['z'] - 3.03))),
                                      'slope': res['slope_t'], 'Iext2': res['Iext2_t']},
                        time_units=res.get('time_units', "ms"), special_idx=seizure_indices,
                        title=hyp_name + ": Simulated controlled parameters", labels=region_labels,
                        save_flag=save_flag, show_flag=show_flag, figure_dir=figure_dir, figure_format=figure_format,
                        figsize=VERY_LARGE_SIZE)
        plot_timeseries(res['time'], {'x0_values': res['x0_t'], 'Iext1':  res['Iext1_t'], 'K': res['K_t']},
                        time_units=res.get('time_units', "ms"), special_idx=seizure_indices,
                        title=hyp_name + ": Simulated parameters", labels=region_labels,
                        save_flag=save_flag, show_flag=show_flag, figure_dir=figure_dir, figure_format=figure_format,
                        figsize=VERY_LARGE_SIZE)
    if trajectories_plot:
        plot_trajectories({'x1': res['x1'], 'z(t)': res['z']}, special_idx=seizure_indices,
                          title=hyp_name + ': State space trajectories', labels=region_labels,
                          show_flag=show_flag, save_flag=save_flag, figure_dir=FOLDER_FIGURES, figure_format=FIG_FORMAT,
                          figsize=LARGE_SIZE)
    if spectral_raster_plot is "lfp":
        plot_spectral_analysis_raster(res["time"], res['lfp'], time_units=res.get('time_units', "ms"),
                                      freq=None, special_idx=seizure_indices,
                                      title=hyp_name + ": Spectral Analysis",
                                      labels=region_labels,
                                      show_flag=show_flag, save_flag=save_flag, figure_dir=figure_dir,
                                      figure_format=figure_format, figsize=LARGE_SIZE, **kwargs)
    if sensorsSEEG is not None:
        sensorsSEEG = ensure_list(sensorsSEEG)
        for i in range(len(sensorsSEEG)):
            if hpf_flag:
                title = hyp_name + ": Simulated high pass filtered SEEG" + str(i) + " raster plot"
                start_plot = int(np.round(0.01 * res['SEEG' + str(i)].shape[0]))
            else:
                title = hyp_name + ": Simulated SEEG" + str(i) + " raster plot"
                start_plot = 0
            plot_raster(res['time'][start_plot:], {'SEEG': res['SEEG'+str(i)][start_plot:, :]},
                        time_units=res.get('time_units', "ms"), title=title,
                        offset=0.0, save_flag=save_flag, show_flag=show_flag, figure_dir=figure_dir,
                        figure_format=figure_format, labels=sensorsSEEG[i].labels, figsize=VERY_LARGE_SIZE)
