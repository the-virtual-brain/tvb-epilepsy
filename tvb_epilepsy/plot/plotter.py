import numpy
from scipy.stats import zscore
from matplotlib import pyplot, gridspec
from matplotlib.colors import Normalize
from mpldatacursor import HighlightingDataCursor

from tvb_epilepsy.base.computations.analyzers_utils import time_spectral_analysis
from tvb_epilepsy.base.utils.data_structures_utils import ensure_list
from tvb_epilepsy.plot.base_plotter import BasePlotter
from tvb_epilepsy.tvb_api.epileptor_models import EpileptorDP2D, EpileptorDPrealistic
from tvb_epilepsy.base.constants.configurations import SHOW_FLAG, FOLDER_FIGURES, FIG_FORMAT, LARGE_SIZE, MOUSEHOOVER, \
    VERY_LARGE_SIZE, SAVE_FLAG


class Plotter(BasePlotter):

    def plot_timeseries(self, time, data_dict, time_units="ms", special_idx=None, title='Time Series', figure_name=None,
                        labels=None, show_flag=SHOW_FLAG, save_flag=False, figure_dir=FOLDER_FIGURES,
                        figure_format=FIG_FORMAT, figsize=LARGE_SIZE):
        pyplot.figure(title, figsize=figsize)
        if not (isinstance(figure_name, basestring)):
            figure_name = title.replace(".", "").replace(' ', "")
        no_rows = len(data_dict)
        lines = []

        def plot_line(color, alpha):
            try:
                return pyplot.plot(time, data[:, iTS], color, alpha=alpha, label=labels[iTS])
            except:
                return pyplot.plot(time, data[:, iTS], color, alpha=alpha, label=str(iTS))

        for i, subtitle in enumerate(data_dict):
            ax = pyplot.subplot(no_rows, 1, i + 1)
            pyplot.hold(True)
            if i == 0:
                pyplot.title(title)
            data = data_dict[subtitle]
            nTS = data.shape[1]
            if labels is None:
                labels = numpy.array(range(nTS)).astype(str)
            lines.append([])
            if special_idx is None:
                for iTS in range(nTS):
                    # line, = pyplot.plot(time, data[:, iTS], 'k', alpha=0.3, label=labels[iTS])
                    line, = plot_line('k', 0.3)
                    lines[i].append(line)
            else:
                mask = numpy.array(range(nTS))
                mask = numpy.delete(mask, special_idx)
                for iTS in special_idx:
                    # line, = pyplot.plot(time, data[:, iTS], 'r', alpha=0.7, label=labels[iTS])
                    line, = plot_line('r', 0.7)
                    lines[i].append(line)
                for iTS in mask:
                    # line, = pyplot.plot(time, data[:, iTS], 'k', alpha=0.3, label=labels[iTS])
                    line, = plot_line('k', 0.3)
                    lines[i].append(line)
            pyplot.ylabel(subtitle)
            ax.set_autoscalex_on(False)
            ax.set_xlim([time[0], time[-1]])
            if MOUSEHOOVER:
                # datacursor( lines[i], formatter='{label}'.format, bbox=dict(fc='white'),
                #           arrowprops=dict(arrowstyle='simple', fc='white', alpha=0.5) )    #hover=True
                HighlightingDataCursor(lines[i], formatter='{label}'.format, bbox=dict(fc='white'),
                                       arrowprops=dict(arrowstyle='simple', fc='white', alpha=0.5))
        pyplot.xlabel("Time (" + time_units + ")")
        self._save_figure(save_flag, pyplot.gcf(), figure_name, figure_dir, figure_format)
        self._check_show(show_flag)

    def plot_raster(self, time, data_dict, time_units="ms", special_idx=None, title='Time Series', subtitles=[],
                    offset=1.0,
                    figure_name=None, labels=None, show_flag=SHOW_FLAG, save_flag=False, figure_dir=FOLDER_FIGURES,
                    figure_format=FIG_FORMAT, figsize=VERY_LARGE_SIZE):
        pyplot.figure(title, figsize=figsize)
        no_rows = len(data_dict)
        lines = []

        def plot_line(color):
            try:
                return pyplot.plot(time, -data[:, iTS] + offset * iTS, color, label=labels[iTS])
            except:
                return pyplot.plot(time, -data[:, iTS] + offset * iTS, color, label=str(iTS))

        for i, var in enumerate(data_dict):
            ax = pyplot.subplot(1, no_rows, i + 1)
            pyplot.hold(True)
            if len(subtitles) > i:
                pyplot.title(subtitles[i])
            data = data_dict[var]
            data = zscore(data, axis=None)
            nTS = data.shape[1]
            ticks = (offset * numpy.array([range(nTS)])).tolist()
            if labels is None:
                labels = numpy.array(range(nTS)).astype(str)
            lines.append([])
            if special_idx is None:
                for iTS in range(nTS):
                    # line, = pyplot.plot(time, -data[:,iTS]+offset*iTS, 'k', label=labels[iTS])
                    line, = plot_line("k")
                    lines[i].append(line)
            else:
                mask = numpy.array(range(nTS))
                mask = numpy.delete(mask, special_idx)
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
                # datacursor( lines[i], formatter='{label}'.format, bbox=dict(fc='white'),
                #           arrowprops=dict(arrowstyle='simple', fc='white', alpha=0.5) )    #hover=True
                HighlightingDataCursor(lines[i], formatter='{label}'.format, bbox=dict(fc='white'),
                                       arrowprops=dict(arrowstyle='simple', fc='white', alpha=0.5))
        pyplot.xlabel("Time (" + time_units + ")")
        self._save_figure(save_flag, pyplot.gcf(), figure_name, figure_dir, figure_format)
        self._check_show(show_flag)

    def plot_trajectories(self, data_dict, special_idx=None, title='State space trajectories', figure_name=None,
                          labels=None, show_flag=SHOW_FLAG, save_flag=SAVE_FLAG, figure_dir=FOLDER_FIGURES,
                          figure_format=FIG_FORMAT, figsize=LARGE_SIZE):
        pyplot.figure(title, figsize=figsize)
        ax = pyplot.subplot(111)
        pyplot.hold(True)
        no_dims = len(data_dict)
        if no_dims > 2:
            ax = pyplot.subplot(111, projection='3d')
        else:
            ax = pyplot.subplot(111)
        lines = []
        ax_labels = []
        data = []
        for i, var in enumerate(data_dict):
            if i == 0:
                pyplot.title(title)
            ax_labels.append(var)
            data.append(data_dict[var])
        nTS = data[0].shape[1]
        if labels is None:
            labels = numpy.array(range(nTS)).astype(str)
        lines.append([])
        if special_idx is None:
            for iTS in range(nTS):
                if no_dims > 2:
                    line, = pyplot.plot(data[0][:, iTS], data[1][:, iTS], data[2][:, iTS], 'k', alpha=0.3,
                                        label=labels[iTS])
                else:
                    line, = pyplot.plot(data[0][:, iTS], data[1][:, iTS], 'k', alpha=0.3, label=labels[iTS])
                lines.append(line)
        else:
            mask = numpy.array(range(nTS))
            mask = numpy.delete(mask, special_idx)
            for iTS in special_idx:
                if no_dims > 2:
                    line, = pyplot.plot(data[0][:, iTS], data[1][:, iTS], data[2][:, iTS], 'r', alpha=0.7,
                                        label=labels[iTS])
                else:
                    line, = pyplot.plot(data[0][:, iTS], data[1][:, iTS], 'r', alpha=0.7, label=labels[iTS])
                lines.append(line)
            for iTS in mask:
                if no_dims > 2:
                    line, = pyplot.plot(data[0][:, iTS], data[1][:, iTS], data[2][:, iTS], 'k', alpha=0.3,
                                        label=labels[iTS])
                else:
                    line, = pyplot.plot(data[0][:, iTS], data[1][:, iTS], 'k', alpha=0.3, label=labels[iTS])
                lines.append(line)
        pyplot.xlabel(ax_labels[0])
        pyplot.ylabel(ax_labels[1])
        if no_dims > 2:
            pyplot.ylabel(ax_labels[2])
        if MOUSEHOOVER:
            # datacursor( lines[0], formatter='{label}'.format, bbox=dict(fc='white'),
            #           arrowprops=dict(arrowstyle='simple', fc='white', alpha=0.5) )    #hover=True
            HighlightingDataCursor(lines[0], formatter='{label}'.format, bbox=dict(fc='white'),
                                   arrowprops=dict(arrowstyle='simple', fc='white', alpha=0.5))
        self._save_figure(save_flag, pyplot.gcf(), figure_name, figure_dir, figure_format)
        self._check_show(show_flag)

    def plot_spectral_analysis_raster(self, time, data, time_units="ms", freq=None, special_idx=None,
                                      title='Spectral Analysis',
                                      figure_name=None, labels=None, show_flag=SHOW_FLAG, save_flag=SAVE_FLAG,
                                      figure_dir=FOLDER_FIGURES, figure_format=FIG_FORMAT, figsize=VERY_LARGE_SIZE,
                                      **kwargs):
        if time_units in ("ms", "msec"):
            fs = 1000.0
        else:
            fs = 1.0
        fs = fs / numpy.mean(numpy.diff(time))
        if special_idx is not None:
            data = data[:, special_idx]
            if labels is not None:
                labels = numpy.array(labels)[special_idx]
        nS = data.shape[1]
        if labels is None:
            labels = numpy.array(range(nS)).astype(str)
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
                                                      nperseg=kwargs.get("nperseg", int(numpy.round(fs / 4))),
                                                      detrend=kwargs.get("detrend", 'constant'),
                                                      noverlap=kwargs.get("noverlap"),
                                                      f_low=kwargs.get("f_low", 10.0),
                                                      log_scale=kwargs.get("log_scale", False))
        min_val = numpy.min(stf.flatten())
        max_val = numpy.max(stf.flatten())
        if nS > 2:
            figsize = VERY_LARGE_SIZE
        fig = pyplot.figure(title, figsize=figsize)
        fig.suptitle(title)
        gs = gridspec.GridSpec(nS, 23)
        ax = numpy.empty((nS, 2), dtype="O")
        img = numpy.empty((nS,), dtype="O")
        line = numpy.empty((nS,), dtype="O")
        for iS in range(nS - 1, -1, -1):
            if iS < nS - 1:
                ax[iS, 0] = pyplot.subplot(gs[iS, :20], sharex=ax[iS, 0])
                ax[iS, 1] = pyplot.subplot(gs[iS, 20:22], sharex=ax[iS, 1], sharey=ax[iS, 0])
            else:
                ax[iS, 0] = pyplot.subplot(gs[iS, :20])
                ax[iS, 1] = pyplot.subplot(gs[iS, 20:22], sharey=ax[iS, 0])
            img[iS] = ax[iS, 0].imshow(numpy.squeeze(stf[:, :, iS]).T, cmap=pyplot.set_cmap('jet'),
                                       interpolation='none',
                                       norm=Normalize(vmin=min_val, vmax=max_val), aspect='auto', origin='lower',
                                       extent=(time.min(), time.max(), freq.min(), freq.max()))
            # img[iS].clim(min_val, max_val)
            ax[iS, 0].set_title(labels[iS])
            ax[iS, 0].set_ylabel("Frequency (Hz)")
            line[iS] = ax[iS, 1].plot(psd[:, iS], freq, 'k', label=labels[iS])
            pyplot.setp(ax[iS, 1].get_yticklabels(), visible=False)
            # ax[iS, 1].yaxis.tick_right()
            # ax[iS, 1].yaxis.set_ticks_position('both')
            if iS == (nS - 1):
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
        self._save_figure(save_flag, pyplot.gcf(), figure_name, figure_dir, figure_format)
        self._check_show(show_flag)
        return fig, ax, img, line, time, freq, stf, psd

    def plot_sim_results(self, model, seizure_indices, res, sensorsSEEG=None, hpf_flag=False,
                         trajectories_plot=False, spectral_raster_plot=False, region_labels=None,
                         save_flag=SAVE_FLAG, show_flag=SHOW_FLAG, figure_dir=FOLDER_FIGURES, figure_format=FIG_FORMAT,
                         **kwargs):
        if isinstance(model, EpileptorDP2D):
            self.plot_timeseries(res['time'], {'x1': res['x1'], 'z(t)': res['z']},
                                 time_units=res.get('time_units', "ms"),
                                 special_idx=seizure_indices, title=model._ui_name + ": Simulated TAVG",
                                 save_flag=save_flag, show_flag=show_flag, figure_dir=figure_dir,
                                 figure_format=figure_format,
                                 labels=region_labels, figsize=VERY_LARGE_SIZE)
            self.plot_raster(res['time'], {'x1': res['x1']},
                             time_units=res.get('time_units', "ms"), special_idx=seizure_indices,
                             title=model._ui_name + ": Simulated x1 rasterplot", offset=5.0, labels=region_labels,
                             save_flag=save_flag, show_flag=show_flag, figure_dir=figure_dir,
                             figure_format=figure_format,
                             figsize=VERY_LARGE_SIZE)
        else:
            self.plot_timeseries(res['time'], {'LFP(t)': res['lfp'], 'z(t)': res['z']},
                                 time_units=res.get('time_units', "ms"),
                                 special_idx=seizure_indices, title=model._ui_name + ": Simulated LFP-z",
                                 save_flag=save_flag, show_flag=show_flag, figure_dir=figure_dir,
                                 figure_format=figure_format,
                                 labels=region_labels, figsize=VERY_LARGE_SIZE)
            self.plot_timeseries(res['time'], {'x1(t)': res['x1'], 'y1(t)': res['y1']},
                                 time_units=res.get('time_units', "ms"),
                                 special_idx=seizure_indices, title=model._ui_name + ": Simulated pop1",
                                 save_flag=save_flag, show_flag=show_flag, figure_dir=figure_dir,
                                 figure_format=figure_format,
                                 labels=region_labels, figsize=VERY_LARGE_SIZE)
            self.plot_timeseries(res['time'], {'x2(t)': res['x2'], 'y2(t)': res['y2'], 'g(t)': res['g']},
                                 time_units=res.get('time_units', "ms"), special_idx=seizure_indices,
                                 title=model._ui_name + ": Simulated pop2-g",
                                 save_flag=save_flag, show_flag=show_flag, figure_dir=figure_dir,
                                 figure_format=figure_format,
                                 labels=region_labels, figsize=VERY_LARGE_SIZE)
            start_plot = int(numpy.round(0.01 * res['lfp'].shape[0]))
            self.plot_raster(res['time'][start_plot:], {'lfp': res['lfp'][start_plot:, :]},
                             time_units=res.get('time_units', "ms"), special_idx=seizure_indices,
                             title=model._ui_name + ": Simulated LFP rasterplot", offset=10.0, labels=region_labels,
                             save_flag=save_flag, show_flag=show_flag, figure_dir=figure_dir,
                             figure_format=figure_format,
                             figsize=VERY_LARGE_SIZE)
        if isinstance(model, EpileptorDPrealistic):
            self.plot_timeseries(res['time'], {'1/(1+exp(-10(z-3.03))': 1 / (1 + numpy.exp(-10 * (res['z'] - 3.03))),
                                               'slope': res['slope_t'], 'Iext2': res['Iext2_t']},
                                 time_units=res.get('time_units', "ms"), special_idx=seizure_indices,
                                 title=model._ui_name + ": Simulated controlled parameters", labels=region_labels,
                                 save_flag=save_flag, show_flag=show_flag, figure_dir=figure_dir,
                                 figure_format=figure_format,
                                 figsize=VERY_LARGE_SIZE)
            self.plot_timeseries(res['time'], {'x0_values': res['x0_t'], 'Iext1': res['Iext1_t'], 'K': res['K_t']},
                                 time_units=res.get('time_units', "ms"), special_idx=seizure_indices,
                                 title=model._ui_name + ": Simulated parameters", labels=region_labels,
                                 save_flag=save_flag, show_flag=show_flag, figure_dir=figure_dir,
                                 figure_format=figure_format,
                                 figsize=VERY_LARGE_SIZE)
        if trajectories_plot:
            self.plot_trajectories({'x1': res['x1'], 'z(t)': res['z']}, special_idx=seizure_indices,
                                   title=model._ui_name + ': State space trajectories', labels=region_labels,
                                   show_flag=show_flag, save_flag=save_flag, figure_dir=FOLDER_FIGURES,
                                   figure_format=FIG_FORMAT,
                                   figsize=LARGE_SIZE)
        if spectral_raster_plot is "lfp":
            self.plot_spectral_analysis_raster(res["time"], res['lfp'], time_units=res.get('time_units', "ms"),
                                               freq=None, special_idx=seizure_indices,
                                               title=model._ui_name + ": Spectral Analysis",
                                               labels=region_labels,
                                               show_flag=show_flag, save_flag=save_flag, figure_dir=figure_dir,
                                               figure_format=figure_format, figsize=LARGE_SIZE, **kwargs)
        if sensorsSEEG is not None:
            sensorsSEEG = ensure_list(sensorsSEEG)
            for i in range(len(sensorsSEEG)):
                if hpf_flag:
                    title = model._ui_name + ": Simulated high pass filtered SEEG" + str(i) + " raster plot"
                    start_plot = int(numpy.round(0.01 * res['SEEG' + str(i)].shape[0]))
                else:
                    title = model._ui_name + ": Simulated SEEG" + str(i) + " raster plot"
                    start_plot = 0
                self.plot_raster(res['time'][start_plot:], {'SEEG': res['SEEG' + str(i)][start_plot:, :]},
                                 time_units=res.get('time_units', "ms"), title=title,
                                 offset=0.0, save_flag=save_flag, show_flag=show_flag, figure_dir=figure_dir,
                                 figure_format=figure_format, labels=sensorsSEEG[i].labels, figsize=VERY_LARGE_SIZE)
