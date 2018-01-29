import numpy
from scipy.stats import zscore
import matplotlib
matplotlib.use('Qt4Agg')
from matplotlib import pyplot, gridspec
from matplotlib.colors import Normalize
from mpldatacursor import HighlightingDataCursor
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tvb_epilepsy.base.utils.log_error_utils import warning
from tvb_epilepsy.plot.base_plotter import BasePlotter
from tvb_epilepsy.base.model.vep.sensors import Sensors
from tvb_epilepsy.base.utils.math_utils import compute_in_degree
from tvb_epilepsy.base.computations.analyzers_utils import time_spectral_analysis
from tvb_epilepsy.tvb_api.epileptor_models import EpileptorDP2D, EpileptorDPrealistic
from tvb_epilepsy.base.utils.data_structures_utils import ensure_list, isequal_string, sort_dict, linspace_broadcast, \
    list_of_dicts_to_dicts_of_ndarrays, extract_dict_stringkeys
from tvb_epilepsy.base.computations.equilibrium_computation import calc_eq_y1, def_x1lin
from tvb_epilepsy.base.computations.calculations_utils import calc_fz, calc_fx1, calc_fx1_2d_taylor, \
    calc_x0_val_to_model_x0, raise_value_error
from tvb_epilepsy.base.constants.model_constants import TAU0_DEF, TAU1_DEF, X1_EQ_CR_DEF, X1_DEF, X0_CR_DEF, X0_DEF
from tvb_epilepsy.base.constants.configurations import SHOW_FLAG, FOLDER_FIGURES, FIG_FORMAT, LARGE_SIZE, MOUSEHOOVER, \
    VERY_LARGE_SIZE, SAVE_FLAG, FIG_SIZE, VERY_LARGE_PROTRAIT


class Plotter(BasePlotter):

    def _plot_connectivity(self, connectivity, show_flag=SHOW_FLAG, save_flag=SAVE_FLAG, figure_dir=FOLDER_FIGURES,
                           figure_format=FIG_FORMAT, figure_name='Connectivity ', figsize=VERY_LARGE_SIZE):
        # plot connectivity
        pyplot.figure(figure_name + str(connectivity.number_of_regions), figsize)
        # plot_regions2regions(conn.weights, conn.region_labels, 121, "weights")
        self.plot_regions2regions(connectivity.normalized_weights, connectivity.region_labels, 121,
                                  "normalised weights")
        self.plot_regions2regions(connectivity.tract_lengths, connectivity.region_labels, 122, "tract lengths")
        if save_flag:
            self._save_figure(figure_dir=figure_dir, figure_format=figure_format,
                              figure_name=figure_name.replace(" ", "_").replace("\t", "_"))
        self._check_show(show_flag=show_flag)

    def _plot_connectivity_stats(self, connectivity, show_flag=SHOW_FLAG, save_flag=SAVE_FLAG,
                                 figure_dir=FOLDER_FIGURES, figure_format=FIG_FORMAT, figsize=VERY_LARGE_SIZE,
                                 figure_name='HeadStats '):
        pyplot.figure("Head stats " + str(connectivity.number_of_regions), figsize=figsize)
        areas_flag = len(connectivity.areas) == len(connectivity.region_labels)
        ax = self.plot_vector(compute_in_degree(connectivity.normalized_weights), connectivity.region_labels,
                              111 + 10 * areas_flag,
                              "w in-degree")
        ax.invert_yaxis()
        if len(connectivity.areas) == len(connectivity.region_labels):
            ax = self.plot_vector(connectivity.areas, connectivity.region_labels, 122, "region areas")
            ax.invert_yaxis()
        if save_flag:
            self._save_figure(figure_dir=figure_dir, figure_format=figure_format,
                              figure_name=figure_name.replace(" ", "").replace("\t", ""))
        self._check_show(show_flag=show_flag)

    def _plot_sensors(self, sensors, region_labels, count=1, show_flag=SHOW_FLAG, save_flag=SAVE_FLAG,
                      figure_dir=FOLDER_FIGURES, figure_format=FIG_FORMAT):
        # plot sensors:
        if sensors.gain_matrix is None:
            return count
        self._plot_gain_matrix(sensors, region_labels, title=str(count) + " - " + sensors.s_type + " - Projection",
                               show_flag=show_flag, save_flag=save_flag, figure_dir=figure_dir,
                               figure_format=figure_format)
        count += 1
        return count

    def _plot_gain_matrix(self, sensors, region_labels, figure=None, title="Projection", y_labels=1, x_labels=1,
                          x_ticks=numpy.array([]), y_ticks=numpy.array([]), show_flag=SHOW_FLAG, save_flag=SAVE_FLAG,
                          figure_dir=FOLDER_FIGURES, figure_format=FIG_FORMAT, figsize=VERY_LARGE_SIZE, figure_name=''):
        if not (isinstance(figure, pyplot.Figure)):
            figure = pyplot.figure(title, figsize=figsize)
        n_sensors = sensors.number_of_sensors
        n_regions = len(region_labels)
        if len(x_ticks) == 0:
            x_ticks = numpy.array(range(n_sensors), dtype=numpy.int32)
        if len(y_ticks) == 0:
            y_ticks = numpy.array(range(n_regions), dtype=numpy.int32)
        cmap = pyplot.set_cmap('autumn_r')
        img = pyplot.imshow(sensors.gain_matrix[x_ticks][:, y_ticks].T, cmap=cmap, interpolation='none')
        pyplot.grid(True, color='black')
        if y_labels > 0:
            region_labels = numpy.array(["%d. %s" % l for l in zip(range(n_regions), region_labels)])
            pyplot.yticks(y_ticks, region_labels[y_ticks])
        else:
            pyplot.yticks(y_ticks)
        if x_labels > 0:
            sensor_labels = numpy.array(["%d. %s" % l for l in zip(range(n_sensors), sensors.labels)])
            pyplot.xticks(x_ticks, sensor_labels[x_ticks], rotation=90)
        else:
            pyplot.xticks(x_ticks)
        ax = figure.get_axes()[0]
        ax.autoscale(tight=True)
        pyplot.title(title)
        divider = make_axes_locatable(ax)
        cax1 = divider.append_axes("right", size="5%", pad=0.05)
        pyplot.colorbar(img, cax=cax1)  # fraction=0.046, pad=0.04) #fraction=0.15, shrink=1.0
        if figure_name == "":
            figure_name = title
        self._save_figure(save_flag, figure_dir=figure_dir, figure_format=figure_format, figure_name=title)
        self._check_show(show_flag)
        return figure

    def plot_head(self, head, show_flag=SHOW_FLAG, save_flag=SAVE_FLAG, figure_dir=FOLDER_FIGURES,
                  figure_format=FIG_FORMAT):
        # plot connectivity
        self._plot_connectivity(head.connectivity, show_flag, save_flag, figure_dir, figure_format)
        self._plot_connectivity_stats(head.connectivity, show_flag, save_flag, figure_dir, figure_format)
        # plot sensor gain_matrixs
        count = 1
        for s_type in Sensors.SENSORS_TYPES:
            sensors = getattr(head, "sensors" + s_type)
            if isinstance(sensors, (list, Sensors)):
                sensors_list = ensure_list(sensors)
                if len(sensors_list) > 0:
                    for s in sensors_list:
                        count = self._plot_sensors(s, head.connectivity.region_labels, count, show_flag,
                                                   save_flag, figure_dir, figure_format)

    def plot_model_configuration(self, model_configuration, n_regions=None, regions_labels=[], x0_indices=[],
                                 e_indices=[], disease_indices=[], title="Model Configuration Overview", figure_name='',
                                 show_flag=SHOW_FLAG, save_flag=SAVE_FLAG, figure_dir=FOLDER_FIGURES,
                                 figure_format=FIG_FORMAT, figsize=VERY_LARGE_SIZE):
        if n_regions == None:
            n_regions = len(model_configuration.x0_values)
        if regions_labels == []:
            regions_labels = numpy.array([str(ii) for ii in range(n_regions)])
        disease_indices = numpy.unique(numpy.concatenate((x0_indices, e_indices, disease_indices), axis=0)).tolist()
        plot_dict_list = model_configuration.prepare_for_plot(x0_indices, e_indices, disease_indices)
        return self.plot_in_columns(plot_dict_list, regions_labels, width_ratios=[],
                                    left_ax_focus_indices=disease_indices, right_ax_focus_indices=disease_indices,
                                    title=title, figure_name=figure_name, show_flag=show_flag, save_flag=save_flag,
                                    figure_dir=figure_dir, figure_format=figure_format, figsize=figsize)

    def plot_statistical_model(self, statistical_model, figure_name="", figure_dir=FOLDER_FIGURES, save_flag=SAVE_FLAG,
                               show_flag=SHOW_FLAG, figure_format=FIG_FORMAT):
        _, ax = pyplot.subplots(len(statistical_model.parameters), 2, figsize=VERY_LARGE_PROTRAIT)
        for ip, p in enumerate(statistical_model.parameters.values()):
            ax[ip] = p.plot(ax=ax[ip], lgnd=False)
            # ax.append(p.plot_stochastic_parameter(lgnd=False))
        self._save_figure(save_flag, pyplot.gcf(), figure_name, figure_dir, figure_format)
        self._check_show(show_flag)
        return ax, pyplot.gcf()

    def _timeseries_plot(self, time, n_vars, nTS, n_times, time_units, subplots, offset=0.0, data_lims=[]):
        def_time = range(n_times)
        if not (isinstance(time, numpy.ndarray) and (len(time) == n_times)):
            time = def_time
            warning("Input time doesn't match data! Setting a default time step vector!")
        data_fun = lambda data, time, icol: (data[icol], time, icol)

        def plot_ts(x, iTS, colors, alphas, labels):
            x, time, ivar = x
            try:
                return pyplot.plot(time, x[:, iTS], colors[iTS], label=labels[iTS], alpha=alphas[iTS])
            except:
                warning("Cannot convert labels' strings for line labels!")
                return pyplot.plot(time, x[:, iTS], colors[iTS], label=str(iTS), alpha=alphas[iTS])

        def plot_ts_raster(x, iTS, colors, alphas, labels, offset):
            x, time, ivar = x
            try:
                return pyplot.plot(time, -x[:, iTS] + offset[ivar] * iTS, colors[iTS], label=labels[iTS],
                                   alpha=alphas[iTS])
            except:
                warning("Cannot convert labels' strings for line labels!")
                return pyplot.plot(time, -x[:, iTS] + offset[ivar] * iTS, colors[iTS], label=str(iTS),
                                   alpha=alphas[iTS])

        def axlabels_ts(labels, n_rows, irow, iTS):
            if irow == n_rows:
                pyplot.gca().set_xlabel("Time (" + time_units + ")")
            if n_rows > 1:
                try:
                    pyplot.gca().set_ylabel(str(iTS) + "." + labels[iTS])
                except:
                    warning("Cannot convert labels' strings for y axis labels!")
                    pyplot.gca().set_ylabel(str(iTS))

        def axlimits_ts(data_lims, time, icol):
            pyplot.gca().set_xlim([time[0], time[-1]])
            if n_rows > 1:
                pyplot.gca().set_ylim([data_lims[icol][0], data_lims[icol][1]])
            else:
                pyplot.autoscale(enable=True, axis='y', tight=True)

        def axYticks(labels, offset, nTS):
            pyplot.gca().set_yticks((offset * numpy.array([range(nTS)])).tolist())
            # try:
            #     pyplot.gca().set_yticklabels(labels)
            # except:
            #     warning("Cannot convert region labels' strings for y axis ticks!")

        if offset > 0.0:
            offsets = offset * numpy.array([numpy.diff(ylim) for ylim in data_lims]).flatten()
            plot_lines = lambda x, iTS, colors, alphas, labels: plot_ts_raster(x, iTS, colors, alphas, labels, offsets)
        else:
            plot_lines = lambda x, iTS, colors, alphas, labels: plot_ts(x, iTS, colors, alphas, labels)
        if subplots:
            n_rows = nTS
            def_alpha = 1.0
        else:
            n_rows = 1
            def_alpha = 0.5
        subtitle_col = lambda subtitle: pyplot.gca().set_title(subtitle)
        subtitle = lambda iTS, labels: None
        projection = None
        axlabels = lambda labels, vars, n_vars, n_rows, irow, iTS: axlabels_ts(labels, n_rows, irow, iTS)
        axlimits = lambda data_lims, time, n_vars, icol: axlimits_ts(data_lims, time, icol)
        loopfun = lambda nTS, n_rows, icol: range(nTS)
        return data_fun, time, plot_lines, projection, n_rows, n_vars, def_alpha, loopfun, \
               subtitle, subtitle_col, axlabels, axlimits, axYticks

    def _trajectories_plot(self, n_dims, nTS, nSamples, subplots):
        data_fun = lambda data, time, icol: data

        def plot_traj_2D(x, iTS, colors, alphas, labels):
            x, y = x
            try:
                return pyplot.plot(x[:, iTS], y[:, iTS], colors[iTS], label=labels[iTS], alpha=alphas[iTS])
            except:
                warning("Cannot convert labels' strings for line labels!")
                return pyplot.plot(x[:, iTS], y[:, iTS], colors[iTS], label=str(iTS), alpha=alphas[iTS])

        def plot_traj_3D(x, iTS, colors, alphas, labels):
            x, y, z = x
            try:
                return pyplot.plot(x[:, iTS], y[:, iTS], z[:, iTS], colors[iTS], label=labels[iTS], alpha=alphas[iTS])
            except:
                warning("Cannot convert labels' strings for line labels!")
                return pyplot.plot(x[:, iTS], y[:, iTS], z[:, iTS], colors[iTS], label=str(iTS), alpha=alphas[iTS])

        def subtitle_traj(labels, iTS):
            try:
                pyplot.gca().set_title(str(iTS) + "." + labels[iTS])
            except:
                warning("Cannot convert labels' strings for subplot titles!")
                pyplot.gca().set_title(str(iTS))

        def axlabels_traj(vars, n_vars):
            pyplot.gca().set_xlabel(vars[0])
            pyplot.gca().set_ylabel(vars[1])
            if n_vars > 2:
                pyplot.gca().set_zlabel(vars[2])

        def axlimits_traj(data_lims, n_vars):
            pyplot.gca().set_xlim([data_lims[0][0], data_lims[0][1]])
            pyplot.gca().set_ylim([data_lims[1][0], data_lims[1][1]])
            if n_vars > 2:
                pyplot.gca().set_zlim([data_lims[2][0], data_lims[2][1]])

        if n_dims == 2:
            plot_lines = lambda x, iTS, colors, labels, alphas: plot_traj_2D(x, iTS, colors, labels, alphas)
            projection = None
        elif n_dims == 3:
            plot_lines = lambda x, iTS, colors, labels, alphas: plot_traj_3D(x, iTS, colors, labels, alphas)
            projection = '3d'
        else:
            raise_value_error("Data dimensions are neigher 2D nor 3D!, but " + str(n_dims) + "D!")
        n_rows = 1
        n_cols = 1
        if subplots is None:
            if nSamples > 1:
                n_rows = int(numpy.floor(numpy.sqrt(nTS)))
                n_cols = int(numpy.ceil((1.0 * nTS) / n_rows))
        elif isinstance(subplots, (list, tuple)):
            n_rows = subplots[0]
            n_cols = subplots[1]
            if n_rows * n_cols < nTS:
                raise_value_error("Not enough subplots for all time series:"
                                  "\nn_rows * n_cols = product(subplots) = product(" + str(subplots) + " = "
                                  + str(n_rows * n_cols) + "!")
        if n_rows * n_cols > 1:
            def_alpha = 0.5
            subtitle = lambda labels, iTS: subtitle_traj(labels, iTS)
            subtitle_col = lambda subtitles, icol: None
        else:
            def_alpha = 1
            subtitle = lambda: None
            subtitle_col = lambda subtitles, icol: pyplot.gca().set_title(pyplot.gcf().title)
        axlabels = lambda labels, vars, n_vars, n_rows, irow, iTS: axlabels_traj(vars, n_vars)
        axlimits = lambda data_lims, time, n_vars, icol: axlimits_traj(data_lims, n_vars)
        loopfun = lambda nTS, n_rows, icol: range(icol, nTS, n_rows)
        return data_fun, plot_lines, projection, n_rows, n_cols, def_alpha, loopfun, \
               subtitle, subtitle_col, axlabels, axlimits

    def plot_timeseries(self, data_dict, time=None, mode="ts", subplots=None, special_idx=None, subtitles=[],
                        offset=1.0, time_units="ms", title='Time series', figure_name=None, labels=None,
                        figure_dir=FOLDER_FIGURES, figsize=LARGE_SIZE, figure_format=FIG_FORMAT,
                        show_flag=SHOW_FLAG, save_flag=SAVE_FLAG):
        n_vars = len(data_dict)
        vars = data_dict.keys()
        data = data_dict.values()
        data_lims = []
        for id, d in enumerate(data):
            data_lims.append([d.min(), d.max()])
            if isequal_string(mode, "raster"):
                data[id] = zscore(d, axis=None)
        data_shape = data[0].shape
        n_times, nTS = data_shape[:2]
        if len(data_shape) > 2:
            nSamples = data_shape[2]
        else:
            nSamples = 1
        if len(subtitles) == 0:
            subtitles = vars
        if labels is None:
            labels = numpy.array(range(nTS)).astype(str)
        if isequal_string(mode, "traj"):
            data_fun, plot_lines, projection, n_rows, n_cols, def_alpha, loopfun, \
            subtitle, subtitle_col, axlabels, axlimits = self._trajectories_plot(n_vars, nTS, nSamples, subplots)
        else:
            if isequal_string(mode, "raster"):
                data_fun, time, plot_lines, projection, n_rows, n_cols, def_alpha, loopfun, \
                subtitle, subtitle_col, axlabels, axlimits, axYticks = \
                    self._timeseries_plot(time, n_vars, nTS, n_times, time_units, 0, offset, data_lims)

            else:
                data_fun, time, plot_lines, projection, n_rows, n_cols, def_alpha, loopfun, \
                subtitle, subtitle_col, axlabels, axlimits, axYticks = \
                    self._timeseries_plot(time, n_vars, nTS, n_times, time_units, ensure_list(subplots)[0])
        alpha_ratio = 1.0 / nSamples
        colors = numpy.array(['k'] * nTS)
        alphas = numpy.maximum(numpy.array([def_alpha] * nTS) * alpha_ratio, 0.1)
        if special_idx is not None:
            colors[special_idx] = 'r'
            alphas[special_idx] = numpy.maximum(alpha_ratio, 0.1)
        lines = []
        pyplot.figure(title, figsize=figsize)
        pyplot.hold(True)
        axes = []
        for icol in range(n_cols):
            if n_rows == 1:
                # If there are no more rows, create axis, and set its limits, labels and possible subtitle
                axes += ensure_list(pyplot.subplot(n_rows, n_cols, icol + 1, projection=projection))
                axlimits(data_lims, time, n_vars, icol)
                axlabels(labels, vars, n_vars, n_rows, 1, 0)
                pyplot.gca().set_title(subtitles[icol])
            for iTS in loopfun(nTS, n_rows, icol):
                if n_rows > 1:
                    # If there are more rows, create axes, and set their limits, labels and possible subtitles
                    axes += ensure_list(pyplot.subplot(n_rows, n_cols, iTS + 1, projection=projection))
                    subtitle(labels, iTS)
                    axlimits(data_lims, time, n_vars, icol)
                    axlabels(labels, vars, n_vars, n_rows, (iTS % n_rows) + 1, iTS)
                lines += ensure_list(plot_lines(data_fun(data, time, icol), iTS, colors, alphas, labels))
            if isequal_string(mode, "raster"):  # set yticks as labels if this is a raster plot
                # axYticks(labels, offset, nTS)
                pyplot.gca().invert_yaxis()
        if MOUSEHOOVER:
            for line in lines:
                # datacursor( lines[0], formatter='{label}'.format, bbox=dict(fc='white'),
                #           arrowprops=dict(arrowstyle='simple', fc='white', alpha=0.5) )    #hover=True
                HighlightingDataCursor(line, formatter='{label}'.format, bbox=dict(fc='white'),
                                       arrowprops=dict(arrowstyle='simple', fc='white', alpha=0.5))
        self._save_figure(save_flag, pyplot.gcf(), figure_name, figure_dir, figure_format)
        self._check_show(show_flag)
        return pyplot.gcf(), axes, lines

    def plot_raster(self, data_dict, time, time_units="ms", special_idx=None, title='Raster plot', subtitles=[],
                    offset=1.0, figure_name=None, labels=None, figure_dir=FOLDER_FIGURES, figsize=VERY_LARGE_SIZE,
                    figure_format=FIG_FORMAT, show_flag=SHOW_FLAG, save_flag=False):
        return self.plot_timeseries(data_dict, time, "raster", None, special_idx, subtitles, offset, time_units, title,
                                    figure_name, labels, figure_dir, figsize, figure_format, show_flag, save_flag)

    def plot_trajectories(self, data_dict, subtitles=None, special_idx=None, title='State space trajectories',
                          figure_name=None, labels=None, figure_dir=FOLDER_FIGURES, figsize=LARGE_SIZE,
                          figure_format=FIG_FORMAT,
                          show_flag=SHOW_FLAG, save_flag=SAVE_FLAG):
        return self.plot_timeseries(data_dict, [], "traj", subtitles, special_idx, title=title, figure_name=figure_name,
                                    labels=labels, figure_dir=figure_dir, figsize=figsize, figure_format=figure_format,
                                    show_flag=show_flag, save_flag=save_flag)

    def plot_spectral_analysis_raster(self, time, data, time_units="ms", freq=None, special_idx=None,
                                      title='Spectral Analysis', figure_name=None, labels=None, show_flag=SHOW_FLAG,
                                      save_flag=SAVE_FLAG, figure_dir=FOLDER_FIGURES, figure_format=FIG_FORMAT,
                                      figsize=VERY_LARGE_SIZE, **kwargs):
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

    def plot_sim_results(self, model, seizure_indices, res, sensorsSEEG=None, hpf_flag=False, trajectories_plot=False,
                         spectral_raster_plot=False, region_labels=None, save_flag=SAVE_FLAG, show_flag=SHOW_FLAG,
                         figure_dir=FOLDER_FIGURES, figure_format=FIG_FORMAT, **kwargs):
        if isinstance(model, EpileptorDP2D):
            self.plot_timeseries({'x1(t)': res['x1'], 'z(t)': res['z']}, res['time'],
                                 time_units=res.get('time_units', "ms"),
                                 special_idx=seizure_indices, title=model._ui_name + ": Simulated TAVG",
                                 save_flag=save_flag, show_flag=show_flag, figure_dir=figure_dir,
                                 figure_format=figure_format,
                                 labels=region_labels, figsize=VERY_LARGE_SIZE)
            self.plot_raster({'x1(t)': res['x1']}, res['time'], time_units=res.get('time_units', "ms"),
                             special_idx=seizure_indices,
                             title=model._ui_name + ": Simulated x1 rasterplot", offset=5.0, labels=region_labels,
                             save_flag=save_flag, show_flag=show_flag, figure_dir=figure_dir,
                             figure_format=figure_format,
                             figsize=VERY_LARGE_SIZE)
        else:
            self.plot_timeseries({'LFP(t)': res['lfp'], 'z(t)': res['z']}, res['time'],
                                 time_units=res.get('time_units', "ms"),
                                 special_idx=seizure_indices, title=model._ui_name + ": Simulated LFP-z",
                                 save_flag=save_flag, show_flag=show_flag, figure_dir=figure_dir,
                                 figure_format=figure_format,
                                 labels=region_labels, figsize=VERY_LARGE_SIZE)
            self.plot_timeseries({'x1(t)': res['x1'], 'y1(t)': res['y1']}, res['time'],
                                 time_units=res.get('time_units', "ms"),
                                 special_idx=seizure_indices, title=model._ui_name + ": Simulated pop1",
                                 save_flag=save_flag, show_flag=show_flag, figure_dir=figure_dir,
                                 figure_format=figure_format,
                                 labels=region_labels, figsize=VERY_LARGE_SIZE)
            self.plot_timeseries({'x2(t)': res['x2'], 'y2(t)': res['y2'], 'g(t)': res['g']}, res['time'],
                                 time_units=res.get('time_units', "ms"), special_idx=seizure_indices,
                                 title=model._ui_name + ": Simulated pop2-g",
                                 save_flag=save_flag, show_flag=show_flag, figure_dir=figure_dir,
                                 figure_format=figure_format,
                                 labels=region_labels, figsize=VERY_LARGE_SIZE)
            start_plot = int(numpy.round(0.01 * res['lfp'].shape[0]))
            self.plot_raster({'lfp': res['lfp'][start_plot:, :]}, res['time'][start_plot:],
                             time_units=res.get('time_units', "ms"), special_idx=seizure_indices,
                             title=model._ui_name + ": Simulated LFP rasterplot", offset=10.0, labels=region_labels,
                             save_flag=save_flag, show_flag=show_flag, figure_dir=figure_dir,
                             figure_format=figure_format,
                             figsize=VERY_LARGE_SIZE)
        if isinstance(model, EpileptorDPrealistic):
            self.plot_timeseries({'1/(1+exp(-10(z-3.03))': 1 / (1 + numpy.exp(-10 * (res['z'] - 3.03))),
                                  'slope': res['slope_t'], 'Iext2': res['Iext2_t']}, res['time'],
                                 time_units=res.get('time_units', "ms"), special_idx=seizure_indices,
                                 title=model._ui_name + ": Simulated controlled parameters", labels=region_labels,
                                 save_flag=save_flag, show_flag=show_flag, figure_dir=figure_dir,
                                 figure_format=figure_format,
                                 figsize=VERY_LARGE_SIZE)
            self.plot_timeseries({'x0_values': res['x0_t'], 'Iext1': res['Iext1_t'], 'K': res['K_t']}, res['time'],
                                 time_units=res.get('time_units', "ms"), special_idx=seizure_indices,
                                 title=model._ui_name + ": Simulated parameters", labels=region_labels,
                                 save_flag=save_flag, show_flag=show_flag, figure_dir=figure_dir,
                                 figure_format=figure_format,
                                 figsize=VERY_LARGE_SIZE)
        if trajectories_plot:
            self.plot_trajectories({'x1': res['x1'], 'z': res['z']}, special_idx=seizure_indices,
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
                self.plot_raster({'SEEG': res['SEEG' + str(i)][start_plot:, :]}, res['time'][start_plot:],
                                 time_units=res.get('time_units', "ms"), title=title,
                                 offset=0.0, save_flag=save_flag, show_flag=show_flag, figure_dir=figure_dir,
                                 figure_format=figure_format, labels=sensorsSEEG[i].labels, figsize=VERY_LARGE_SIZE)

    def plot_lsa(self, disease_hypothesis, model_configuration, weighted_eigenvector_sum, eigen_vectors_number,
                 region_labels=[], pse_results=None, title="Hypothesis Overview", figure_dir=FOLDER_FIGURES,
                 figure_format=FIG_FORMAT, show_flag=SHOW_FLAG, save_flag=SAVE_FLAG):

        hyp_dict_list = disease_hypothesis.prepare_for_plot(model_configuration.model_connectivity)
        model_config_dict_list = model_configuration.prepare_for_plot()[:2]

        model_config_dict_list += hyp_dict_list
        plot_dict_list = model_config_dict_list

        if pse_results is not None and isinstance(pse_results, dict):
            fig_name = disease_hypothesis.name + " PSE " + title
            ind_ps = len(plot_dict_list) - 2
            for ii, value in enumerate(["lsa_propagation_strengths", "e_values", "x0_values"]):
                ind = ind_ps - ii
                if ind >= 0:
                    if pse_results.get(value, False).any():
                        plot_dict_list[ind]["data_samples"] = pse_results.get(value)
                        plot_dict_list[ind]["plot_type"] = "vector_violin"

        else:
            fig_name = disease_hypothesis.name + " " + title

        description = ""
        if weighted_eigenvector_sum:
            description = "LSA PS: absolut eigenvalue-weighted sum of "
            if eigen_vectors_number is not None:
                description += "first " + str(eigen_vectors_number) + " "
            description += "eigenvectors has been used"

        return self.plot_in_columns(plot_dict_list, region_labels, width_ratios=[],
                                    left_ax_focus_indices=disease_hypothesis.get_all_disease_indices(),
                                    right_ax_focus_indices=disease_hypothesis.lsa_propagation_indices,
                                    description=description, title=title, figure_name=fig_name,
                                    figure_dir=figure_dir,
                                    figure_format=figure_format,
                                    show_flag=show_flag, save_flag=save_flag)

    def plot_state_space(self, model_config, region_labels, special_idx, model, zmode, figure_name,
                         approximations=False, show_flag=SHOW_FLAG, save_flag=SAVE_FLAG, figure_dir=FOLDER_FIGURES,
                         figure_format=FIG_FORMAT, **kwargs):
        add_name = " " + "Epileptor " + model + " z-" + str(zmode)
        figure_name = figure_name + add_name

        # Fixed parameters for all regions:
        x1eq = model_config.x1EQ
        zeq = model_config.zEQ
        x0 = a = b = d = yc = slope = Iext1 = Iext2 = s = 0.0
        for p in ["x0", "a", "b", "d", "yc", "slope", "Iext1", "Iext2", "s"]:
            exec (p + " = numpy.mean(model_config." + p + ")")
        # x0 = np.mean(model_config.x0)
        # a = np.mean(model_config.a)
        # b = np.mean(model_config.b)
        # d = np.mean(model_config.d)
        # yc = np.mean(model_config.yc)
        # slope = np.mean(model_config.slope)
        # Iext1 = np.mean(model_config.Iext1)
        # Iext2 = np.mean(model_config.Iext2)
        # s = np.mean(model_config.s)

        fig = pyplot.figure(figure_name, figsize=FIG_SIZE)

        # Lines:
        x1 = numpy.linspace(-2.0, 1.0, 100)
        if isequal_string(model, "2d"):
            y1 = yc
        else:
            y1 = calc_eq_y1(x1, yc, d=d)
        # x1 nullcline:
        zX1 = calc_fx1(x1, z=0, y1=y1, Iext1=Iext1, slope=slope, a=a, b=b, d=d, tau1=1.0, x1_neg=True, model=model,
                       x2=0.0)  # yc + Iext1 - x1 ** 3 - 2.0 * x1 ** 2
        x1null, = pyplot.plot(x1, zX1, 'b-', label='x1 nullcline', linewidth=1)
        ax = pyplot.gca()
        ax.axes.hold(True)
        # z nullcines
        # center point (critical equilibrium point) without approximation:
        # zsq0 = yc + Iext1 - x1sq0 ** 3 - 2.0 * x1sq0 ** 2
        x0e = calc_x0_val_to_model_x0(X0_CR_DEF, yc, Iext1, a=a, b=b, d=d, zmode=zmode)
        x0ne = calc_x0_val_to_model_x0(X0_DEF, yc, Iext1, a=a, b=b, d=d, zmode=zmode)
        zZe = calc_fz(x1, z=0.0, x0=x0e, tau1=1.0, tau0=1.0, zmode=zmode)  # for epileptogenic regions
        zZne = calc_fz(x1, z=0.0, x0=x0ne, tau1=1.0, tau0=1.0, zmode=zmode)  # for non-epileptogenic regions
        zE1null, = pyplot.plot(x1, zZe, 'g-', label='z nullcline at critical point (e_values=1)', linewidth=1)
        zE2null, = pyplot.plot(x1, zZne, 'g--', label='z nullcline for e_values=0', linewidth=1)
        if approximations:
            # The point of the linear approximation (1st order Taylor expansion)
            x1LIN = def_x1lin(X1_DEF, X1_EQ_CR_DEF, len(region_labels))
            x1SQ = X1_EQ_CR_DEF
            x1lin0 = numpy.mean(x1LIN)
            # The point of the square (parabolic) approximation (2nd order Taylor expansion)
            x1sq0 = numpy.mean(x1SQ)
            # approximations:
            # linear:
            x1lin = numpy.linspace(-5.5 / 3.0, -3.5 / 3, 30)
            # x1 nullcline after linear approximation:
            # yc + Iext1 + 2.0 * x1lin0 ** 3 + 2.0 * x1lin0 ** 2 - \
            # (3.0 * x1lin0 ** 2 + 4.0 * x1lin0) * x1lin  # x1
            zX1lin = calc_fx1_2d_taylor(x1lin, x1lin0, z=0, y1=yc, Iext1=Iext1, slope=slope, a=a, b=b, d=d, tau1=1.0,
                                        x1_neg=None, order=2)  #
            # center point without approximation:
            # zlin0 = yc + Iext1 - x1lin0 ** 3 - 2.0 * x1lin0 ** 2
            # square:
            x1sq = numpy.linspace(-5.0 / 3, -1.0, 30)
            # x1 nullcline after parabolic approximation: + 2.0 * x1sq ** 2 + 16.0 * x1sq / 3.0 + yc + Iext1 + 64.0 / 27.0
            zX1sq = calc_fx1_2d_taylor(x1sq, x1sq0, z=0, y1=yc, Iext1=Iext1, slope=slope, a=a, b=b, d=d, tau1=1.0,
                                       x1_neg=None, order=3, shape=x1sq.shape)
            sq, = pyplot.plot(x1sq, zX1sq, 'm--', label='Parabolic local approximation', linewidth=2)
            lin, = pyplot.plot(x1lin, zX1lin, 'c--', label='Linear local approximation', linewidth=2)
            pyplot.legend(handles=[x1null, zE1null, zE2null, lin, sq])
        else:
            pyplot.legend(handles=[x1null, zE1null, zE2null])

        # Points:
        ii = range(len(region_labels))
        if special_idx is None:
            ii = numpy.delete(ii, special_idx)
        points = []
        for i in ii:
            point, = pyplot.plot(x1eq[i], zeq[i], '*', mfc='k', mec='k',
                                 ms=10, alpha=0.3, label=str(i) + '.' + region_labels[i])
            points.append(point)
        if special_idx is None:
            for i in special_idx:
                point, = pyplot.plot(x1eq[i], zeq[i], '*', mfc='r', mec='r', ms=10, alpha=0.8,
                                     label=str(i) + '.' + region_labels[i])
                points.append(point)
        # ax.plot(x1lin0, zlin0, '*', mfc='r', mec='r', ms=10)
        # ax.axes.text(x1lin0 - 0.1, zlin0 + 0.2, 'e_values=0.0', fontsize=10, color='r')
        # ax.plot(x1sq0, zsq0, '*', mfc='m', mec='m', ms=10)
        # ax.axes.text(x1sq0, zsq0 - 0.2, 'e_values=1.0', fontsize=10, color='m')

        # Vector field
        tau1 = kwargs.get("tau1", TAU1_DEF)
        tau0 = kwargs.get("tau0", TAU0_DEF)
        X1, Z = numpy.meshgrid(numpy.linspace(-2.0, 1.0, 41), numpy.linspace(0.0, 6.0, 31), indexing='ij')
        if isequal_string(model, "2d"):
            y1 = yc
            x2 = 0.0
        else:
            y1 = calc_eq_y1(X1, yc, d=d)
            x2 = 0.0  # as a simplification for faster computation without important consequences
            # x2 = calc_eq_x2(Iext2, y2eq=None, zeq=X1, x1eq=Z, s=s)[0]
        fx1 = calc_fx1(X1, Z, y1=y1, Iext1=Iext1, slope=slope, a=a, b=b, d=d, tau1=tau1, x1_neg=None, model=model,
                       x2=x2)
        fz = calc_fz(X1, Z, x0=x0, tau1=tau1, tau0=tau0, zmode=zmode)
        C = numpy.abs(fx1) + numpy.abs(fz)
        pyplot.quiver(X1, Z, fx1, fz, C, edgecolor='k', alpha=.5, linewidth=.5)
        pyplot.contour(X1, Z, fx1, 0, colors='b', linestyles="dashed")

        ax.set_title("Epileptor states pace at the x1-z phase plane of the" + add_name)
        ax.axes.autoscale(tight=True)
        ax.axes.set_ylim([0.0, 6.0])
        ax.axes.set_xlabel('x1')
        ax.axes.set_ylabel('z')
        if MOUSEHOOVER:
            # datacursor( lines[0], formatter='{label}'.format, bbox=dict(fc='white'),
            #           arrowprops=dict(arrowstyle='simple', fc='white', alpha=0.5) )    #hover=True

            HighlightingDataCursor(points[0], formatter='{label}'.format, bbox=dict(fc='white'),
                                   arrowprops=dict(arrowstyle='simple', fc='white', alpha=0.5))
        if len(fig.get_label()) == 0:
            fig.set_label(figure_name)
        else:
            figure_name = fig.get_label().replace(": ", "_").replace(" ", "_").replace("\t", "_")

        self._save_figure(save_flag, figure_dir=figure_dir, figure_format=figure_format, figure_name=figure_name)
        self._check_show(show_flag)

    def plot_fit_results(self, model_inversion_service, ests, samples, statistical_model, signals, time=None,
                         seizure_indices=None, x1_str="x1",
                         x0_str="x0", mc_str="MC", signals_str="fit_signals", sig_str="sig", eps_str="eps",
                         trajectories_plot=True, save_flag=SAVE_FLAG, show_flag=SHOW_FLAG, figure_dir=FOLDER_FIGURES,
                         figure_format=FIG_FORMAT, **kwargs):
        # plot scalar parameters in pair plots
        # TODO: move these 2 plots here
        model_inversion_service.parameters_pair_plots(samples,
                                                      kwargs.get("pair_plot_params",
                                                                 ["tau1", "tau0", "K", "sig_eq", "sig_init", "sig",
                                                                  "eps", "scale_signal",
                                                                  "offset_signal"]),
                                                      kwargs.get("skip_samples", 0),
                                                      title=statistical_model.name + " parameters samples")
        # plot region-wise parameters
        model_inversion_service.region_parameters_violin_plots(samples,
                                                               kwargs.get("params", ["x0", "x1eq", "x1init", "zinit"]),
                                                               skip_samples=kwargs.get("skip_samples", 0),
                                                               per_chain=kwargs.get("violin_plot_per_chain", False),
                                                               figure_name=statistical_model.name + " regions parameters samples")
        if time is None:
            time = numpy.array(range(signals.shape[0]))
        time = time.flatten()
        sig_prior = statistical_model.parameters["sig"].mean
        eps_prior = statistical_model.parameters["eps"].mean
        x0_prior = str(model_inversion_service.x0[statistical_model.active_regions])
        MC_prior = statistical_model.parameters["MC"].mean
        K_prior = statistical_model.parameters["K"].mean
        region_labels = kwargs.get("regions_labels", model_inversion_service.region_labels)
        if statistical_model.observation_model.find("seeg") >= 0:
            sensor_labels = kwargs.get("signals_labels", None)[model_inversion_service.signals_inds]
        else:
            sensor_labels = region_labels[model_inversion_service.signals_inds]
        for id_est, est in enumerate(ensure_list(ests)):
            name = statistical_model.name + kwargs.get("_id_est", "")
            self.plot_raster(sort_dict({'observation signals': signals,
                                        'observation signals fit': samples[signals_str].T}), time,
                             special_idx=seizure_indices, time_units=est.get('time_units', "ms"),
                             title=name + ": Observation signals vs fit rasterplot",
                             subtitles=['observation signals ' +
                                        '\nobservation noise eps_prior =  ' + str(eps_prior) + " eps_post =" + str(
                                 est[eps_str]),
                                        'observation signals fit'], offset=1.0,
                             labels=sensor_labels, save_flag=save_flag, show_flag=show_flag,
                             figure_dir=figure_dir, figure_format=figure_format, figsize=VERY_LARGE_SIZE)
            self.plot_raster(sort_dict({'x1': samples[x1_str].T, 'z': samples["z"].T}), time,
                             special_idx=seizure_indices, time_units=est.get('time_units', "ms"),
                             title=name + ": Hidden states fit rasterplot",
                             subtitles=['hidden state x1',
                                        'hidden state z' + '\ndynamic noise sig_prior = ' + str(sig_prior) +
                                        " sig_post = " + str(est[sig_str])], offset=1.0,
                             labels=region_labels, save_flag=save_flag,
                             show_flag=show_flag, figure_dir=figure_dir,
                             figure_format=figure_format, figsize=VERY_LARGE_SIZE)
            if trajectories_plot:
                title = name + ': Fit hidden state space trajectories'
                title += "\n prior x0: " + x0_prior
                x0 = est[x0_str]
                if len(x0) > statistical_model.n_active_regions:
                    x0 = x0[statistical_model.active_regions]
                title += "\n x0 fit: " + str(x0)
                self.plot_trajectories({'x1': samples[x1_str].T, 'z': samples['z'].T}, special_idx=seizure_indices,
                                       title=title, labels=model_inversion_service.region_labels, show_flag=show_flag,
                                       save_flag=save_flag,
                                       figure_dir=figure_dir, figure_format=figure_format, figsize=VERY_LARGE_SIZE)
            # plot connectivity
            conn_figure_name = name + "Model Connectivity"
            pyplot.figure(conn_figure_name, VERY_LARGE_SIZE)
            # plot_regions2regions(conn.weights, conn.region_labels, 121, "weights")
            self.plot_regions2regions(MC_prior, region_labels, 121,
                                      "Prior Model Connectivity" + "\nglobal scaling prior: K = " + str(K_prior))
            self.plot_regions2regions(est[mc_str], region_labels, 122,
                                      "Posterior Model  Connectivity" + "\nglobal scaling fit: K = " + str(est["K"]))
            self._save_figure(save_flag, pyplot.gcf(), conn_figure_name, figure_dir, figure_format)
            self._check_show(show_flag=show_flag)

    def _prepare_distribution_axes(self, distribution, loc=0.0, scale=1.0, x=numpy.array([]), ax=None, linestyle="-",
                                   lgnd=True):
        if len(x) < 1:
            x = linspace_broadcast(distribution.scipy(distribution.loc, distribution.scale).ppf(0.01),
                                   distribution.scipy(distribution.loc, distribution.scale).ppf(0.99), 100)
        if x is not None:
            pdf = distribution.scipy(loc, scale).pdf(x)
            if ax is None:
                _, ax = pyplot.subplots(1, 1)
            for ip, (xx, pp) in enumerate(zip(x.T, pdf.T)):
                if xx.shape != pp.shape:
                    print("WTF?")
                ax.plot(xx.T, pp.T, linestyle=linestyle, linewidth=1, label=str(ip), alpha=0.5)
            if lgnd:
                pyplot.legend()
            return ax
        else:
            raise_value_error("Distribution parameters do not broadcast!")

    def plot_distribution(self, distribution, loc=0.0, scale=1.0, x=numpy.array([]), ax=None, linestyle="-", lgnd=True,
                          figure_name="", figure_dir=FOLDER_FIGURES, save_flag=SAVE_FLAG, show_flag=SHOW_FLAG,
                          figure_format=FIG_FORMAT):
        ax = self._prepare_distribution_axes(loc, scale, x, ax, linestyle, lgnd)
        ax.set_title(distribution.type + " distribution")
        self._save_figure(save_flag, pyplot.gcf(), figure_name, figure_dir, figure_format)
        self._check_show(show_flag)
        return ax, pyplot.gcf()

    def _prepare_parameter_axes(self, parameter, x=numpy.array([]), ax=None, lgnd=True):
        if ax is None:
            _, ax = pyplot.subplots(1, 2)
        if len(x) < 1:
            x = linspace_broadcast(
                numpy.maximum(parameter.low, parameter.scipy(parameter.loc, parameter.scale).ppf(0.01)),
                numpy.minimum(parameter.high, parameter.scipy(parameter.loc, parameter.scale).ppf(0.99)), 100)
        if x is not None:
            plotter = Plotter()
            ax[0] = plotter._prepare_distribution_axes(parameter, parameter.loc, parameter.scale, x, ax[0], "-", lgnd)
            ax[0].set_title(parameter.name + ": " + parameter.type + " distribution")
            ax[1] = plotter._prepare_distribution_axes(parameter, 0.0, 1.0, (x - parameter.loc) / parameter.scale,
                                                       ax[1], "--", lgnd)
            ax[1].set_title(parameter.name + "_star: " + parameter.type + " distribution")
            return ax
        else:
            raise_value_error("Stochastic parameter's parameters do not broadcast!")

    def plot_stochastic_parameter(self, parameter, x=numpy.array([]), ax=None, lgnd=True, figure_name="",
                                  figure_dir=FOLDER_FIGURES,
                                  save_flag=SAVE_FLAG, show_flag=SHOW_FLAG, figure_format=FIG_FORMAT):
        ax = self._prepare_parameter_axes(parameter, x, ax, lgnd)
        if len(figure_name) < 1:
            figure_name = "parameter_" + parameter.name
        self._save_figure(save_flag, pyplot.gcf(), figure_name, figure_dir, figure_format)
        self._check_show(show_flag)
        return ax, pyplot.gcf()

    def plot_HMC(self, samples, skip_samples=0, title='HMC NUTS trace', figure_name=None, figure_dir=FOLDER_FIGURES,
                 figsize=LARGE_SIZE, figure_format=FIG_FORMAT, show_flag=SHOW_FLAG, save_flag=SAVE_FLAG):
        samples = ensure_list(samples)
        if len(samples) > 1:
            samples = list_of_dicts_to_dicts_of_ndarrays(samples)
        else:
            samples = samples[0]
        nuts = extract_dict_stringkeys(samples, '__', modefun="find")
        self.plots(nuts, shape=(2, 4), skip=skip_samples, xlabels={}, xscales={}, yscales={"stepsize__": "log"},
                   title=title, figure_name=figure_name, figure_dir=figure_dir, figure_format=figure_format,
                   figsize=figsize, show_flag=show_flag, save_flag=save_flag)
