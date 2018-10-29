
from tvb_fit.tvb_epilepsy.base.constants.config import FiguresConfig
import matplotlib
matplotlib.use(FiguresConfig().MATPLOTLIB_BACKEND)
from matplotlib import pyplot, gridspec


import numpy
from collections import OrderedDict

from tvb_fit.base.utils.log_error_utils import warning
from tvb_fit.base.utils.data_structures_utils import ensure_list, generate_region_labels, extract_dict_stringkeys
from tvb_fit.samplers.stan.stan_interface import merge_samples
from tvb_fit.plot.timeseries_plotter import TimeseriesPlotter


class ModelInversionPlotter(TimeseriesPlotter):

    def __init__(self, config=None):
        super(ModelInversionPlotter, self).__init__(config)
        self.marker = "o"
        self.print_ts_indices = False
        self.print_regions_indices = False

    def _params_stats_subtitles(self, params, stats):
        subtitles = list(params)
        if isinstance(stats, dict):
            for ip, param in enumerate(params):
                subtitles[ip] = subtitles[ip] + ": "
                for skey, sval in stats.items():
                    subtitles[ip] = subtitles[ip] + skey + "=" + str(sval[param]) + ", "
                subtitles[ip] = subtitles[ip][:-2]
        return subtitles

    def _params_stats_labels(self, param, stats, labels):
        subtitles = list(labels)
        if isinstance(stats, dict):
            n_params = len(stats.values()[0][param])
            if len(subtitles) == 1 and n_params > 1:
                subtitles = subtitles * n_params
            elif len(subtitles) == 0:
                subtitles = [""] * n_params
            for ip in range(n_params):
                if len(subtitles[ip]) > 0:
                    subtitles[ip] = subtitles[ip] + ": "
                for skey, sval in stats.items():
                    subtitles[ip] = subtitles[ip] + skey + "=" + str(sval[param][ip]) + ", "
                    subtitles[ip] = subtitles[ip][:-2]
        return subtitles

    def _parameters_pair_plots(self, samples_all, params=[],
                               stats=None, priors={}, truth={}, skip_samples=0, title='Parameters samples',
                               figure_name=None, figsize=FiguresConfig.VERY_LARGE_SIZE):
        subtitles = list(self._params_stats_subtitles(params, stats))
        samples = []
        samples_all = ensure_list(samples_all)
        params = [param for param in params if param in samples_all[0].keys()]
        for sample in samples_all:
            samples.append(extract_dict_stringkeys(sample, params, modefun="equal"))
        if len(samples) > 1:
            samples = merge_samples(samples)
        else:
            samples = samples[0]
            n_samples = (samples.values()[0]).shape[0]
            for p_key, p_val in samples.items():
                samples[p_key] = numpy.reshape(p_val, (1, n_samples))
        diagonal_plots = {}
        # for param_key in samples.keys():
        for p_key in params:
            diagonal_plots.update({p_key: [priors.get(p_key, ()), truth.get(p_key, ())]})

        return self.pair_plots(samples, params, diagonal_plots, True, skip_samples,
                               title, "chain/run ", subtitles, figure_name, figsize)

    def _region_parameters_violin_plots(self, samples_all, values=None, lines=None, stats=None,
                                        params=[], skip_samples=0, per_chain_or_run=False,
                                        labels=[], special_idx=None, figure_name="Regions parameters samples",
                                        figsize=FiguresConfig.VERY_LARGE_SIZE):
        if isinstance(values, dict):
            vals_fun = lambda param: values.get(param, numpy.array([]))
        else:
            vals_fun = lambda param: []
        if isinstance(lines, dict):
            lines_fun = lambda param: lines.get(param, numpy.array([]))
        else:
            lines_fun = lambda param: []
        samples_all = ensure_list(samples_all)
        params = [param for param in params if param in samples_all[0].keys()]
        samples = []
        for sample in samples_all:
            samples.append(extract_dict_stringkeys(sample, params, modefun="equal"))
        if len(labels) == 0:
            labels = generate_region_labels(samples[0].values()[0].shape[-1], labels, numbering=False)
        n_chains = len(samples_all)
        n_samples = samples[0].values()[0].shape[0]
        if n_samples > 1:
            violin_flag = True
        else:
            violin_flag = False
        if not per_chain_or_run and n_chains > 1:
            samples = [merge_samples(samples)]
            plot_samples = lambda s: numpy.concatenate(numpy.split(s[:, skip_samples:].T, n_chains, axis=2),
                                                       axis=1).squeeze().T
            plot_figure_name = lambda ichain: figure_name
        else:
            plot_samples = lambda s: s[skip_samples:]
            plot_figure_name = lambda ichain: figure_name + ": chain " + str(ichain + 1)
        params_labels = {}
        for ip, p in enumerate(params):
            if ip == 0:
                params_labels[p] = self._params_stats_labels(p, stats, labels)
            else:
                params_labels[p] = self._params_stats_labels(p, stats, "")
        n_params = len(params)
        if n_params > 9:
            warning("Number of subplots in column wise vector-violin-plots cannot be > 9 and it is "
                              + str(n_params) + "!")
        subplot_ind = 100 + n_params * 10
        figs = []
        for ichain, chain_sample in enumerate(samples):
            pyplot.figure(plot_figure_name(ichain), figsize=figsize)
            for ip, param in enumerate(params):
                fig = self.plot_vector_violin(plot_samples(chain_sample[param]), vals_fun(param),
                                              lines_fun(param), params_labels[param],
                                              subplot_ind + ip + 1, param, violin_flag=violin_flag,
                                              colormap="YlOrRd", show_y_labels=True,
                                              indices_red=special_idx, sharey=None)
            self._save_figure(pyplot.gcf(), None)
            self._check_show()
            figs.append(fig)
        return tuple(figs)

    def plot_fit_scalar_params_iters(self, samples_all, params=[], skip_samples=0, title_prefix="",
                                     subplot_shape=None, figure_name=None, figsize=FiguresConfig.LARGE_SIZE):
        if len(title_prefix) > 0:
            title_prefix = title_prefix + ": "
        title = title_prefix + " Parameters samples per iteration"
        samples_all = ensure_list(samples_all)
        params = [param for param in params if param in samples_all[0].keys()]
        samples = []
        for sample in samples_all:
            samples.append(extract_dict_stringkeys(sample, params, modefun="equal"))
        if len(samples) > 1:
            samples = merge_samples(samples)
        else:
            samples = samples[0]
        if subplot_shape is None:
            n_params = len(samples)
            # subplot_shape = self.rect_subplot_shape(n_params, mode="col")
            if n_params > 1:
                subplot_shape = (int(numpy.ceil(1.0*n_params/2)), 2)
            else:
                subplot_shape = (1, 1)
        n_chains_or_runs = samples.values()[0].shape[0]
        legend = {samples.keys()[0]: ["chain/run " + str(ii+1) for ii in range(n_chains_or_runs)]}
        return self.plots(samples, shape=subplot_shape, transpose=True, skip=skip_samples, xlabels={}, xscales={},
                          yscales={}, title=title, lgnd=legend, figure_name=figure_name, figsize=figsize)

    def plot_fit_scalar_params(self, samples, stats, probabilistic_model=None, params=[], skip_samples=0,
                               title_prefix=""):
        # plot scalar parameters in pair plots
        if len(title_prefix) > 0:
            title_prefix = title_prefix + ": "
        title = title_prefix + " Parameters samples"
        priors = {}
        truth = {}
        if probabilistic_model is not None:
            for p in params:
                pdf = probabilistic_model.get_prior_pdf(p)
                # TODO: a better hack than the following for when pdf returns p_mean, nan and p_mean is not a scalar
                if pdf[1] is numpy.nan:
                    pdf = list(pdf)
                    pdf[0] = numpy.mean(pdf[0])
                    pdf = tuple(pdf)
                priors.update({p: pdf})
                truth.update({p: numpy.nanmean(probabilistic_model.get_truth(p))})
        return self._parameters_pair_plots(samples, params, stats, priors, truth, skip_samples, title=title)

    def plot_fit_region_params(self, samples, stats=None, probabilistic_model=None,
                               params=[], special_idx=[], region_labels=[],
                               regions_mode="all", per_chain_or_run=False, skip_samples=0, title_prefix=""):
        if len(title_prefix) > 0:
            title_prefix = title_prefix + " "
        title_pair_plot = title_prefix + "Regions parameters pair plot"
        title_violin_plot = title_prefix + "Regions parameters samples"
        # We assume in this function that regions_inds run for all regions for the statistical model,
        # and either among all or only among active regions for samples, ests and stats, depending on regions_mode
        samples = ensure_list(samples)
        priors = {}
        truth = {}
        if probabilistic_model is not None:
            if regions_mode == "active":
                regions_inds = ensure_list(probabilistic_model.active_regions)
            else:
                regions_inds = range(probabilistic_model.number_of_regions)
            I = numpy.ones((probabilistic_model.number_of_regions, )) #, 1
            for param in params:
                pdf = ensure_list(probabilistic_model.get_prior_pdf(param))
                for ip, p in enumerate(pdf):
                    pdf[ip] = ((numpy.array(p) * I).T[regions_inds])
                priors.update({param: (pdf[0].squeeze(), pdf[1].squeeze())})
                truth.update({param: ((probabilistic_model.get_truth(param) * I)[regions_inds]).squeeze()}) #[:, 0]
        # plot region-wise parameters
        f1 = self._region_parameters_violin_plots(samples, truth, priors, stats, params, skip_samples,
                                                  per_chain_or_run=per_chain_or_run, labels=region_labels,
                                                  special_idx=special_idx, figure_name=title_violin_plot)
        f2 = []
        for p in params:
            if not(per_chain_or_run) and samples[0][p].shape[1] < 10:
                p_title_pair_plot = p + " " + title_pair_plot
                p_pair_plot_params = []
                p_pair_plot_samples = [{} for _ in range(len(samples))]
                for inode, label in enumerate(region_labels):
                    temp_name = "x0[" + label + "]"
                    p_pair_plot_params.append(temp_name)
                    for ichain, s in enumerate(samples):
                        inode[ichain].update({temp_name: s[p][:, inode]})
                        if probabilistic_model is not None:
                            priors.update({temp_name: (priors[p][0][inode], priors[p][1][inode])})
                            truth.update({temp_name: truth[p][inode]})
                f2.append(self._parameters_pair_plots(p_pair_plot_samples, p_pair_plot_params, None, priors, truth,
                                                 skip_samples, title=p_title_pair_plot))
        return f1, f2

    def plot_fit_timeseries(self, target_data, samples, ests, stats=None, probabilistic_model=None,
                            target_data_str="fit_target_data", state_variables_str=["x1", "x2"], dWt_str=["dWt"],
                            scalar_params_str=["sigma"], special_idx=[], skip_samples=0, trajectories_plot=False,
                            region_labels=[], title_prefix=""):
        if len(title_prefix) > 0:
            title_prefix = title_prefix + ": "
        samples = ensure_list(samples)
        if len(state_variables_str) > 0:
            if len(region_labels) == 0:
                region_labels = samples[0][state_variables_str[0]].space_labels
            n_regions = samples[0][state_variables_str[0]].number_of_labels
        else:
            n_regions = 0
        stats_region_labels = region_labels
        if len(stats_region_labels) != n_regions:
            stats_region_labels = n_regions * [""]
        stats_region_labels_x = stats_region_labels
        stats_region_labels_dWt = stats_region_labels
        stats_region_titles = stats_region_labels
        stats_target_data_labels = target_data.space_labels
        n_target_data = target_data.number_of_labels
        if len(stats_target_data_labels) != n_target_data:
            stats_target_data_labels = n_target_data * [""]
        dWt_star = [dWt+"_star" for dWt in dWt_str]
        ts_strings = [target_data_str] + state_variables_str + dWt_star
        scalar_str = []
        for p_str in scalar_params_str:
            if probabilistic_model is not None:
                scalar_str.append(p_str + " prior = " + str(probabilistic_model.get_prior(p_str)[0]))
        scalar_str = ", ".join(scalar_str)
        if stats is not None:
            stats_string = dict(zip(ts_strings, len(ts_strings)*["\n"]))
            if isinstance(stats, dict):
                for skey, sval in stats.items():
                    x_p_str_means = {}
                    dWt_p_str_means = {}
                    targ_p_str_means = {}
                    for p_str in ts_strings:
                        try:
                            stats_string[p_str] \
                                = stats_string[p_str] + skey + "_mean=" + str(numpy.mean(sval[p_str])) + ", "
                            if p_str in state_variables_str:
                                x_p_str_means[p_str] = [skey + "_" + p_str + "_mean=" + str(sval[p_str][:, ip].mean())
                                                      for ip in range(n_regions)]
                            elif p_str in dWt_star:
                                dWt_p_str_means[p_str] = [(skey + "_" + p_str + "_mean=" +
                                                       str(sval[p_str][:, ip].mean())).replace("_star", "")
                                                      for ip in range(n_regions)]
                            else:
                                targ_p_str_means[p_str] = [skey + "_" + p_str + "_mean=" + str(sval[p_str][:, ip].mean())
                                                      for ip in range(n_target_data)]
                        except:
                            pass
                    if len(targ_p_str_means) > 0:
                        stats_target_data_labels = [", ".join([stats_target_data_labels[ip],
                                                               targ_p_str_means[target_data_str][ip]])
                                                        for ip in range(n_target_data)]
                    stats_region_labels_x = [", ".join([stats_region_labels[ip],
                                                        ", ".join([x_p_str_means[p_str][ip]
                                                                   for p_str in x_p_str_means.keys()])])
                                           for ip in range(n_regions)]
                    stats_region_labels_dWt = [", ".join([stats_region_labels[ip],
                                                          ", ".join([dWt_p_str_means[p_str][ip]
                                                                     for p_str in dWt_p_str_means.keys()])])
                                           for ip in range(n_regions)]
                if len(state_variables_str) > 0:
                    stats_region_titles = [label.replace(", " + skey + "_" + state_variables_str[0],
                                                         "\n" + skey + "_" + state_variables_str[0])
                                           for skey in stats.keys() for label in stats_region_labels_x]
                for p_str in ts_strings:
                    stats_string[p_str] = stats_string[p_str][:-2]
        else:
            stats_string = dict(zip(ts_strings, 3*[""]))

        observation_dict = OrderedDict({'observation time series': target_data.squeezed})
        time = target_data.time_line
        figs = []
        # x1_pair_plot_samples = []
        for id_est, (est, sample) in enumerate(zip(ensure_list(ests), samples)):
            name = title_prefix + "_chain" + str(id_est + 1)

            x = OrderedDict()
            for x_str in state_variables_str:
                try:
                    this_x = {x_str: sample[x_str].data[:, :, :, skip_samples:].squeeze()}
                    x.update(this_x)
                    subtitles = ['hidden state ' + x_str + stats_string[x_str]]
                    figs.append(self.plot_raster(this_x, time, special_idx=special_idx, time_units=target_data.time_unit,
                                                 title=name + ": Hidden states fit rasterplot " + x_str,
                                                 subtitles=subtitles, offset=0.25, labels=stats_region_labels_x,
                                                 figsize=FiguresConfig.VERY_LARGE_SIZE))
                except:
                    pass

                if trajectories_plot and (len(x) == 2 or len(x) == 3):
                    title = name + ' Fit hidden state space trajectories'
                    figs.append(self.plot_trajectories(x, special_idx=special_idx, title=title,
                                                       labels=stats_region_titles, figsize=FiguresConfig.SUPER_LARGE_SIZE))

            dWt = OrderedDict()
            subtitles = []
            for d_str in dWt_str:
                try:
                    dWt[d_str] = sample.get(d_str+"_star", sample.get(d_str)).data[:, :, :, skip_samples:].squeeze()
                    subtitles.append(d_str + stats_string[x_str])
                except:
                    pass
            if len(dWt) > 0:
                subtitles[-1] = ", ".join([subtitles[-1], "dynamic noise "])
                for p_str in scalar_params_str:
                    p_est = est.get("sigma", None)
                    if p_est is not None:
                        scalar_str += ", " + p_str + " post = " + str(p_est)
                if len(scalar_str) > 0:
                    subtitles[-1] += scalar_str
                figs.append(self.plot_raster(dWt, time[:-1], time_units=target_data.time_unit,
                                             special_idx=special_idx,
                                             title=name + ": Hidden states random walk rasterplot",
                                             subtitles=subtitles, offset=0.25, labels=stats_region_labels_dWt,
                                             figsize=FiguresConfig.VERY_LARGE_SIZE))
            try:
                observation_dict.update({"fit chain " + str(id_est + 1):
                                         sample[target_data_str].data[:, :, :, skip_samples:].squeeze()})
                chain_observation_dict = {observation_dict.keys()[0]: observation_dict.values()[0],
                                          observation_dict.keys()[-1]: observation_dict.values()[-1]}
                figs.append(
                    self.plot_raster(chain_observation_dict, time, special_idx=[], time_units=target_data.time_unit,
                                     title=name + "Observation target vs fit time series",
                                     figure_name=title_prefix + "ObservationTarget_VS_FitRasterPlot",
                                     offset=0.25, labels=stats_target_data_labels,
                                     figsize=FiguresConfig.VERY_LARGE_SIZE))
                figs.append(
                    self.plot_timeseries(chain_observation_dict, time, special_idx=[], time_units=target_data.time_unit,
                                         title=name + "Observation target vs fit time series", subplots=(2, 1),
                                         figure_name=title_prefix + "ObservationTarget_VS_FitTimeSeries",
                                         labels=stats_target_data_labels, figsize=FiguresConfig.VERY_LARGE_SIZE))
            except:
                pass

        if len(observation_dict) > 1:
            figs.append(self.plot_raster(observation_dict, time, special_idx=[], time_units=target_data.time_unit,
                                         title=title_prefix + "Observation target vs fit time series: "
                                                + stats_string[target_data_str],
                                         figure_name=title_prefix + "ObservationTarget_VS_FitRasterPlot",
                                         offset=0.25, labels=stats_target_data_labels,
                                         figsize=FiguresConfig.VERY_LARGE_SIZE))
            figs.append(self.plot_timeseries(observation_dict, time, special_idx=[], time_units=target_data.time_unit,
                                             title=title_prefix + "Observation target vs fit time series: "
                                                   + stats_string[target_data_str], subplots=(len(observation_dict), 1),
                                             figure_name=title_prefix + "ObservationTarget_VS_FitTimeSeries",
                                             labels=stats_target_data_labels, figsize=FiguresConfig.VERY_LARGE_SIZE))
        return tuple(figs)

    def plot_fit_connectivity(self, ests, samples, stats=None, probabilistic_model=None, model_conn_str="MC",
                              region_labels=[], title_prefix=""):
        # plot connectivity
        if len(title_prefix) > 0:
            title_prefix = title_prefix + "_"
        if probabilistic_model is not None:
            MC_prior = probabilistic_model.get_prior(model_conn_str)
            MC_subplot = 122
        else:
            MC_prior = False
            MC_subplot = 111
        for id_est, (est, sample) in enumerate(zip(ensure_list(ests), ensure_list(samples))):
            conn_figure_name = title_prefix + "chain" + str(id_est + 1) + ": Model Connectivity"
            pyplot.figure(conn_figure_name, FiguresConfig.VERY_LARGE_SIZE)
            # plot_regions2regions(conn.weights, conn.region_labels, 121, "weights")
            if MC_prior:
                self.plot_regions2regions(MC_prior, region_labels, 121,
                                          "Prior Model Connectivity")
            MC_title = "Posterior Model  Connectivity"
            if isinstance(stats, dict):
                MC_title = MC_title + ": "
                for skey, sval in stats.items():
                    MC_title = MC_title + skey + "_mean=" + str(sval[model_conn_str].mean()) + ", "
                MC_title = MC_title[:-2]
            fig=self.plot_regions2regions(est[model_conn_str], region_labels, MC_subplot, MC_title)
            self._save_figure(pyplot.gcf(), conn_figure_name)
            self._check_show()
            return fig

    def plot_scalar_model_comparison(self, model_comps, title_prefix="",
                                     metrics=["aic", "aicc", "bic", "dic", "waic", "p_waic", "elpd_waic", "loo"],
                                     subplot_shape=None, figsize=FiguresConfig.VERY_LARGE_SIZE, figure_name=None):
        metrics = [metric for metric in metrics if metric in model_comps.keys()]
        if subplot_shape is None:
            n_metrics = len(metrics)
            # subplot_shape = self.rect_subplot_shape(n_metrics, mode="col")
            if n_metrics > 1:
                subplot_shape = (int(numpy.ceil(1.0*n_metrics/2)), 2)
            else:
                subplot_shape = (1, 1)
        if len(title_prefix) > 0:
            title = title_prefix + ": " + "information criteria"
        else:
            title = "Information criteria"
        fig, axes = pyplot.subplots(subplot_shape[0], subplot_shape[1], figsize=figsize)
        fig.suptitle(title)
        fig.set_label(title)
        for imetric, metric in enumerate(metrics):
            if isinstance(model_comps[metric], dict):
                metric_data = model_comps[metric].values()
                group_names = model_comps[metric].keys()
            else:
                metric_data = model_comps[metric]
                group_names = [""]
            metric_data[numpy.abs(metric_data) == numpy.inf] = numpy.nan
            isb, jsb = numpy.unravel_index(imetric, subplot_shape)
            axes[isb, jsb] = self.plot_bars(metric_data, ax=axes[isb, jsb], fig=fig, title=metric,
                                            group_names=group_names, legend_prefix="chain/run ")[1]
        # fig.tight_layout()
        self._save_figure(fig, figure_name)
        self._check_show()
        return fig, axes

    # TODO: refactor to not have the plot commands here
    def plot_array_model_comparison(self, model_comps, title_prefix="", metrics=["loos", "ks"], labels=[],
                                    xdata=None, xlabel="", figsize=FiguresConfig.VERY_LARGE_SIZE, figure_name=None):

        def arrange_chains_or_runs(metric_data):
            n_chains_or_runs = 1
            for imodel, model in enumerate(metric_data):
                if model.ndim > 2:
                    if model.shape[0] > n_chains_or_runs:
                        n_chains_or_runs = model.shape[0]
                else:
                    metric_data[imodel] = numpy.expand_dims(model, axis=0)
            return metric_data

        colorcycle = pyplot.rcParams['axes.prop_cycle'].by_key()['color']
        n_colors = len(colorcycle)
        metrics = [metric for metric in metrics if metric in model_comps.keys()]
        figs=[]
        axs = []
        for metric in metrics:
            if isinstance(model_comps[metric], dict):
                # Multiple models as a list of np.arrays of chains x data
                metric_data = model_comps[metric].values()
                model_names = model_comps[metric].keys()
            else:
                # Single models as a one element list of one np.array of chains x data
                metric_data = [model_comps[metric]]
                model_names = [""]
            metric_data = arrange_chains_or_runs(metric_data)
            n_models = len(metric_data)
            for jj in range(n_models):
                # Necessary because ks gets infinite sometimes...
                temp = metric_data[jj] == numpy.inf
                if numpy.all(temp):
                    warning("All values are inf for metric " + metric + " of model " + model_names[ii] + "!\n")
                    return
                elif numpy.any(temp):
                    warning("Inf values found for metric " + metric + " of model " + model_names[ii] + "!\n" +
                            "Substituting them with the maximum non-infite value!")
                    metric_data[jj][temp] = metric_data[jj][~temp].max()
            n_subplots = metric_data[0].shape[1]
            n_labels = len(labels)
            if n_labels != n_subplots:
                if n_labels != 0:
                    warning("Ignoring labels because their number (" + str(n_labels) +
                            ") is not equal to the number of row subplots (" + str(n_subplots) + ")!")
                labels = [str(ii + 1) for ii in range(n_subplots)]
            if xdata is None:
                xdata = numpy.arange(metric_data[jj].shape[-1])
            else:
                xdata = xdata.flatten()
            xdata0 = numpy.concatenate([numpy.reshape(xdata[0] - 0.1*(xdata[-1]-xdata[0]), (1,)), xdata])
            xdata1 = xdata[-1] + 0.1 * (xdata[-1] - xdata[0])
            if len(title_prefix) > 0:
                title = title_prefix + ": " + metric
            else:
                title = metric
            fig = pyplot.figure(title, figsize=figsize)
            fig.suptitle(title)
            fig.set_label(title)
            gs = gridspec.GridSpec(n_subplots, n_models)
            axes = numpy.empty((n_subplots, n_models), dtype="O")
            for ii in range(n_subplots-1,-1, -1):
                for jj in range(n_models):
                    if ii > n_subplots-1:
                        if jj > 0:
                            axes[ii, jj] = pyplot.subplot(gs[ii, jj], sharex=axes[n_subplots-1, jj], sharey=axes[ii, 0])
                        else:
                            axes[ii, jj] = pyplot.subplot(gs[ii, jj], sharex=axes[n_subplots-1, jj])
                    else:
                        if jj > 0:
                            axes[ii, jj] = pyplot.subplot(gs[ii, jj], sharey=axes[ii, 0])
                        else:
                            axes[ii, jj] = pyplot.subplot(gs[ii, jj])
                    n_chains_or_runs = metric_data[jj].shape[0]
                    for kk in range(n_chains_or_runs):
                        c = colorcycle[kk % n_colors]
                        axes[ii, jj].plot(xdata, metric_data[jj][kk][ii, :],  label="chain/run " + str(kk + 1),
                                          marker="o", markersize=5, markeredgecolor=c, markerfacecolor=c,
                                          linestyle="-", linewdith=1)
                        m = numpy.nanmean(metric_data[jj][kk][ii, :])
                        axes[ii, jj].plot(xdata0, m * numpy.ones(xdata0.shape), color=c, linewidth=1)
                        axes[ii, jj].text(xdata0[0], 1.1 * m, 'mean=%0.2f' % m, ha='center', va='bottom', color=c)
                    axes[ii, jj].set_xlabel(xlabel)
                    if ii == 0:
                        axes[ii, jj].set_title(model_names[ii])
                    if n_chains_or_runs > 1 and ii == n_subplots-1:
                        axes[ii, jj].legend()
                if ii == n_subplots-1:
                    axes[ii, 0].autoscale()  # tight=True
                    axes[ii, 0].set_xlim([xdata0[0], xdata1])  # tight=True
            # fig.tight_layout()
            self._save_figure(fig, figure_name)
            self._check_show()
            figs.append(fig)
            axs.append(axes)
        return tuple(figs), tuple(axs)
