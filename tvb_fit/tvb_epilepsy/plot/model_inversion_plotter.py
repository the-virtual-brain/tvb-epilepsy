# from collections import OrderedDict

import numpy


from tvb_fit.tvb_epilepsy.base.constants.config import FiguresConfig
import matplotlib
matplotlib.use(FiguresConfig().MATPLOTLIB_BACKEND)
# from matplotlib import pyplot, gridspec

# from tvb_fit.base.utils.log_error_utils import warning
# from tvb_fit.samplers.stan.stan_interface import merge_samples

from tvb_fit.base.utils.data_structures_utils import ensure_list, isequal_string, \
    generate_region_labels   #, sort_dict,  extract_dict_stringkeys
from tvb_fit.plot.model_inversion_plotter import ModelInversionPlotter as ModelInversionPlotterBase


class ModelInversionPlotter(ModelInversionPlotterBase):

    def __init__(self, config=None):
        super(ModelInversionPlotter, self).__init__(config)

    # def _params_stats_subtitles(self, params, stats):
    #     subtitles = list(params)
    #     if isinstance(stats, dict):
    #         for ip, param in enumerate(params):
    #             subtitles[ip] = subtitles[ip] + ": "
    #             for skey, sval in stats.items():
    #                 subtitles[ip] = subtitles[ip] + skey + "=" + str(sval[param]) + ", "
    #             subtitles[ip] = subtitles[ip][:-2]
    #     return subtitles
    #
    # def _params_stats_labels(self, param, stats, labels):
    #     subtitles = list(labels)
    #     if isinstance(stats, dict):
    #         n_params = len(stats.values()[0][param])
    #         if len(subtitles) == 1 and n_params > 1:
    #             subtitles = subtitles * n_params
    #         elif len(subtitles) == 0:
    #             subtitles = [""] * n_params
    #         for ip in range(n_params):
    #             if len(subtitles[ip]) > 0:
    #                 subtitles[ip] = subtitles[ip] + ": "
    #             for skey, sval in stats.items():
    #                 subtitles[ip] = subtitles[ip] + skey + "=" + str(sval[param][ip]) + ", "
    #                 subtitles[ip] = subtitles[ip][:-2]
    #     return subtitles
    #
    # def _parameters_pair_plots(self, samples_all, params=["tau1", "K", "sigma", "epsilon", "scale", "offset"],
    #                            stats=None, priors={}, truth={}, skip_samples=0, title='Parameters samples',
    #                            figure_name=None, figsize=FiguresConfigVERY_LARGE_SIZE):
    #     subtitles = list(self._params_stats_subtitles(params, stats))
    #     samples = []
    #     samples_all = ensure_list(samples_all)
    #     params = [param for param in params if param in samples_all[0].keys()]
    #     for sample in samples_all:
    #         samples.append(extract_dict_stringkeys(sample, params, modefun="equal"))
    #     if len(samples) > 1:
    #         samples = merge_samples(samples)
    #     else:
    #         samples = samples[0]
    #         n_samples = (samples.values()[0]).shape[0]
    #         for p_key, p_val in samples.items():
    #             samples[p_key] = numpy.reshape(p_val, (1, n_samples))
    #     diagonal_plots = {}
    #     # for param_key in samples.keys():
    #     for p_key in params:
    #         diagonal_plots.update({p_key: [priors.get(p_key, ()), truth.get(p_key, ())]})
    #
    #     return self.pair_plots(samples, params, diagonal_plots, True, skip_samples,
    #                            title, "chain/run ", subtitles, figure_name, figsize)
    #
    # def _region_parameters_violin_plots(self, samples_all, values=None, lines=None, stats=None,
    #                                     params=["x0", "x1_init", "z_init"], skip_samples=0, per_chain_or_run=False,
    #                                     labels=[], special_idx=None, figure_name="Regions parameters samples",
    #                                     figsize=FiguresConfigVERY_LARGE_SIZE):
    #     if isinstance(values, dict):
    #         vals_fun = lambda param: values.get(param, numpy.array([]))
    #     else:
    #         vals_fun = lambda param: []
    #     if isinstance(lines, dict):
    #         lines_fun = lambda param: lines.get(param, numpy.array([]))
    #     else:
    #         lines_fun = lambda param: []
    #     samples_all = ensure_list(samples_all)
    #     params = [param for param in params if param in samples_all[0].keys()]
    #     samples = []
    #     for sample in samples_all:
    #         samples.append(extract_dict_stringkeys(sample, params, modefun="equal"))
    #     labels = generate_region_labels(samples[0].values()[0].shape[-1], labels, numbering=False)
    #     n_chains = len(samples_all)
    #     n_samples = samples[0].values()[0].shape[0]
    #     if n_samples > 1:
    #         violin_flag = True
    #     else:
    #         violin_flag = False
    #     if not per_chain_or_run and n_chains > 1:
    #         samples = [merge_samples(samples)]
    #         plot_samples = lambda s: numpy.concatenate(numpy.split(s[:, skip_samples:].T, n_chains, axis=2),
    #                                                    axis=1).squeeze().T
    #         plot_figure_name = lambda ichain: figure_name
    #     else:
    #         plot_samples = lambda s: s[skip_samples:]
    #         plot_figure_name = lambda ichain: figure_name + ": chain " + str(ichain + 1)
    #     params_labels = {}
    #     for ip, p in enumerate(params):
    #         if ip == 0:
    #             params_labels[p] = self._params_stats_labels(p, stats, labels)
    #         else:
    #             params_labels[p] = self._params_stats_labels(p, stats, "")
    #     n_params = len(params)
    #     if n_params > 9:
    #         warning("Number of subplots in column wise vector-violin-plots cannot be > 9 and it is "
    #                           + str(n_params) + "!")
    #     subplot_ind = 100 + n_params * 10
    #     figs = []
    #     for ichain, chain_sample in enumerate(samples):
    #         pyplot.figure(plot_figure_name(ichain), figsize=figsize)
    #         for ip, param in enumerate(params):
    #             fig = self.plot_vector_violin(plot_samples(chain_sample[param]), vals_fun(param),
    #                                           lines_fun(param), params_labels[param],
    #                                           subplot_ind + ip + 1, param, violin_flag=violin_flag,
    #                                           colormap="YlOrRd", show_y_labels=True,
    #                                           indices_red=special_idx, sharey=None)
    #         self._save_figure(pyplot.gcf(), None)
    #         self._check_show()
    #         figs.append(fig)
    #     return tuple(figs)
    #
    # def plot_fit_scalar_params_iters(self, samples_all, params=["tau1", "K", "sigma", "epsilon", "scale", "offset"],
    #                                  skip_samples=0, title_prefix="", subplot_shape=None, figure_name=None,
    #                                  figsize=FiguresConfigLARGE_SIZE):
    #     if len(title_prefix) > 0:
    #         title_prefix = title_prefix + ": "
    #     title = title_prefix + " Parameters samples per iteration"
    #     samples_all = ensure_list(samples_all)
    #     params = [param for param in params if param in samples_all[0].keys()]
    #     samples = []
    #     for sample in samples_all:
    #         samples.append(extract_dict_stringkeys(sample, params, modefun="equal"))
    #     if len(samples) > 1:
    #         samples = merge_samples(samples)
    #     else:
    #         samples = samples[0]
    #     if subplot_shape is None:
    #         n_params = len(samples)
    #         # subplot_shape = self.rect_subplot_shape(n_params, mode="col")
    #         if n_params > 1:
    #             subplot_shape = (int(numpy.ceil(1.0*n_params/2)), 2)
    #         else:
    #             subplot_shape = (1, 1)
    #     n_chains_or_runs = samples.values()[0].shape[0]
    #     legend = {samples.keys()[0]: ["chain/run " + str(ii+1) for ii in range(n_chains_or_runs)]}
    #     return self.plots(samples, shape=subplot_shape, transpose=True, skip=skip_samples, xlabels={}, xscales={},
    #                       yscales={}, title=title, lgnd=legend, figure_name=figure_name, figsize=figsize)
    #
    # def plot_fit_scalar_params(self, samples, stats, probabilistic_model=None,
    #                            params=["tau1", "K", "sigma", "epsilon", "scale", "offset"], skip_samples=0,
    #                            title_prefix=""):
    #     # plot scalar parameters in pair plots
    #     if len(title_prefix) > 0:
    #         title_prefix = title_prefix + ": "
    #     title = title_prefix + " Parameters samples"
    #     priors = {}
    #     truth = {}
    #     if probabilistic_model is not None:
    #         for p in params:
    #             pdf = probabilistic_model.get_prior_pdf(p)
    #             # TODO: a better hack than the following for when pdf returns p_mean, nan and p_mean is not a scalar
    #             if pdf[1] is numpy.nan:
    #                 pdf = list(pdf)
    #                 pdf[0] = numpy.mean(pdf[0])
    #                 pdf = tuple(pdf)
    #             priors.update({p: pdf})
    #             truth.update({p: numpy.nanmean(probabilistic_model.get_truth(p))})
    #     return self._parameters_pair_plots(samples, params, stats, priors, truth, skip_samples, title=title)

    def plot_fit_region_params(self, samples, stats=None, probabilistic_model=None,
                               params=["x0", "x1eq", "zeq"], special_idx=[], region_labels=[],
                               regions_mode="all", per_chain_or_run=False, skip_samples=0, title_prefix=""):
        if len(title_prefix) > 0:
            title_prefix = title_prefix + " "
        title_pair_plot = title_prefix + "Global coupling vs x0 pair plot"
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
        if not(per_chain_or_run) and "x0" in params and samples[0]["x0"].shape[1] < 10:
            x0_K_pair_plot_params = []
            x0_K_pair_plot_samples = [{} for _ in range(len(samples))]
            if samples[0].get("K", None) is not None:
                # plot K-x0 parameters in pair plots
                x0_K_pair_plot_params = ["K"]
                x0_K_pair_plot_samples = [{"K": s["K"]} for s in samples]
                if probabilistic_model is not None:
                    pdf = probabilistic_model.get_prior_pdf("K")
                    # TODO: a better hack than the following for when pdf returns p_mean, nan and p_mean is not a scalar
                    if pdf[1] is numpy.nan:
                        pdf = list(pdf)
                        pdf[0] = numpy.mean(pdf[0])
                        pdf = tuple(pdf)
                    priors.update({"K": pdf})
                    truth.update({"K": probabilistic_model.get_truth("K")})
            for inode, label in enumerate(region_labels):
                temp_name = "x0[" + label + "]"
                x0_K_pair_plot_params.append(temp_name)
                for ichain, s in enumerate(samples):
                    x0_K_pair_plot_samples[ichain].update({temp_name: s["x0"][:, inode]})
                    if probabilistic_model is not None:
                        priors.update({temp_name: (priors["x0"][0][inode], priors["x0"][1][inode])})
                        truth.update({temp_name: truth["x0"][inode]})
            f2 = self._parameters_pair_plots(x0_K_pair_plot_samples, x0_K_pair_plot_params, None, priors, truth,
                                             skip_samples, title=title_pair_plot)
            return f1, f2
        else:
            return f1

    # def plot_fit_timeseries(self, target_data, samples, ests, stats=None, probabilistic_model=None,
    #                         special_idx=[], skip_samples=0, trajectories_plot=False, title_prefix=""):
    #     if len(title_prefix) > 0:
    #         title_prefix = title_prefix + ": "
    #     samples = ensure_list(samples)
    #     region_labels = samples[0]["x1"].space_labels
    #     if probabilistic_model is not None:
    #         sig_prior_str = " sig_prior = " + str(probabilistic_model.get_prior("sigma")[0])
    #     else:
    #         sig_prior_str = ""
    #     stats_region_labels = region_labels
    #     n_labels = len(region_labels)
    #     if stats is not None:
    #         stats_string = {"fit_target_data": "\n", "x1": "\n", "z": "\n", "MC": ""}
    #         if isinstance(stats, dict):
    #             for skey, sval in stats.items():
    #                 p_str_means = {}
    #                 for p_str in ["fit_target_data", "x1", "z"]:
    #                     try:
    #                         stats_string[p_str] \
    #                             = stats_string[p_str] + skey + "_mean=" + str(numpy.mean(sval[p_str])) + ", "
    #                         p_str_means[p_str] = [", " + skey + "_" + p_str + "_mean=" + str(sval[p_str][:, ip].mean())
    #                                               for ip in range(n_labels)]
    #                     except:
    #                         p_str_means[p_str] = ["" for ip in range(n_labels)]
    #                         pass
    #                 stats_region_labels = [stats_region_labels[ip] +
    #                                        p_str_means["x1"][ip] + p_str_means["z"][ip]
    #                                         for ip in range(n_labels)]
    #             for p_str in ["fit_target_data", "x1", "z"]:
    #                 stats_string[p_str] = stats_string[p_str][:-2]
    #     else:
    #         stats_string = dict(zip(["fit_target_data", "x1", "z"], 3*[""]))
    #
    #     observation_dict = OrderedDict({'observation time series': target_data.squeezed})
    #     time = target_data.time_line
    #     figs = []
    #     # x1_pair_plot_samples = []
    #     for id_est, (est, sample) in enumerate(zip(ensure_list(ests), samples)):
    #         name = title_prefix + "_chain" + str(id_est + 1)
    #         try:
    #             observation_dict.update({"fit chain " + str(id_est + 1):
    #                                      sample["fit_target_data"].data[:, :, :, skip_samples:].squeeze()})
    #         except:
    #             pass
    #
    #         x = OrderedDict()
    #         subtitles = ""
    #         for x_str in ["x1", "z"]:
    #             try:
    #                 x[x_str] = sample[x_str].data[:, :, :, skip_samples:].squeeze()
    #                 subtitles += 'hidden state ' + x_str + stats_string[x_str]
    #             except:
    #                 pass
    #         if len(x) > 0:
    #             figs.append(self.plot_raster(x, time, special_idx=special_idx, time_units=target_data.time_unit,
    #                                          title=name + ": Hidden states fit rasterplot", subtitles=subtitles, offset=1.0,
    #                                          labels=region_labels, figsize=FiguresConfigVERY_LARGE_SIZE))
    #             if trajectories_plot:
    #                 title = name + ' Fit hidden state space trajectories'
    #                 figs.append(self.plot_trajectories(x, special_idx=special_idx, title=title,
    #                                                    labels=stats_region_labels, figsize=FiguresConfigSUPER_LARGE_SIZE))
    #             # TODO: add time series probability distribution plots per region
    #             # x1 = x1.T
    #             # z = z.T
    #             # x1_pair_plot_samples.append({})
    #             # figs.append(self._parameters_pair_plots(x1_pair_plot_samples, region_labels,
    #             #                                         None, None, None, skip_samples,
    #             #                                         title=name + ": x1 pair plot per region"))
    #
    #         dWt = OrderedDict()
    #         subtitles = []
    #         for d_str in ["dX1t", "dZt", "dWt"]:
    #             try:
    #                 dWt[d_str] = sample.get(d_str+"_star", sample.get(d_str)).data[:, :, :, skip_samples:].squeeze()
    #                 subtitles.append(d_str)
    #             except:
    #                 pass
    #         if len(dWt) > 0:
    #             subtitles[-1] += "\ndynamic noise" + sig_prior_str + ", sig_post = " + str(est["sigma"])
    #             figs.append(self.plot_raster(dWt, time[:-1], time_units=target_data.time_unit,
    #                                          special_idx=special_idx,
    #                                          title=name + ": Hidden states random walk rasterplot",
    #                                          subtitles=subtitles, offset=1.0, labels=region_labels,
    #                                          figsize=FiguresConfigVERY_LARGE_SIZE))
    #     if len(observation_dict) > 1:
    #         figs.append(self.plot_raster(observation_dict, time, special_idx=[], time_units=target_data.time_unit,
    #                                      title=title_prefix + "Observation target vs fit time series: "
    #                                             + stats_string["fit_target_data"],
    #                                      figure_name=title_prefix + "ObservationTarget_VS_FitRasterPlot",
    #                                      offset=1.0, labels=target_data.space_labels,
    #                                      figsize=FiguresConfigVERY_LARGE_SIZE))
    #         figs.append(self.plot_timeseries(observation_dict, time, special_idx=[], time_units=target_data.time_unit,
    #                                          title=title_prefix + "Observation target vs fit time series: "
    #                                                + stats_string["fit_target_data"],
    #                                          figure_name=title_prefix + "ObservationTarget_VS_FitTimeSeries",
    #                                          labels=target_data.space_labels, figsize=FiguresConfigVERY_LARGE_SIZE))
    #     return tuple(figs)
    #
    # def plot_fit_connectivity(self, ests, samples, stats=None, probabilistic_model=None,
    #                           region_labels=[], title_prefix=""):
    #     # plot connectivity
    #     if len(title_prefix) > 0:
    #         title_prefix = title_prefix + "_"
    #     if probabilistic_model is not None:
    #         MC_prior = probabilistic_model.get_prior("MC")
    #         MC_subplot = 122
    #     else:
    #         MC_prior = False
    #         MC_subplot = 111
    #     for id_est, (est, sample) in enumerate(zip(ensure_list(ests), ensure_list(samples))):
    #         conn_figure_name = title_prefix + "chain" + str(id_est + 1) + ": Model Connectivity"
    #         pyplot.figure(conn_figure_name, FiguresConfigVERY_LARGE_SIZE)
    #         # plot_regions2regions(conn.weights, conn.region_labels, 121, "weights")
    #         if MC_prior:
    #             self.plot_regions2regions(MC_prior, region_labels, 121,
    #                                       "Prior Model Connectivity")
    #         MC_title = "Posterior Model  Connectivity"
    #         if isinstance(stats, dict):
    #             MC_title = MC_title + ": "
    #             for skey, sval in stats.items():
    #                 MC_title = MC_title + skey + "_mean=" + str(sval["MC"].mean()) + ", "
    #             MC_title = MC_title[:-2]
    #         fig=self.plot_regions2regions(est["MC"], region_labels, MC_subplot, MC_title)
    #         self._save_figure(pyplot.gcf(), conn_figure_name)
    #         self._check_show()
    #         return fig
    #
    # def plot_scalar_model_comparison(self, model_comps, title_prefix="",
    #                                  metrics=["aic", "aicc", "bic", "dic", "waic", "p_waic", "elpd_waic", "loo"],
    #                                  subplot_shape=None, figsize=FiguresConfigVERY_LARGE_SIZE, figure_name=None):
    #     metrics = [metric for metric in metrics if metric in model_comps.keys()]
    #     if subplot_shape is None:
    #         n_metrics = len(metrics)
    #         # subplot_shape = self.rect_subplot_shape(n_metrics, mode="col")
    #         if n_metrics > 1:
    #             subplot_shape = (int(numpy.ceil(1.0*n_metrics/2)), 2)
    #         else:
    #             subplot_shape = (1, 1)
    #     if len(title_prefix) > 0:
    #         title = title_prefix + ": " + "information criteria"
    #     else:
    #         title = "Information criteria"
    #     fig, axes = pyplot.subplots(subplot_shape[0], subplot_shape[1], figsize=figsize)
    #     fig.suptitle(title)
    #     fig.set_label(title)
    #     for imetric, metric in enumerate(metrics):
    #         if isinstance(model_comps[metric], dict):
    #             metric_data = model_comps[metric].values()
    #             group_names = model_comps[metric].keys()
    #         else:
    #             metric_data = model_comps[metric]
    #             group_names = [""]
    #         metric_data[numpy.abs(metric_data) == numpy.inf] = numpy.nan
    #         isb, jsb = numpy.unravel_index(imetric, subplot_shape)
    #         axes[isb, jsb] = self.plot_bars(metric_data, ax=axes[isb, jsb], fig=fig, title=metric,
    #                                         group_names=group_names, legend_prefix="chain/run ")[1]
    #     # fig.tight_layout()
    #     self._save_figure(fig, figure_name)
    #     self._check_show()
    #     return fig, axes
    #
    # # TODO: refactor to not have the plot commands here
    # def plot_array_model_comparison(self, model_comps, title_prefix="", metrics=["loos", "ks"], labels=[],
    #                                 xdata=None, xlabel="", figsize=FiguresConfigVERY_LARGE_SIZE, figure_name=None):
    #
    #     def arrange_chains_or_runs(metric_data):
    #         n_chains_or_runs = 1
    #         for imodel, model in enumerate(metric_data):
    #             if model.ndim > 2:
    #                 if model.shape[0] > n_chains_or_runs:
    #                     n_chains_or_runs = model.shape[0]
    #             else:
    #                 metric_data[imodel] = numpy.expand_dims(model, axis=0)
    #         return metric_data
    #
    #     colorcycle = pyplot.rcParams['axes.prop_cycle'].by_key()['color']
    #     n_colors = len(colorcycle)
    #     metrics = [metric for metric in metrics if metric in model_comps.keys()]
    #     figs=[]
    #     axs = []
    #     for metric in metrics:
    #         if isinstance(model_comps[metric], dict):
    #             # Multiple models as a list of np.arrays of chains x data
    #             metric_data = model_comps[metric].values()
    #             model_names = model_comps[metric].keys()
    #         else:
    #             # Single models as a one element list of one np.array of chains x data
    #             metric_data = [model_comps[metric]]
    #             model_names = [""]
    #         metric_data = arrange_chains_or_runs(metric_data)
    #         n_models = len(metric_data)
    #         for jj in range(n_models):
    #             # Necessary because ks gets infinite sometimes...
    #             temp = metric_data[jj] == numpy.inf
    #             if numpy.all(temp):
    #                 warning("All values are inf for metric " + metric + " of model " + model_names[ii] + "!\n")
    #                 return
    #             elif numpy.any(temp):
    #                 warning("Inf values found for metric " + metric + " of model " + model_names[ii] + "!\n" +
    #                         "Substituting them with the maximum non-infite value!")
    #                 metric_data[jj][temp] = metric_data[jj][~temp].max()
    #         n_subplots = metric_data[0].shape[1]
    #         n_labels = len(labels)
    #         if n_labels != n_subplots:
    #             if n_labels != 0:
    #                 warning("Ignoring labels because their number (" + str(n_labels) +
    #                         ") is not equal to the number of row subplots (" + str(n_subplots) + ")!")
    #             labels = [str(ii + 1) for ii in range(n_subplots)]
    #         if xdata is None:
    #             xdata = numpy.arange(metric_data[jj].shape[-1])
    #         else:
    #             xdata = xdata.flatten()
    #         xdata0 = numpy.concatenate([numpy.reshape(xdata[0] - 0.1*(xdata[-1]-xdata[0]), (1,)), xdata])
    #         xdata1 = xdata[-1] + 0.1 * (xdata[-1] - xdata[0])
    #         if len(title_prefix) > 0:
    #             title = title_prefix + ": " + metric
    #         else:
    #             title = metric
    #         fig = pyplot.figure(title, figsize=figsize)
    #         fig.suptitle(title)
    #         fig.set_label(title)
    #         gs = gridspec.GridSpec(n_subplots, n_models)
    #         axes = numpy.empty((n_subplots, n_models), dtype="O")
    #         for ii in range(n_subplots-1,-1, -1):
    #             for jj in range(n_models):
    #                 if ii > n_subplots-1:
    #                     if jj > 0:
    #                         axes[ii, jj] = pyplot.subplot(gs[ii, jj], sharex=axes[n_subplots-1, jj], sharey=axes[ii, 0])
    #                     else:
    #                         axes[ii, jj] = pyplot.subplot(gs[ii, jj], sharex=axes[n_subplots-1, jj])
    #                 else:
    #                     if jj > 0:
    #                         axes[ii, jj] = pyplot.subplot(gs[ii, jj], sharey=axes[ii, 0])
    #                     else:
    #                         axes[ii, jj] = pyplot.subplot(gs[ii, jj])
    #                 n_chains_or_runs = metric_data[jj].shape[0]
    #                 for kk in range(n_chains_or_runs):
    #                     c = colorcycle[kk % n_colors]
    #                     axes[ii, jj].plot(xdata, metric_data[jj][kk][ii, :],  label="chain/run " + str(kk + 1),
    #                                       marker="o", markersize=1, markeredgecolor=c, markerfacecolor=None,
    #                                       linestyle="None")
    #                     if n_chains_or_runs > 1:
    #                         axes[ii, jj].legend()
    #                     m = numpy.nanmean(metric_data[jj][kk][ii, :])
    #                     axes[ii, jj].plot(xdata0, m * numpy.ones(xdata0.shape), color=c, linewidth=1)
    #                     axes[ii, jj].text(xdata0[0], 1.1 * m, 'mean=%0.2f' % m, ha='center', va='bottom', color=c)
    #                 axes[ii, jj].set_xlabel(xlabel)
    #                 if ii == 0:
    #                     axes[ii, jj].set_title(model_names[ii])
    #             if ii == n_subplots-1:
    #                 axes[ii, 0].autoscale()  # tight=True
    #                 axes[ii, 0].set_xlim([xdata0[0], xdata1])  # tight=True
    #         # fig.tight_layout()
    #         self._save_figure(fig, figure_name)
    #         self._check_show()
    #         figs.append(fig)
    #         axs.append(axes)
    #     return tuple(figs), tuple(axs)

    def plot_fit_results(self, ests, samples, model_data, target_data, probabilistic_model=None, info_crit=None,
                         stats=None, pair_plot_params=["tau1", "sigma", "epsilon", "scale", "offset"],
                         region_violin_params=["x0", "x1eq", "zeq"],
                         region_labels=[], regions_mode="active", seizure_indices=[],
                         trajectories_plot=True, connectivity_plot=False, skip_samples=0, title_prefix=""):
        sigma = []
        if probabilistic_model is not None:
            n_regions = probabilistic_model.number_of_regions
            region_labels = generate_region_labels(n_regions, region_labels, ". ", True)
            if probabilistic_model.parameters.get("sigma", None) is not None:
                sigma = ["sigma"]
            active_regions = ensure_list(probabilistic_model.active_regions)
        else:
            active_regions = ensure_list(model_data.get("active_regions", []))

        if isequal_string(regions_mode, "all"):
            if len(seizure_indices) == 0:
                seizure_indices = active_regions
        else:
            if len(active_regions) > 0:
                seizure_indices = [active_regions.index(ind) for ind in seizure_indices]
                if len(region_labels) > 0:
                    region_labels = region_labels[active_regions]


        figs = []

        # Pack fit samples time series into timeseries objects:
        from tvb_fit.tvb_epilepsy.top.scripts.fitting_scripts import samples_to_timeseries
        samples, target_data, x1prior, x1eps = samples_to_timeseries(samples, model_data, target_data, region_labels)
        figs.append(self.plot_fit_timeseries(target_data, samples, ests, stats, probabilistic_model,
                                             "fit_target_data", ["x1", "z"], ["dWt"], sigma, seizure_indices,
                                             skip_samples, trajectories_plot, region_labels, title_prefix))

        figs.append(
            self.plot_fit_region_params(samples, stats, probabilistic_model, region_violin_params, seizure_indices,
                                        region_labels, regions_mode, False, skip_samples, title_prefix))

        figs.append(
            self.plot_fit_region_params(samples, stats, probabilistic_model, region_violin_params, seizure_indices,
                                        region_labels, regions_mode, True, skip_samples, title_prefix))

        figs.append(self.plot_fit_scalar_params(samples, stats, probabilistic_model, pair_plot_params,
                                                skip_samples, title_prefix))

        figs.append(self.plot_fit_scalar_params_iters(samples, pair_plot_params, 0, title_prefix, subplot_shape=None))


        if info_crit is not None:
            figs.append(self.plot_scalar_model_comparison(info_crit, title_prefix))
            figs.append(self.plot_array_model_comparison(info_crit, title_prefix, labels=target_data.space_labels,
                                                         xdata=target_data.time_line, xlabel="Time"))

        if connectivity_plot:
            figs.append(self.plot_fit_connectivity(ests, stats, probabilistic_model, "MC", region_labels, title_prefix))

        return tuple(figs)