
import numpy

from tvb_fit.tvb_epilepsy.base.constants.config import FiguresConfig
import matplotlib
matplotlib.use(FiguresConfig().MATPLOTLIB_BACKEND)

from tvb_fit.base.utils.data_structures_utils import ensure_list, isequal_string, generate_region_labels
from tvb_fit.plot.model_inversion_plotter import ModelInversionPlotter as ModelInversionPlotterBase


class ModelInversionPlotter(ModelInversionPlotterBase):

    def __init__(self, config=None):
        super(ModelInversionPlotter, self).__init__(config)

    def plot_fit_region_params(self, samples, stats=None, probabilistic_model=None,
                               params=["x0", "PZ", "x1eq", "zeq"], special_idx=[], region_labels=[],
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


    def plot_fit_results(self, ests, samples, model_data, target_data, probabilistic_model=None, info_crit=None,
                         stats=None, pair_plot_params=["tau1", "sigma", "epsilon", "scale", "offset"],
                         region_violin_params=["x0", "PZ", "x1eq", "zeq"],
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

        if len(region_labels) == 0:
            self.print_regions_indices = True
            self.print_ts_indices = True

        figs = []

        # Pack fit samples time series into timeseries objects:
        from tvb_fit.tvb_epilepsy.top.scripts.fitting_scripts import samples_to_timeseries
        samples, target_data, x1prior, x1eps = samples_to_timeseries(samples, model_data, target_data, region_labels)
        figs.append(self.plot_fit_timeseries(target_data, samples, ests, stats, probabilistic_model, "fit_target_data",
                                             ["x1", "z"], ["dWt", "dX1t", "dZt"], sigma, seizure_indices,
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
