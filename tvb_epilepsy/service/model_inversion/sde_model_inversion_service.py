import time
import numpy as np
from matplotlib import pyplot
from tvb_epilepsy.base.constants.configurations import FOLDER_FIGURES, FIG_FORMAT, LARGE_SIZE, VERY_LARGE_SIZE, \
    SAVE_FLAG, SHOW_FLAG
from tvb_epilepsy.base.constants.model_constants import model_noise_intensity_dict
from tvb_epilepsy.base.constants.model_inversion_constants import SIG_DEF
from tvb_epilepsy.base.utils.log_error_utils import initialize_logger
from tvb_epilepsy.base.utils.data_structures_utils import sort_dict  # isequal_string,
from tvb_epilepsy.base.utils.plot_utils import save_figure, check_show
from tvb_epilepsy.base.model.statistical_models.sde_statistical_model import SDEStatisticalModel
from tvb_epilepsy.plot.plotter import Plotter
from tvb_epilepsy.service.stochastic_parameter_factory import set_parameter_defaults
from tvb_epilepsy.service.epileptor_model_factory import AVAILABLE_DYNAMICAL_MODELS_NAMES, EPILEPTOR_MODEL_NVARS
from tvb_epilepsy.service.model_inversion.ode_model_inversion_service import ODEModelInversionService

LOG = initialize_logger(__name__)


class SDEModelInversionService(ODEModelInversionService):

    def __init__(self, model_configuration, hypothesis=None, head=None, dynamical_model=None, model_name=None,
                 logger=LOG, **kwargs):
        super(SDEModelInversionService, self).__init__(model_configuration, hypothesis, head, dynamical_model,
                                                       model_name, logger, **kwargs)
        self.set_default_parameters(**kwargs)

    def get_default_sig(self, **kwargs):
        if kwargs.get("sig", None):
            return kwargs.pop("sig")
        elif np.in1d(self.dynamical_model, AVAILABLE_DYNAMICAL_MODELS_NAMES):
                if EPILEPTOR_MODEL_NVARS[self.dynamical_model] == 2:
                    return model_noise_intensity_dict[self.dynamical_model][1]
                elif EPILEPTOR_MODEL_NVARS[self.dynamical_model] > 2:
                    return model_noise_intensity_dict[self.dynamical_model][2]
        else:
            return SIG_DEF

    def set_default_parameters(self, **kwargs):
        sig = self.get_default_sig(**kwargs)
        # Generative model:
        # Integration:
        # self.default_parameters.update(set_parameter_defaults("x1_dWt", "normal", (),  # name, pdf, shape
        #                                                       -10.0*sig, 10.0*sig,     # min, max
        #                                                       pdf_params={"mu": 0.0, "sigma": sig}))
        self.default_parameters.update(set_parameter_defaults("z_dWt", "normal", (),  # name, pdf, shape
                                                              pdf_params={"mu": 0.0, "sigma": sig}))
        sig_std = sig / kwargs.get("sig_scale_ratio", 3)
        self.default_parameters.update(set_parameter_defaults("sig", "gamma", (),  # name, pdf, shape
                                                              0.0, 10.0*sig,  # min, max
                                                              sig, sig_std,
                                                              pdf_params={"mean": sig/sig_std, "skew": -1.0},
                                                              **kwargs))

    def generate_statistical_model(self, model_name="vep_sde", **kwargs):
        tic = time.time()
        self.logger.info("Generating model...")
        active_regions = kwargs.pop("active_regions", [])
        self.default_parameters.update(kwargs)
        model = SDEStatisticalModel(model_name, self.n_regions, active_regions, self.n_signals, self.n_times, self.dt,
                                    self.get_default_sig_eq(**kwargs), self.get_default_sig_init(**kwargs),
                                    **self.default_parameters)
        self.model_generation_time = time.time() - tic
        self.logger.info(str(self.model_generation_time) + ' sec required for model generation')
        return model

    def generate_model_data(self, statistical_model, signals, gain_matrix=None):
        return super(SDEModelInversionService, self).generate_model_data(statistical_model, signals, gain_matrix)

    def plot_fit_results(self, est, statistical_model, signals, time=None, seizure_indices=None, trajectories_plot=False,
                        save_flag=SAVE_FLAG, show_flag=SHOW_FLAG, figure_dir=FOLDER_FIGURES, figure_format=FIG_FORMAT,
                        x1_str="x1", x0_str="x0", mc_str="MC",
                         signals_str="fit_signals", sig_str="sig", eps_str="eps",
                        **kwargs):
        name = statistical_model.name + kwargs.get("_id_est", "")
        if time is None:
            time = np.array(range(signals.shape[0]))
        time = time.flatten()
        sig_prior = statistical_model.parameters["sig"].mean / statistical_model.sig
        eps_prior = statistical_model.parameters["eps"].mean
        plotter = Plotter()
        plotter.plot_raster(time, sort_dict({'observation signals': signals,
                                     'observation signals fit': est[signals_str]}),
                    special_idx=seizure_indices, time_units=est.get('time_units', "ms"),
                    title=name + ": Observation signals vs fit rasterplot",
                    subtitles=['observation signals ' +
                               '\nobservation noise eps_prior =  ' + str(eps_prior) + " eps_post =" + str(est[eps_str]),
                               'observation signals fit'], offset=0.1,
                    labels=None, save_flag=save_flag, show_flag=show_flag, figure_dir=figure_dir,
                    figure_format=figure_format, figsize=VERY_LARGE_SIZE)
        plotter.plot_raster(time, sort_dict({'x1': est[x1_str], 'z': est["z"]}),
                    special_idx=seizure_indices, time_units=est.get('time_units', "ms"),
                    title=name + ": Hidden states fit rasterplot",
                    subtitles=['hidden state x1' '\ndynamic noise sig_prior = ' + str(sig_prior) +
                               " sig_post = " + str(est[sig_str]),
                               'hidden state z'], offset=3.0,
                    labels=None, save_flag=save_flag, show_flag=show_flag, figure_dir=figure_dir,
                    figure_format=figure_format, figsize=VERY_LARGE_SIZE)
        if trajectories_plot:
            title = name + ': Fit hidden state space trajectories'
            title += "\n prior x0: " + str(self.x0_values[statistical_model.active_regions])
            x0 = est[x0_str]
            if len(x0) > statistical_model.n_active_regions:
                x0 = x0[statistical_model.active_regions]
            title += "\n x0 fit: " + str(x0)
            plotter.plot_trajectories({'x1': est[x1_str], 'z(t)': est['z']}, special_idx=seizure_indices,
                              title=title, labels=self.region_labels, show_flag=show_flag, save_flag=save_flag,
                              figure_dir=FOLDER_FIGURES, figure_format=FIG_FORMAT, figsize=LARGE_SIZE)
        # plot connectivity
        conn_figure_name = name + "Model Connectivity"
        pyplot.figure(conn_figure_name, VERY_LARGE_SIZE)
        # plot_regions2regions(conn.weights, conn.region_labels, 121, "weights")
        MC_prior = statistical_model.parameters["MC"].mean
        K_prior = statistical_model.parameters["K"].mean
        plotter.plot_regions2regions(MC_prior, self.region_labels[statistical_model.active_regions], 121,
                                     "Prior Model Connectivity" + "\nglobal scaling prior: K = " + str(K_prior))
        plotter.plot_regions2regions(est[mc_str], self.region_labels[statistical_model.active_regions], 122,
                                     "Posterior Model  Connectivity" + "\nglobal scaling fit: K = " + str(est["K"]))
        save_figure(save_flag, pyplot.gcf(), conn_figure_name, figure_dir, figure_format)
        check_show(show_flag=show_flag)
