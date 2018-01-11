import time
import numpy as np
from matplotlib import pyplot
from tvb_epilepsy.base.constants.configurations import FOLDER_FIGURES, FIG_FORMAT, LARGE_SIZE, VERY_LARGE_SIZE, \
    SAVE_FLAG, SHOW_FLAG
from tvb_epilepsy.base.constants.model_constants import model_noise_intensity_dict
from tvb_epilepsy.base.utils.log_error_utils import initialize_logger
from tvb_epilepsy.base.utils.data_structures_utils import isequal_string, sort_dict
from tvb_epilepsy.base.utils.plot_utils import plot_raster, plot_regions2regions, plot_trajectories, save_figure, \
    check_show
from tvb_epilepsy.base.model.statistical_models.sde_statistical_model import SDEStatisticalModel
from tvb_epilepsy.service.stochastic_parameter_factory import set_parameter_defaults
from tvb_epilepsy.service.epileptor_model_factory import AVAILABLE_DYNAMICAL_MODELS_NAMES, EPILEPTOR_MODEL_NVARS
from tvb_epilepsy.service.model_inversion.ode_model_inversion_service import ODEModelInversionService

LOG = initialize_logger(__name__)


class SDEModelInversionService(ODEModelInversionService):
    SIG_DEF = 10 ** -3
    X1_MIN = -2.0
    X1_MAX = 1.0
    Z_MIN = 0.0
    Z_MAX = 6.0

    def __init__(self, model_configuration, hypothesis=None, head=None, dynamical_model=None, model_name=None,
                 sde_mode="dWt", logger=LOG, **kwargs):
        self.sde_mode = sde_mode
        for constant, default in zip(["SIG_DEF", "X1_MIN", "X1_MAX", "Z_MIN", "Z_MAX"],
                                     [10 ** -3, -2.0, 1.0, 0.0, 6.0]):
            setattr(self, constant, kwargs.get(constant, default))
        if isequal_string(sde_mode, "dWt"):
            self.x1var = "x1_dWt"
            self.zvar = "z_dWt"
            default_model = "vep_dWt"
        else:
            self.x1var = "x1"
            self.zvar = "z"
            default_model = "vep_sde"
        if not (isinstance(model_name, basestring)):
            model_name = default_model
        self.sig = self.set_default_sig(dynamical_model, **kwargs)
        super(SDEModelInversionService, self).__init__(model_configuration, hypothesis, head, dynamical_model,
                                                       model_name, logger, **kwargs)
        self.set_default_parameters(**kwargs)

    def set_default_sig(self, dynamical_model, **kwargs):
        if kwargs.get("sig", None):
            return kwargs.pop("sig")
        elif np.in1d(dynamical_model, AVAILABLE_DYNAMICAL_MODELS_NAMES):
            if EPILEPTOR_MODEL_NVARS[dynamical_model] == 2:
                return model_noise_intensity_dict[dynamical_model][1]
            elif EPILEPTOR_MODEL_NVARS[dynamical_model] > 2:
                return model_noise_intensity_dict[dynamical_model][2]
        else:
            return self.SIG_DEF

    def set_default_parameters(self, **kwargs):
        # Generative model:
        # Integration:
        if isequal_string(self.sde_mode, "dWt"):
            self.default_parameters.update(set_parameter_defaults("x1_dWt", "normal", (),  # name, pdf, shape
                                                                  -6.0, 6.0,  # min, max
                                                                  pdf_params={"mean": 0.0, "sigma": 1.0}))
            self.default_parameters.update(set_parameter_defaults("z_dWt", "normal", (),  # name, pdf, shape
                                                                  -6.0, 6.0,  # min, max
                                                                  pdf_params={"mean": 0.0, "sigma": 1.0}))
        else:
            self.default_parameters.update(set_parameter_defaults("x1", "normal", (),  # name, pdf, shape
                                                                  self.X1_MIN, self.X1_MAX,  # min, max
                                                                  pdf_params={"mean": self.x1EQ, "sigma": 1.0}))
            self.default_parameters.update(set_parameter_defaults("z", "normal", (),  # name, pdf, shape
                                                                  self.Z_MIN, self.Z_MAX,  # min, max
                                                                  pdf_params={"mean": self.zEQ, "sigma": 1.0}))
        self.default_parameters.update(set_parameter_defaults("sig", "gamma", (),  # name, pdf, shape
                                                              0.0, 10.0 * self.sig,  # min, max
                                                              self.sig, self.sig, **kwargs))  # mean, (std)

    def generate_statistical_model(self, model_name=None, **kwargs):
        if model_name is None:
            model_name = self.model_name
        tic = time.time()
        self.logger.info("Generating model...")
        active_regions = kwargs.pop("active_regions", [])
        self.default_parameters.update(kwargs)
        model = SDEStatisticalModel(model_name, self.n_regions, active_regions, self.n_signals, self.n_times, self.dt,
                                    x1var=self.x1var, zvar=self.zvar, **self.default_parameters)
        self.model_generation_time = time.time() - tic
        self.logger.info(str(self.model_generation_time) + ' sec required for model generation')
        return model

    def generate_model_data(self, statistical_model, signals, gain_matrix=None):
        return super(SDEModelInversionService, self).generate_model_data(statistical_model, signals, gain_matrix,
                                                                         x1var=self.x1var, zvar=self.zvar)

    def plot_fit_results(self, est, statistical_model, signals, time=None, seizure_indices=None,
                         trajectories_plot=False,
                         save_flag=SAVE_FLAG, show_flag=SHOW_FLAG, figure_dir=FOLDER_FIGURES, figure_format=FIG_FORMAT,
                         x1_str="x", x0_str="PathologicalExcitability", mc_str="MC",
                         signals_str="fit_signals", sig_str="sig", eps_str="eps",
                         **kwargs):
        name = statistical_model.name + kwargs.get("_id_est", "")
        if time is None:
            time = np.array(range(signals.shape[0]))
        time = time.flatten()
        sig_prior = statistical_model.parameters["sig"].mean / statistical_model.sig
        eps_prior = statistical_model.parameters["eps"].mean
        plot_raster(time, sort_dict({'observation signals': signals,
                                     'observation signals fit': est[signals_str]}),
                    special_idx=seizure_indices, time_units=est.get('time_units', "ms"),
                    title=name + ": Observation signals vs fit rasterplot",
                    subtitles=['observation signals ' +
                               '\nobservation noise eps_prior =  ' + str(eps_prior) + " eps_post =" + str(est[eps_str]),
                               'observation signals fit'], offset=0.1,
                    labels=None, save_flag=save_flag, show_flag=show_flag, figure_dir=figure_dir,
                    figure_format=figure_format, figsize=VERY_LARGE_SIZE)
        plot_raster(time, sort_dict({'x1': est[x1_str], 'z': est["z"]}),
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
            plot_trajectories({'x1': est[x1_str], 'z(t)': est['z']}, special_idx=seizure_indices,
                              title=title, labels=self.region_labels, show_flag=show_flag, save_flag=save_flag,
                              figure_dir=FOLDER_FIGURES, figure_format=FIG_FORMAT, figsize=LARGE_SIZE)
        # plot connectivity
        conn_figure_name = name + "Model Connectivity"
        pyplot.figure(conn_figure_name, VERY_LARGE_SIZE)
        # plot_regions2regions(conn.weights, conn.region_labels, 121, "weights")
        MC_prior = statistical_model.parameters["MC"].mean
        K_prior = statistical_model.parameters["K"].mean
        plot_regions2regions(MC_prior, self.region_labels[statistical_model.active_regions], 121,
                             "Prior Model Connectivity" + "\nglobal scaling prior: K = " + str(K_prior))
        plot_regions2regions(est[mc_str], self.region_labels[statistical_model.active_regions], 122,
                             "Posterior Model  Connectivity" + "\nglobal scaling fit: K = " + str(est["K"]))
        save_figure(save_flag, pyplot.gcf(), conn_figure_name, figure_dir, figure_format)
        check_show(show_flag=show_flag)
