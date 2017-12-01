
import time

import numpy as np
from matplotlib import pyplot

from tvb_epilepsy.base.constants.configurations import FOLDER_FIGURES, FIG_FORMAT, LARGE_SIZE, VERY_LARGE_SIZE, \
                                                                                                    SAVE_FLAG, SHOW_FLAG
from tvb_epilepsy.base.constants.model_constants import model_noise_intensity_dict
from tvb_epilepsy.base.utils.log_error_utils import initialize_logger
from tvb_epilepsy.base.utils.data_structures_utils import isequal_string, construct_import_path, sort_dict
from tvb_epilepsy.base.utils.plot_utils import plot_raster, plot_regions2regions, plot_trajectories, save_figure, \
                                                                                                            check_show
from tvb_epilepsy.base.model.statistical_models.sde_statistical_model import SDEStatisticalModel
from tvb_epilepsy.service.stochastic_parameter_factory import set_parameter_defaults
from tvb_epilepsy.service.epileptor_model_factory import AVAILABLE_DYNAMICAL_MODELS_NAMES, EPILEPTOR_MODEL_NVARS
from tvb_epilepsy.service.model_inversion.ode_model_inversion_service import ODEModelInversionService


LOG = initialize_logger(__name__)


class SDEModelInversionService(ODEModelInversionService):

    SIG_DEF = 10 ** -4
    X1_MIN = -3.0
    X1_MAX = 3.0
    Z_MIN = 0.0
    Z_MAX = 7.0

    def __init__(self, model_configuration, hypothesis=None, head=None, dynamical_model=None, model_name=None, 
                 sde_mode="dWt", logger=LOG, **kwargs):
        self.sde_mode = sde_mode
        if isequal_string(sde_mode, "dWt"):
            self.x1var = "x1_dWt"
            self.zvar = "z_dWt"
            default_model = "vep_dWt"
        else:
            self.x1var = "x1"
            self.zvar = "z"
            default_model = "vep_sde"
        if not(isinstance(model_name, basestring)):
            model_name = default_model
        self.sig = self.set_default_sig(dynamical_model, **kwargs)
        super(SDEModelInversionService, self).__init__(model_configuration, hypothesis, head, dynamical_model,
                                                       model_name, logger, **kwargs)
        self.set_default_parameters(**kwargs)
        self.context_str = "from " + construct_import_path(__file__) + " import " + self.__class__.__name__
        self.context_str += "; from tvb_epilepsy.base.model.model_configuration import ModelConfiguration"
        self.create_str = "ODEModelInversionService(ModelConfiguration())"

    def set_default_sig(self, dynamical_model, **kwargs):
        if kwargs.get("sig", None):
            return kwargs.pop("sig")
        elif np.in1d(dynamical_model, AVAILABLE_DYNAMICAL_MODELS_NAMES):
                if EPILEPTOR_MODEL_NVARS[dynamical_model] == 2:
                    return model_noise_intensity_dict[dynamical_model][1]
                elif EPILEPTOR_MODEL_NVARS[dynamical_model] > 2:
                    return model_noise_intensity_dict[dynamical_model][2]
        else:
            return 1.0 / self.SIG_DEF

    def set_default_parameters(self, **kwargs):
        # Generative model:
        # Integration:
        if isequal_string(self.sde_mode, "dWt"):
            self.default_parameters.update(set_parameter_defaults("x1_dWt", "normal", (),  # name, pdf, shape
                                                                  -6.0, 6.0,               # min, max
                                                                  pdf_params={"mean": 0.0, "sigma": 1.0}))
            self.default_parameters.update(set_parameter_defaults("z_dWt", "normal", (),  # name, pdf, shape
                                                                  -6.0, 6.0,              # min, max
                                                                  pdf_params={"mean": 0.0, "sigma": 1.0}))
        else:
            self.default_parameters.update(set_parameter_defaults("x1", "normal", (),        # name, pdf, shape
                                                                  self.X1_MIN, self.X1_MAX,  # min, max
                                                                  pdf_params={"mean": self.x1EQ, "sigma": 1.0}))
            self.default_parameters.update(set_parameter_defaults("z", "normal", (),          # name, pdf, shape
                                                                  self.Z_MIN, self.Z_MAX,     # min, max
                                                                  pdf_params={"mean": self.zEQ, "sigma": 1.0}))
        self.default_parameters.update(set_parameter_defaults("sig", "gamma", (),              # name, pdf, shape
                                                              0.0, 10.0*self.sig,              # min, max
                                                              self.sig, self.sig, **kwargs))   # mean, (std)

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

    def generate_model_data(self, statistical_model, signals, projection=None):
        return super(SDEModelInversionService, self).generate_model_data(statistical_model, signals, projection,
                                                                         x1var=self.x1var, zvar=self.zvar)

    def plot_fit_results(self, est, statistical_model, signals, time=None, seizure_indices=None, trajectories_plot=False,
                        save_flag=SAVE_FLAG, show_flag=SHOW_FLAG, figure_dir=FOLDER_FIGURES, figure_format=FIG_FORMAT,
                        **kwargs):
        name = statistical_model.name + kwargs.get("_id_est", "")
        if time is None:
            time = np.array(range(signals.shape[0]))
        time = time.flatten()
        sig_prior = statistical_model.parameters["sig"].mean / statistical_model.sig
        eps_prior = statistical_model.parameters["eps"].mean
        plot_raster(time, sort_dict({'observation signals': signals,
                                     'observation signals fit': est['fit_signals']}),
                    special_idx=seizure_indices, time_units=est.get('time_units', "ms"),
                    title=name + ": Observation signals vs fit rasterplot",
                    subtitles=['observation signals ' +
                               '\nobservation noise eps_prior =  ' + str(eps_prior) + " eps_post =" + str(est["eps"]),
                               'observation signals fit'], offset=3.0,
                    labels=None, save_flag=save_flag, show_flag=show_flag, figure_dir=figure_dir,
                    figure_format=figure_format, figsize=VERY_LARGE_SIZE)
        plot_raster(time, sort_dict({'x1': est["x1"], 'z': est["z"]}),
                    special_idx=seizure_indices, time_units=est.get('time_units', "ms"),
                    title=name + ": Hidden states fit rasterplot",
                    subtitles=['hidden state x1' '\ndynamic noise sig_prior = ' + str(sig_prior) +
                               " sig_post = " + str(est["sig"]/statistical_model.sig),
                               'hidden state z'], offset=3.0,
                    labels=None, save_flag=save_flag, show_flag=show_flag, figure_dir=figure_dir,
                    figure_format=figure_format, figsize=VERY_LARGE_SIZE)
        if trajectories_plot:
            title = name + ': Fit hidden state space trajectories'
            title += "\n prior x0: " + str(self.x0_values)
            title += "\n x0 fit: " + str(est["PathologicalExcitability"])
            plot_trajectories({'x1': est['x'], 'z(t)': est['z']}, special_idx=seizure_indices,
                              title=title, labels=self.region_labels, show_flag=show_flag, save_flag=save_flag,
                              figure_dir=FOLDER_FIGURES, figure_format=FIG_FORMAT, figsize=LARGE_SIZE)
        # plot connectivity
        conn_figure_name = name + "Model Connectivity"
        pyplot.figure(conn_figure_name, VERY_LARGE_SIZE)
        # plot_regions2regions(conn.weights, conn.region_labels, 121, "weights")
        MC_prior = statistical_model.parameters["MC"].mean / statistical_model.MC_scale
        K_prior = statistical_model.parameters["K"].mean
        plot_regions2regions(MC_prior, self.region_labels[statistical_model.active_regions], 131,
                             "Prior Model Connectivity" + "\nglobal scaling prior: K = " + str(K_prior))
        plot_regions2regions(est['model_connectivity'], self.region_labels[statistical_model.active_regions], 132,
                             "Posterior Model  Connectivity" + "\nglobal scaling fit: K = " + str(est["K"]))
        plot_regions2regions(est['MC']-1.0, self.region_labels[statistical_model.active_regions], 133,
                             "Posterior-Prior MC")
        save_figure(save_flag, pyplot.gcf(), conn_figure_name, figure_dir, figure_format)
        check_show(show_flag=show_flag)