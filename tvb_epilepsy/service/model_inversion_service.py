import numpy as np

from tvb_epilepsy.base.utils import raise_value_error, formal_repr, sort_dict, ensure_list
from tvb_epilepsy.base.model.model_configuration import ModelConfiguration
from tvb_epilepsy.base.model.disease_hypothesis import DiseaseHypothesis
from tvb_epilepsy.service.epileptor_model_factory import model_noise_intensity_dict
from tvb_epilepsy.tvb_api.epileptor_models import *
from tvb_epilepsy.custom.simulator_custom import EpileptorModel
from tvb.simulator.models import Epileptor

AVAILABLE_DYNAMICAL_MODELS = (Epileptor, EpileptorModel, EpileptorDP2D, EpileptorDP, EpileptorDPrealistic)

class ModelInversionService(object):

    def __init__(self, model_configuration, hypothesis, target_data, time=None, fs=None, dynamical_model=None,
                 active_regions=None,  region_labels=None,  active_regions=None,
                 active_regions_th=0.1, euler_method=-1, observation_model=1, observation_expression=1,
                 sensors=None, channel_inds=None, **kwargs):

        # model configuration
        if isinstance(model_configuration, ModelConfiguration):
            self.model_configuration = model_configuration
        else:
            raise_value_error("Input model configuration is not a ModelConfiguration object:\n"
                              + str(model_configuration))

        # hypothesis
        if isinstance(hypothesis, DiseaseHypothesis):
            self.hypothesis = hypothesis
        else:
            raise_value_error("Input hypothesis is not a DiseaseHypothesis object:\n" + str(hypothesis))

        # dynamical model and defaults for time scales and noise
        if isinstance(dynamical_model, AVAILABLE_DYNAMICAL_MODELS):
            self.dynamical_model = dynamical_model
            noise_intensity = kwargs.get("noise_intensity", model_noise_intensity_dict[self.dynamic_model._ui_name])
            self.sig_def = np.mean(noise_intensity)
            if isinstance(self.dynamic_model, (Epileptor, EpileptorModel)):
                self.tau1_def = kwargs.get("tau1", np.mean(1.0 / self.dynamic_model.r))
                self.tau0_def = kwargs.get("tau0", np.mean(self.dynamic_model.tt))

            elif isinstance(self.dynamic_model, (EpileptorDP, EpileptorDP2D, EpileptorDPrealistic)):
                self.tau1_def =  kwargs.get("tau1", np.mean(self.dynamic_model.tau1))
                self.tau0_def =  kwargs.get("tau0", np.mean(self.dynamic_model.tau0))

        else:
            self.tau1_def =  kwargs.get("tau1", 0.5)
            self.tau0_def =  kwargs.get("tau0", 30)
            self.sig_def = kwargs.get("noise_intensity", 10 ** -4)

        # active regions
        self.n_regions = self.hypothesis.number_of_regions
        active_regions_flag = np.zeros((self.n_regions,), dtype="i")
        self.active_regions_th = active_regions_th
        if active_regions is None:
            # Initialize as all those regions whose equilibria lie further away from the healthy equilibrium:
            self.active_regions = np.where(model_configuration.e_values > self.active_regions_th)[0]
            # If LSA has been run, add all regions with a propagation strength greater than the minimal one:
            if len(hypothesis.propagation_strengths) > 0:
                self.active_regions = np.unique(self.active_regions.tolist() +
                                      np.where(hypothesis.propagation_strengths /
                                               np.max(hypothesis.propagation_strengths)
                                                                                    > active_regions_th)[0].tolist())
            else:
                self.active_regions = active_regions

        self.active_regions_flag[self.active_regions] = 1
        self.n_active_regions = len(self.active_regions)
        self.nonactive_regions = np.where(1-self.active_regions_flag)[0]
        self.n_nonactive_regions = len(self.nonactive_regions)

        if isinstance(target_data, np.ndarray):
            self.signals = target_data
            self.data_type = "empirical"

        elif isinstance(target_data, dict):
            self.signals = target_data.get("signals", None)
            if self.signals  is None:
                if observation_expression == 1:
                    self.signals = (target_data["x1"][:, self.active_regions].T -
                                    np.expand_dims(self.model_configuration.x1EQ[self.active_regions], 1)).T + \
                                    (target_data["z"][:, active_regions].T -
                                     np.expand_dims(model_configuration.zEQ[self.active_regions], 1)).T
                    # TODO: a better normalization
                    self.signals = self.signals / 2.75
                elif observation_expression == 2:
                    # TODO: a better normalization
                    signals = (target_data["x1"][:, self.active_regions].T -
                               np.expand_dims(model_configuration.x1EQ[self.active_regions], 1)).T / 2.0
                else:
                    signals = target_data["x1"][:, self.active_regions]

        else:
            raise_value_error("Input target data is neither a ndarray of empirical data nor a dictionary of "
                              "simulated data:\n" + str(target_data))

        (self.n_times, self.n_signals) = self.signals


        logger.info("Constructing data dictionary...")


        # Gamma distributions' parameters
        # visualize gamma distributions here: http://homepage.divms.uiowa.edu/~mbognar/applets/gamma.html
        tau1_mu = tau1_def
        tau1 = gamma_from_mu_std(kwargs.get("tau1_mu", tau1_mu), kwargs.get("tau1_std", 3 * tau1_mu))
        tau0_mu = tau0_def
        tau0 = gamma_from_mu_std(kwargs.get("tau0_mu", tau0_mu), kwargs.get("tau0_std", 3 * 10000.0))
        K_def = np.mean(model_configuration.K)
        K = gamma_from_mu_std(kwargs.get("K_mu", K_def),
                              kwargs.get("K_std", 10 * K_def))
        # zero effective connectivity:
        conn0 = gamma_from_mu_std(kwargs.get("conn0_mu", 0.001), kwargs.get("conn0_std", 0.001))
        if noise_intensity is None:
            sig_mu = np.mean(model_noise_intensity_dict["EpileptorDP2D"])
        else:
            sig_mu = noise_intensity
        sig = gamma_from_mu_std(kwargs.get("sig_mu", sig_mu), kwargs.get("sig_std", 3 * sig_mu))
        sig_eq_mu = (X1_EQ_CR_DEF - X1_DEF) / 3.0
        sig_eq_std = 3 * sig_eq_mu
        sig_eq = gamma_from_mu_std(kwargs.get("sig_eq_mu", sig_eq_mu), kwargs.get("sig_eq_std", sig_eq_std))
        sig_init_mu = sig_eq_mu
        sig_init_std = sig_init_mu
        sig_init = gamma_from_mu_std(kwargs.get("sig_init_mu", sig_init_mu), kwargs.get("sig_init_std", sig_init_std))

        if mixing is None or len(channel_inds) < 1:
            if observation_model == 2:
                mixing = np.random.rand(n_active_regions, n_active_regions)
                for ii in range(len(n_active_regions)):
                    mixing[ii, :] = mixing[ii, :] / np.sum(mixing[ii, :])
            else:
                observation_model = 3
                mixing = np.eye(n_active_regions)

        else:
            mixing = mixing[channel_inds][:, active_regions]
            for ii in range(len(channel_inds)):
                mixing[ii, :] = mixing[ii, :] / np.sum(mixing[ii, :])


            signals = (np.dot(mixing, signals.T)).T

        # from matplotlib import pyplot
        # pyplot.plot(signals)
        # pyplot.show()

        data = {"n_regions": hypothesis.number_of_regions,
                "n_active_regions": n_active_regions,
                "n_nonactive_regions": hypothesis.number_of_regions - n_active_regions,
                "active_regions_flag": active_regions_flag,
                "n_time": signals.shape[0],
                "n_signals": signals.shape[1],
                "x0_nonactive": model_configuration.x0[~active_regions_flag.astype("bool")],
                "x1eq0": model_configuration.x1EQ,
                "zeq0": model_configuration.zEQ,
                "x1eq_lo": kwargs.get("x1eq_lo", -2.0),
                "x1eq_hi": kwargs.get("x1eq_hi", X1_EQ_CR_DEF),
                "x1init_lo": kwargs.get("x1init_lo", -2.0),
                "x1init_hi": kwargs.get("x1init_hi", -1.0),
                "x1_lo": kwargs.get("x1_lo", -2.5),
                "x1_hi": kwargs.get("x1_hi", 1.5),
                "z_lo": kwargs.get("z_lo", 2.0),
                "z_hi": kwargs.get("z_hi", 5.0),
                "tau1_lo": kwargs.get("tau1_lo", tau1_mu / 2),
                "tau1_hi": kwargs.get("tau1_hi", np.min([3 * tau1_mu / 2, 1.0])),
                "tau0_lo": kwargs.get("tau0_lo", np.min([tau0_mu / 2, 10])),
                "tau0_hi": kwargs.get("tau0_hi", np.max([3 * tau1_mu / 2, 30.0])),
                "tau1_a": kwargs.get("tau1_a", tau1["alpha"]),
                "tau1_b": kwargs.get("tau1_b", tau1["beta"]),
                "tau0_a": kwargs.get("tau0_a", tau0["alpha"]),
                "tau0_b": kwargs.get("tau0_b", tau0["beta"]),
                "SC": model_configuration.connectivity_matrix,
                "SC_sig": kwargs.get("SC_sig", 0.1),
                "K_lo": kwargs.get("K_lo", K_def / 10.0),
                "K_hi": kwargs.get("K_hi", 30.0 * K_def),
                "K_a": kwargs.get("K_a", K["alpha"]),
                "K_b": kwargs.get("K_b", K["beta"]),
                "gamma0": kwargs.get("gamma0", np.array([conn0["alpha"], conn0["beta"]])),
                "dt": 1000.0 / fs,
                "sig_hi": kwargs.get("sig_hi", 3 * sig_mu),
                "sig_a": kwargs.get("sig_a", sig["alpha"]),
                "sig_b": kwargs.get("sig_b", sig["beta"]),
                "sig_eq_hi": kwargs.get("sig_eq_hi", sig_eq_std),
                "sig_eq_a": kwargs.get("sig_eq_a", sig_eq["alpha"]),
                "sig_eq_b": kwargs.get("sig_eq_b", sig_eq["beta"]),
                "sig_init_mu": kwargs.get("sig_init_mu", sig_init_mu),
                "sig_init_hi": kwargs.get("sig_init_hi", sig_init_std),
                "sig_init_a": kwargs.get("sig_init_a", sig_init["alpha"]),
                "sig_init_b": kwargs.get("sig_init_b", sig_init["beta"]),
                "observation_model": observation_model,
                "signals": signals,
                "mixing": mixing,
                "eps_hi": kwargs.get("eps_hi", (np.max(signals.flatten()) - np.min(signals.flatten()) / 100.0)),
                "eps_x0": kwargs.get("eps_x0", 0.1),
                }

        for p in ["a", "b", "d", "yc", "Iext1", "slope"]:

            temp = getattr(model_configuration, p)
            if isinstance(temp, (np.ndarray, list)):
                if np.all(temp[0], np.array(temp)):
                    temp = temp[0]
                else:
                    raise_not_implemented_error("Statistical models where not all regions have the same value " +
                                                " for parameter " + p + " are not implemented yet!")
            data.update({p: temp})

        zeq_lo = calc_eq_z(data["x1eq_hi"], data["yc"], data["Iext1"], "2d", x2=0.0, slope=data["slope"], a=data["a"],
                           b=data["b"], d=data["d"])
        zeq_hi = calc_eq_z(data["x1eq_lo"], data["yc"], data["Iext1"], "2d", x2=0.0, slope=data["slope"], a=data["a"],
                           b=data["b"], d=data["d"])
        data.update({"zeq_lo": kwargs.get("zeq_lo", zeq_lo),
                     "zeq_hi": kwargs.get("zeq_hi", zeq_hi)})
        data.update({"zinit_lo": kwargs.get("zinit_lo", zeq_lo - sig_init_std),
                     "zinit_hi": kwargs.get("zinit_hi", zeq_hi + sig_init_std)})

        x0cr, rx0 = calc_x0cr_r(data["yc"], data["Iext1"], data["a"], data["b"], data["d"], zmode=np.array("lin"),
                                x1_rest=X1_DEF, x1_cr=X1_EQ_CR_DEF, x0def=X0_DEF, x0cr_def=X0_CR_DEF, test=False,
                                shape=None, calc_mode="non_symbol")

        data.update({"x0cr": x0cr, "rx0": rx0})
        logger.info("data dictionary completed with " + str(len(data)) + " fields:\n" + str(data.keys()))

