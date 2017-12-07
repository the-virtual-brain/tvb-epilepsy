
import time
from copy import deepcopy

import numpy as np
import pylab as pl
from scipy.signal import decimate

from tvb_epilepsy.base.utils.log_error_utils import initialize_logger, warning
from tvb_epilepsy.base.utils.data_structures_utils import isequal_string, ensure_list, sort_dict, assert_arrays, \
                                                                        extract_dict_stringkeys, construct_import_path
from tvb_epilepsy.base.utils.math_utils import select_greater_values_array_inds
from tvb_epilepsy.base.model.vep.sensors import Sensors
from tvb_epilepsy.base.model.statistical_models.ode_statistical_model import \
                                                        EULER_METHODS, OBSERVATION_MODEL_EXPRESSIONS, OBSERVATION_MODELS
from tvb_epilepsy.service.stochastic_parameter_factory import set_parameter_defaults
from tvb_epilepsy.base.model.statistical_models.ode_statistical_model import ODEStatisticalModel
from tvb_epilepsy.service.probability_distribution_factory import AVAILABLE_DISTRIBUTIONS
from tvb_epilepsy.service.model_inversion.model_inversion_service import ModelInversionService
from tvb_epilepsy.tvb_api.epileptor_models import *


LOG = initialize_logger(__name__)


class ODEModelInversionService(ModelInversionService):

    X1INIT_MIN = -2.0
    X1INIT_MAX = 0.0
    ZINIT_MIN = 1.0
    ZINIT_MAX = 5.0

    def __init__(self, model_configuration, hypothesis=None, head=None, dynamical_model=None, model_name="vep_ode",
                 logger=LOG, **kwargs):
        super(ODEModelInversionService, self).__init__(model_configuration, hypothesis, head, dynamical_model,
                                                       model_name, logger, **kwargs)
        self.time = None
        self.dt = 0.0
        self.n_times = 0
        self.n_signals = self.n_regions
        self.data_type = "lfp"
        self.signals_inds = range(self.n_signals)
        self.sig_init = self.set_default_sig_init(**kwargs)
        self._set_default_parameters(**kwargs)
        self.context_str = "from " + construct_import_path(__file__) + " import " + self.__class__.__name__
        self.context_str += "; from tvb_epilepsy.base.model.model_configuration import ModelConfiguration"
        self.create_str = "ODEModelInversionService(ModelConfiguration())"

    def set_default_sig_init(self, **kwargs):
        return kwargs.pop("sig_init", self.sig_eq / 3)

    def set_time(self, time=None):
        if time is not None:
            time = np.array(time)
            try:
                if time.size == 1:
                    self.dt = time
                    return np.arange(self.dt * (self.n_times - 1))
                elif time.size == self.n_times:
                    self.dt = np.mean(time)
                    return time
                else:
                    raise_value_error("Input time is neither a scalar nor a vector of length equal " +
                                      "to target_data.shape[0]!" + "\ntime = " + str(time))
            except:
                raise_value_error(
                    "Input time is neither a scalar nor a vector of length equal to target_data.shape[0]!" +
                    "\ntime = " + str(time))
        else:
            raise_value_error("Input time is neither a scalar nor a vector of length equal to target_data.shape[0]!" +
                              "\ntime = " + str(time))

    def select_signals_seeg(self, signals, rois, auto_selection, **kwargs):
        sensors = Sensors(self.sensors_labels, self.sensors_locations, projection=self.projection)
        if auto_selection.find("rois") >= 0:
            if sensors.projection is not None:
                current_selection = sensors.select_contacts_rois(kwargs.get("rois", rois), self.signals_inds,
                                                                 kwargs.get("projection_th", None))
                inds = np.where([s in current_selection for s in self.signals_inds])[0]
                self.signals_inds = np.array(self.signals_inds)[inds].tolist()
                signals = signals[:, inds]
        if auto_selection.find("correlation-power") >= 0:
            power = kwargs.get("power", np.sum((signals-np.mean(signals, axis=0))**2, axis=0)/signals.shape[0])
            correlation = kwargs.get("correlation", np.corrcoef(signals.T))
            current_selection = sensors.select_contacts_corr(correlation, self.signals_inds, power=power,
                                                             n_electrodes=kwargs.get("n_electrodes"),
                                                             contacts_per_electrode=kwargs.get("contacts_per_electrode", 1),
                                                             group_electrodes=kwargs.get("group_electrodes", True))
            inds = np.where([s in current_selection for s in self.signals_inds])[0]
            self.signals_inds = np.array(self.signals_inds)[inds].tolist()
        elif auto_selection.find("power"):
            power = kwargs.get("power", np.sum(signals ** 2, axis=0) / signals.shape[0])
            inds = select_greater_values_array_inds(power, kwargs.get("power_th", None))
            self.signals_inds = (np.array(self.signals_inds)[inds]).tolist()
        return signals[:, inds]
    
    def select_signals_lfp(self, signals, rois, auto_selection, **kwargs):
        if auto_selection.find("rois") >= 0:
            if kwargs.get("rois", rois):
                inds = np.where([s in rois for s in self.signals_inds])[0]
                signals = signals[:, inds]
                self.signals_inds = np.array(self.signals_inds)[inds].tolist()
        if auto_selection.find("power") >= 0:
            power = kwargs.get("power", np.sum((signals-np.mean(signals, axis=0))**2, axis=0) / signals.shape[0])
            inds = select_greater_values_array_inds(power, kwargs.get("power_th",  None))
            signals = signals[:, inds]
            self.signals_inds = (np.array(self.signals_inds)[inds]).tolist()
        return signals

    def decimate_signals(self, time, signals, decim_ratio):
        signals = decimate(signals, decim_ratio, axis=0, zero_phase=True)
        time = decimate(time, decim_ratio, zero_phase=True)
        self.dt = np.mean(time)
        self.observation_shape = signals.shape
        (self.n_times, self.n_signals) = self.observation_shape
        return signals, time

    def cut_signals_tails(self, time, signals, cut_tails):
        signals = signals[cut_tails[0]:-cut_tails[-1]]
        time = time[cut_tails[0]:-cut_tails[-1]]
        self.observation_shape = signals.shape
        (self.n_times, self.n_signals) = self.observation_shape
        return signals, time

    def set_empirical_target_data(self, target_data, **kwargs):
        self.data_type = "seeg"
        self.signals_inds = range(len(self.sensors_labels))
        manual_selection = kwargs.get("manual_selection", [])
        if len(manual_selection) > 0:
            self.signals_inds = manual_selection
        if isinstance(target_data, dict):
            signals = target_data.get("signals", target_data.get("target_data", None))
        if len(self.signals_inds) != signals.shape[1]:
            signals = signals[:, self.signals_inds]
        self.observation_shape = signals.shape
        (self.n_times, self.n_signals) = self.observation_shape
        return signals

    def set_simulated_target_data(self, target_data,  statistical_model, **kwargs):
        self.signals_inds = range(self.n_regions)
        self.data_type = "lfp"
        if statistical_model.observation_model.find("seeg") >= 0:
            project = True
        if statistical_model.observation_expression == "x1z_offset":
            signals = ((target_data["x1"].T - np.expand_dims(self.x1EQ, 1)).T +
                       (target_data["z"].T - np.expand_dims(self.zEQ, 1)).T) / 2.75
            # TODO: a better normalization
        elif statistical_model.observation_expression == "x1_offset":
            # TODO: a better normalization
            signals = (target_data["x1"].T - np.expand_dims(self.x1EQ, 1)).T / 2.0
        else: # statistical_models.observation_expression == "lfp"
            if project:
                # try for SEEG
                signals = extract_dict_stringkeys(sort_dict(target_data), "SEEG",
                                                  modefun="find", break_after=1)
                if len(signals) > 0:
                    signals = signals.values()[0]
                    project = False
                    self.data_type = "seeg"
                    self.signals_inds = range(self.projection.shape[0])
                else:
                    signals = target_data.get("lfp", target_data["x1"])
        target_data["signals"] = signals
        manual_selection = kwargs.get("manual_selection", [])
        if len(manual_selection) > 0:
            self.signals_inds = manual_selection
        if len(self.signals_inds) != signals.shape[1]:
            signals = signals[:, self.signals_inds]
        if project is True:
            signals = (np.dot(self.projection[:, self.signals_inds], signals.T)).T
            self.data_type = "seeg"
            self.signals_inds = range(self.projection.shape[0])
        self.observation_shape = signals.shape
        (self.n_times, self.n_signals) = self.observation_shape
        return signals, target_data

    def set_target_data_and_time(self, target_data_type, target_data, statistical_model, **kwargs):
        if isequal_string(target_data_type, "simulated"):
            signals, target_data = self.set_simulated_target_data(target_data, statistical_model, **kwargs)
            self.target_data_type = "simulated"
        else:  # isequal_string(target_data_type, "empirical"):
            signals = self.set_empirical_target_data(target_data, **kwargs)
            self.target_data_type = "empirical"
        if kwargs.get("auto_selection", True) is not False:
            if self.data_type == "lfp":
                signals = self.select_signals_lfp(signals, statistical_model.active_regions,
                                                  kwargs.pop("auto_selection", "rois"), **kwargs)
            else:
                signals = self.select_signals_seeg(signals,  statistical_model.active_regions,
                                                   kwargs.pop("auto_selection", "rois-correlation-power"), **kwargs)
        time = self.set_time(target_data.get("time", None))
        if kwargs.get("decimate", 1) > 1:
            signals, time = self.decimate_signals(time, signals, kwargs.get("decimate"))
        if np.sum(kwargs.get("cut_signals_tails", (0, 0))) > 0:
            signals, time = self.cut_signals_tails(time, signals, kwargs.get("cut_signals_tails"))
        signals -= signals.min()
        signals /= signals.max()
        statistical_model.n_signals = self.n_signals
        statistical_model.n_times = self.n_times
        statistical_model.dt = self.dt
        return signals, time, statistical_model, target_data

    def update_active_regions_e_values(self, statistical_model, active_regions_th=0.1, reset=False):
        if reset:
            statistical_model.update_active_regions([])
        statistical_model.update_active_regions(statistical_model.active_regions +
                                            select_greater_values_array_inds(self.e_values, active_regions_th).tolist())
        return statistical_model

    def update_active_regions_x0_values(self, statistical_model, active_regions_th=0.1, reset=False):
        if reset:
            statistical_model.update_active_regions([])
        statistical_model.update_active_regions(statistical_model.active_regions +
                                           select_greater_values_array_inds(self.x0_values, active_regions_th).tolist())
        return statistical_model

    def update_active_regions_lsa(self, statistical_model, active_regions_th=None, reset=False):
        if reset:
            statistical_model.update_active_regions([])
        if len(self.lsa_propagation_strengths) > 0:
            ps_strengths = self.lsa_propagation_strengths / np.max(self.lsa_propagation_strengths)
            statistical_model.update_active_regions(statistical_model.active_regions +
                                             select_greater_values_array_inds(ps_strengths, active_regions_th).tolist())
        else:
            warning("No LSA results found (empty propagations_strengths vector)!" +
                    "\nSkipping of setting active_regios according to LSA!")
        return statistical_model

    def update_active_regions_seeg(self, statistical_model, active_regions_th=None, seeg_inds=[], reset=False):
        if reset:
            statistical_model.update_active_regions([])
        if self.projection is not None:
            active_regions = statistical_model.active_regions
            if len(seeg_inds) == 0:
                seeg_inds = self.signals_inds
            if len(seeg_inds) != 0:
                projection = self.projection[seeg_inds]
            else:
                projection = self.projection
            for proj in projection:
                active_regions += select_greater_values_array_inds(proj, active_regions_th).tolist()
            statistical_model.update_active_regions(active_regions)
        else:
            warning("Projection is not found!" + "\nSkipping of setting active_regios according to SEEG power!")
        return statistical_model

    def update_active_regions(self, statistical_model, methods=["e_values", "LSA"], reset=False, **kwargs):
        if reset:
            statistical_model.update_active_regions([])
        for m, th in zip(*assert_arrays([ensure_list(methods),
                                         ensure_list(kwargs.get("active_regions_th", None))])):
            if isequal_string(m, "e_values"):
                statistical_model = self.update_active_regions_e_values(statistical_model, th)
            elif isequal_string(m, "x0_values"):
                statistical_model = self.update_active_regions_x0_values(statistical_model, th)
            elif isequal_string(m, "lsa"):
                statistical_model = self.update_active_regions_lsa(statistical_model, th)
            elif isequal_string(m, "seeg"):
                statistical_model = self.update_active_regions_seeg(statistical_model, th,
                                                                    seeg_inds=kwargs.get("seeg_inds"))
        return statistical_model

    def _set_default_parameters(self, **kwargs):
        # Generative model:
        # Integration:
        self.default_parameters.update(set_parameter_defaults("x1init", "normal", (self.n_regions,),  # name, pdf, shape
                                                              self.X1INIT_MIN, self.X1INIT_MAX,       # min, max
                                                              pdf_params={"mean": self.x1EQ, "sigma":0.003}))
        self.default_parameters.update(set_parameter_defaults("zinit", "normal", (self.n_regions,),  # name, pdf, shape
                                                              self.ZINIT_MIN, self.ZINIT_MAX,  # min, max
                                                              pdf_params={"mean": self.zEQ, "sigma": 0.003}))
        self.default_parameters.update(set_parameter_defaults("sig_init", "lognormal", (),
                                                              0.0, 3.0*self.sig_init,
                                                              self.sig_init, self.sig_init / 6.0, **kwargs))
        self.default_parameters.update(set_parameter_defaults("scale_signal", "lognormal", (),
                                                              0.5, 1.5,
                                                              1.0, 0.1, **kwargs))
        self.default_parameters.update(set_parameter_defaults("offset_signal", "normal", (),
                                                              -0.5, 0.5,
                                                              0.0, 0.1, **kwargs))

    def generate_statistical_model(self, model_name=None, **kwargs):
        if model_name is None:
            model_name = self.model_name
        tic = time.time()
        self.logger.info("Generating model...")
        active_regions = kwargs.pop("active_regions", [])
        self.default_parameters.update(kwargs)
        model = ODEStatisticalModel(model_name, self.n_regions, active_regions, self.n_signals, self.n_times, self.dt,
                                    **self.default_parameters)
        self.model_generation_time = time.time() - tic
        self.logger.info(str(self.model_generation_time) + ' sec required for model generation')
        return model

    def generate_model_data(self, statistical_model, signals, projection=None, x1var="", zvar=""):
        active_regions_flag = np.zeros((statistical_model.n_regions,), dtype="i")
        active_regions_flag[statistical_model.active_regions] = 1
        if projection is None:
            projection = self.projection
        mixing = deepcopy(projection)
        if mixing.shape[0] > len(self.signals_inds):
            mixing = mixing[self.signals_inds]
        if mixing.shape[1] > statistical_model.n_active_regions:
            mixing = mixing[:, statistical_model.active_regions]
        model_data = {"n_regions": statistical_model.n_regions,
                      "n_times": statistical_model.n_times,
                      "n_signals": statistical_model.n_signals,
                      "n_active_regions": statistical_model.n_active_regions,
                      "n_nonactive_regions": statistical_model.n_nonactive_regions,
                      "active_regions_flag": np.array(active_regions_flag),
                      "active_regions": np.array(statistical_model.active_regions) + 1,  # cmdstan cannot take lists!
                      "nonactive_regions": np.where(1 - active_regions_flag)[0] + 1,  # indexing starts from 1!
                      "dt": statistical_model.dt,
                      "euler_method": np.where(np.in1d(EULER_METHODS, statistical_model.euler_method))[0][0] - 1,
                      "observation_model": np.where(np.in1d(OBSERVATION_MODELS,
                                                            statistical_model.observation_model))[0][0],
                      "observation_expression": np.where(np.in1d(OBSERVATION_MODEL_EXPRESSIONS,
                                                                 statistical_model.observation_expression))[0][0],
                      "signals": signals,
                      "mixing": mixing,
                      "x1eq0": statistical_model.parameters["x1eq"].mean}
        for key, val in self.epileptor_parameters.iteritems():
            model_data.update({key: val})
        for p in statistical_model.parameters.values():
            if isequal_string(p.name, x1var) or isequal_string(p.name, zvar):
                # TODO: find better solution with these boundaries
                model_data.update({p.name + "_lo": p.low, p.name + "_hi": p.high})
                pass
            elif isequal_string(p.name, "x1eq") or isequal_string(p.name, "x1init") or isequal_string(p.name, "zinit"):
                model_data.update({p.name + "_lo": p.low, p.name + "_hi": p.high})
                self.logger.info("For the moment only normal distribution is allowed for parameters " + p.name +
                                 "!\nIgnoring the selected probability distribution!")
            else:
                # TODO: find better solution with these boundaries
                model_data.update({p.name + "_lo": np.maximum(np.max(p.low / p.scale - p.loc), 0.0),
                                   p.name + "_hi": np.min(p.high / p.scale - p.loc)})
                model_data.update({p.name + "_loc": p.loc, p.name + "_scale": p.scale})
                model_data.update({p.name + "_pdf": np.where(np.in1d(AVAILABLE_DISTRIBUTIONS, p.type))[0][0]})
                model_data.update({p.name + "_p": np.array(p.pdf_params().values()) + np.ones((2,))})
        return sort_dict(model_data)


def viz_phase_space(data):
    opt = len(data['x1']) == 1
    npz = np.load('data.R.npz')
    tr = lambda A: np.transpose(A, (0, 2, 1))
    x, z = tr(data['x']), tr(data['z'])
    tau0 = npz['tau0']
    X, Z = np.mgrid[-5.0:5.0:50j, -5.0:5.0:50j]
    dX = (npz['I1'] + 1.0) - X ** 3.0 - 2.0 * X ** 2.0 - Z
    x0mean = data['x0'].mean(axis=0)
    Kmean = data['K'].mean(axis=0)

    def nullclines(i):
        pl.contour(X, Z, dX, 0, colors='r')
        dZ = (1.0 / tau0) * (4.0 * (X - x0mean[i])) - Z - Kmean * (-npz['Ic'][i] * (1.8 + X))
        pl.contour(X, Z, dZ, 0, colors='b')

    for i in range(x.shape[-1]):
        pl.subplot(2, 3, i + 1)
        if opt:
            pl.plot(x[0, :, i], z[0, :, i], 'k', alpha=0.5)
        else:
            for j in range(1 if opt else 10):
                pl.plot(x[-j, :, i], z[-j, :, i], 'k', alpha=0.2, linewidth=0.5)
        nullclines(i)
        pl.grid(True)
        pl.xlabel('x(t)')
        pl.ylabel('z(t)')
        # pl.title(f'node {i}')
    pl.tight_layout()


def viz_pair_plots(csv, keys, skip=0):
    n = len(keys)
    if isinstance(csv, dict):
        csv = [csv]  # following assumes list of chains' results
    for i, key_i in enumerate(keys):
        for j, key_j in enumerate(keys):
            pl.subplot(n, n, i * n + j + 1)
            for csvi in csv:
                if i == j:
                    pl.hist(csvi[key_i][skip:], 20, log=True)
                else:
                    pl.plot(csvi[key_j][skip:], csvi[key_i][skip:], '.')
            if i == 0:
                pl.title(key_j)
            if j == 0:
                pl.ylabel(key_i)
    pl.tight_layout()
