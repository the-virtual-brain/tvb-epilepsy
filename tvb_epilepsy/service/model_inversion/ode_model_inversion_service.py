
import time
from copy import deepcopy

import numpy as np
import pylab as pl

from tvb_epilepsy.base.utils.log_error_utils import initialize_logger, warning
from tvb_epilepsy.base.utils.data_structures_utils import isequal_string, ensure_list, sort_dict, construct_import_path
from tvb_epilepsy.base.utils.math_utils import select_greater_values_array_inds
from tvb_epilepsy.base.model.vep.sensors import Sensors
from tvb_epilepsy.base.model.statistical_models.ode_statistical_model import \
                                                        EULER_METHODS, OBSERVATION_MODEL_EXPRESSIONS, OBSERVATION_MODELS
from tvb_epilepsy.base.model.statistical_models.stochastic_parameter import set_model_parameter
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
        self.n_signals = 0
        self.signals_inds = range(self.n_signals)
        self.context_str = "from " + construct_import_path(__file__) + " import " + self.__class__.__name__
        self.context_str += "; from tvb_epilepsy.base.model.model_configuration import ModelConfiguration"
        self.create_str = "ODEModelInversionService(ModelConfiguration())"

    def get_default_sig_init(self):
        return 0.1

    def set_time(self, time=None):
        if time is not None:
            time = np.array(time)
            try:
                if time.size == 1:
                    self.dt = time
                    self.time = np.arange(self.dt * (self.n_times - 1))
                elif time.size == self.n_times:
                    self.time = time
                    self.dt = np.mean(self.time)
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

    def select_signals(self, signals, **kwargs):
        sensors = Sensors(self.sensors_labels, self.sensors_locations, projection=self.projection)
        power=kwargs.get("power", None)
        if power:
            if not(isinstance(power, np.ndarray)):
                power_inds = kwargs.get("power_inds", range(self.n_signals))
                power = np.sum(signals ** 2, axis=0)
                power = power/np.max(power)
        return sensors.select_contacts(rois=kwargs.get("rois", None), projection_th=kwargs.get("rois", None),
                                       power=power, power_inds=power_inds, power_th=kwargs.get("power_th", None))

    def set_signals_inds(self, signals, **kwargs):
        if kwargs.pop("select_signals", False):
            signals_inds = self.select_signals(signals, **kwargs)
        else:
            signals_inds = kwargs.get("signals_inds", self.signals_inds)
        if len(signals_inds) == 0:
            signals_inds = range(self.n_signals)
        elif len(signals_inds) < self.n_signals:
            try:
                signals = signals[:, signals_inds]
                self.observation_shape = signals.shape
                (self.n_times, self.n_signals) = self.observation_shape
            except:
                raise_value_error("Failed to extract signals with indices " + str(self.signals_inds) +
                                  " from signals' array with shape " + str(self.observation_shape) + "!")
        else:
            raise_value_error("Signals indices " + str(self.signals_inds) +
                              " more than signals'number " + str(self.n_signals) + "!")
        self.signals_inds = signals_inds
        return signals

    def set_target_data_and_time(self, target_data_type, target_data, statistical_model, **kwargs):
        if isequal_string(target_data_type, "simulated"):
            signals = self.set_simulated_target_data(target_data, statistical_model)
            self.target_data_type = "simulated"
        else:  # isequal_string(target_data_type, "empirical"):
            signals = self.set_empirical_target_data(target_data)
            self.target_data_type = "empirical"
        time = self.set_time(target_data.get("time", None))
        signals = self.set_signals_inds(signals, **kwargs)
        statistical_model.n_signals = self.n_signals
        statistical_model.n_times = self.n_times
        statistical_model.dt = self.dt
        return signals, time, statistical_model

    def set_empirical_target_data(self, target_data):
        if isinstance(target_data, dict):
            signals = target_data.get("signals", target_data.get("target_data", None))
        self.observation_shape = signals
        (self.n_times, self.n_signals) = self.observation_shape
        return signals

    def set_simulated_target_data(self, target_data,  statistical_model=None):
        #TODO: this function needs to be improved substantially. It lacks generality right now.
        if statistical_model is None or not(isinstance(target_data, dict)):
            signals = self.set_empirical_target_data(target_data)
        else:
            if statistical_model.observation_expression == "x1z_offset":
                signals = ((target_data["x1"].T - np.expand_dims(self.x1EQ, 1)).T +
                           (target_data["z"].T - np.expand_dims(self.zEQ, 1)).T) / 2.75
                # TODO: a better normalization
            elif statistical_model.observation_expression == "x1_offset":
                # TODO: a better normalization
                signals = (target_data["x1"].T - np.expand_dims(self.x1EQ, 1)).T / 2.0
            else: # statistical_models.observation_expression == "x1"
                signals = target_data["x1"]
            signals = signals[:, statistical_model.active_regions]
            if statistical_model.observation_model.find("seeg") >= 0:
                if not(isinstance(self.projection, np.ndarray)):
                    projection = np.eye(statistical_model.n_active_regions)
                else:
                    projection = self.projection[:, statistical_model.active_regions]
                signals = (np.dot(projection, signals.T)).T
            self.observation_shape = signals.shape
            (self.n_times, self.n_signals) = self.observation_shape
        return signals

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
        methods = ensure_list(methods)
        n_methods = len(methods)
        active_regions_th = ensure_list(kwargs.get("active_regions_th", None))
        n_thresholds = len(active_regions_th)
        if n_thresholds != n_methods:
            if n_thresholds ==1 and n_methods > 1:
                active_regions_th = np.repeat(active_regions_th, n_methods).tolist()
            else:
                raise_value_error("Number of input methods:\n" + str(methods) +
                                  "and active region thresholds:\n" + str(active_regions_th) +
                                  "does not match!")
        if reset:
            statistical_model.update_active_regions([])
        for m, th in zip(methods, active_regions_th):
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

    def generate_model_parameters(self, **kwargs):
        parameters = super(ODEModelInversionService, self).generate_model_parameters(**kwargs)
        # Integration
        parameters.update({"x1init": set_model_parameter("x1init", "normal", self.x1EQ, 0.1,
                                                         self.X1INIT_MIN, self.X1INIT_MAX, (self.n_regions,), False,
                                                          **kwargs)})
        parameters.update({"zinit": set_model_parameter("zinit", "normal", self.zEQ, 0.1,
                                                         self.ZINIT_MIN, self.ZINIT_MAX, (self.n_regions,), False,
                                                         **kwargs)})
        parameters.update({"sig_init": set_model_parameter("sig_init", "lognormal", 0.003, None,
                                                            0.0, lambda s: 2 * s, (), True, **kwargs)})
        # Observation model
        parameters.update({"scale_signal": set_model_parameter("scale_signal", "lognormal", 1.0, None,
                                                               lambda s: 0.5 * s, lambda s: 2 * s, (), True, **kwargs)})
        parameters.update({"offset_signal": set_model_parameter("offset_signal", "lognormal", 0.0, 1.0,
                                                                -1.0, 1.0, (), True, **kwargs)})
        return parameters

    def generate_statistical_model(self, model_name=None, **kwargs):
        if model_name is None:
            model_name = self.model_name
        tic = time.time()
        self.logger.info("Generating model...")
        model = ODEStatisticalModel(model_name, self.generate_model_parameters(**kwargs), self.n_regions,
                                   kwargs.get("active_regions", []), self.n_signals, self.n_times, self.dt, **kwargs)
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
            model_data.update({p.name + "_lo": p.low, p.name + "_hi": p.high})
            if isequal_string(p.name, x1var) or isequal_string(p.name, zvar):
                pass
            elif isequal_string(p.name, "x1eq") or isequal_string(p.name, "x1init") or isequal_string(p.name, "zinit"):
                    self.logger.info("For the moment only normal distribution is allowed for parameters " + p.name +
                                     "!\nIgnoring the selected probability distribution!")
            else:
                model_data.update({p.name + "_pdf": np.where(np.in1d(AVAILABLE_DISTRIBUTIONS, p.type))[0][0]})
                pdf_params = p.pdf_params().values()
                model_data.update({p.name + "_p1": pdf_params[0]})
                if len(pdf_params) == 1:
                    model_data.update({p.name + "_p2": pdf_params[0]})
                else:
                    model_data.update({p.name + "_p2": pdf_params[1]})
        return sort_dict(model_data)


def viz_phase_space(data):
    opt = len(data['x']) == 1
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
