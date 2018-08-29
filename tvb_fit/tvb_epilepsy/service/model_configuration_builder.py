# coding=utf-8
"""
Service to create a model configuration, starting from a disease hypothesis and/or a configured TVB simulator
"""

import numpy as np

from tvb.simulator.models.base import Model
from tvb.simulator.simulator import Simulator
from tvb.datatypes.connectivity import Connectivity as TVBConnectivity

from tvb_fit.base.utils.log_error_utils import initialize_logger, warning, raise_value_error
from tvb_fit.base.utils.data_structures_utils import formal_repr, ensure_list
from tvb_fit.base.computations.math_utils import normalize_weights
from tvb_fit.base.model.virtual_patient.connectivity import Connectivity
from tvb_fit.service.model_configuration_builder import ModelConfigurationBuilder as ModelConfigurationBuilderBase
from tvb_fit.tvb_epilepsy.base.constants.model_constants import *
from tvb_fit.tvb_epilepsy.base.computation_utils.calculations_utils import calc_x0cr_r, calc_coupling, calc_x0, \
    calc_x0_val_to_model_x0, calc_model_x0_to_x0_val
from tvb_fit.tvb_epilepsy.base.computation_utils.equilibrium_computation import calc_eq_z, eq_x1_hypo_x0_linTaylor, \
    eq_x1_hypo_x0_optimize, compute_initial_conditions_from_eq_point
from tvb_fit.tvb_epilepsy.base.model.epileptor_model_configuration \
    import EpileptorModelConfiguration as ModelConfiguration
from tvb_fit.tvb_epilepsy.base.model.epileptor_model_configuration import EPILEPTOR_PARAMS
from tvb_fit.tvb_epilepsy.service.simulator.epileptor_model_factory import EPILEPTOR_MODEL_NVARS


class ModelConfigurationBuilder(ModelConfigurationBuilderBase):
    logger = initialize_logger(__name__)

    # For the momdent coupling, monitor, and noise are left to be None.
    # If in the future they are targeted for probabilistic modeling they will obtain contents

    x0 = np.array([-2.0])
    a = np.array([A_DEF])
    b = np.array([B_DEF])
    yc = np.array([YC_DEF])
    d = np.array([D_DEF])
    Iext1 = np.array([I_EXT1_DEF])
    Iext2 = np.array([I_EXT2_DEF])
    slope = np.array([SLOPE_DEF])
    s = np.array([S_DEF])
    gamma = np.array([GAMMA_DEF])
    tau1 = np.array([TAU1_DEF])
    tau0 = np.array([TAU0_DEF])
    tau2 = np.array([TAU2_DEF])
    zmode = np.array([ZMODE_DEF])
    pmode = np.array([PMODE_DEF])
    K = np.array([K_DEF])
    Kvf = np.array([0.0])
    Kf = np.array([0.0])

    def __init__(self, input="EpileptorDP", connectivity=None, K_unscaled=np.array([K_UNSCALED_DEF]),
                 x0_values=X0_DEF, e_values=E_DEF, x1eq_mode="optimize", **kwargs):
        if isinstance(input, Simulator):
            # TODO: make this more specific once we clarify the model configuration representation compared to simTVB
            self.model_name = input.model._ui_name
            self.set_params_from_tvb_model(input.model)
            self.connectivity = normalize_weights(input.connectivity.weights)
            # self.coupling = input.coupling
            self.initial_conditions = np.squeeze(input.initial_conditions)  # initial conditions in a reduced form
            # self.noise = input.integrator.noise
            # self.monitor = ensure_list(input.monitors)[0]
        else:
            if isinstance(input, Model):
                self.model_name = input._ui_name
                self.set_params_from_tvb_model(input)
            elif isinstance(input, basestring):
                self.model_name = input
            else:
                raise_value_error("Input (%s) is not a TVB simulator, an epileptor model, "
                                  "\nor a string of an epileptor model!")
        if isinstance(connectivity, Connectivity):
            self.connectivity = connectivity.normalized_weights
        elif isinstance(connectivity, TVBConnectivity):
            self.connectivity = normalize_weights(connectivity.weights)
        elif isinstance(connectivity, np.ndarray):
            self.connectivity = normalize_weights(connectivity)
        else:
            if not(isinstance(input, Simulator)):
                warning("Input connectivity (%s) is not a virtual patient connectivity, a TVB connectivity, "
                        "\nor a numpy.array!" % str(connectivity))
        self.x0_values = x0_values * np.ones((self.number_of_regions,), dtype=np.float32)
        self.x1eq_mode = x1eq_mode
        if len(ensure_list(K_unscaled)) == 1:
            K_unscaled = np.array(K_unscaled) * np.ones((self.number_of_regions,), dtype=np.float32)
        elif len(ensure_list(K_unscaled)) == self.number_of_regions:
            K_unscaled = np.array(K_unscaled)
        else:
            self.logger.warning(
                "The length of input global coupling K_unscaled is neither 1 nor equal to the number of regions!" +
                "\nSetting model_configuration_builder.K_unscaled = K_UNSCALED_DEF for all regions")
        self.set_K_unscaled(K_unscaled)
        for pname in EPILEPTOR_PARAMS:
            self.set_parameter(pname, kwargs.get(pname, getattr(self, pname)))
        # Update K_unscaled
        self.e_values = e_values * np.ones((self.number_of_regions,), dtype=np.float32)
        self.x0cr = 0.0
        self.rx0 = 0.0
        self._compute_critical_x0_scaling()

    def __repr__(self):
        d = {"01. model": self.model,
             "02. Number of regions": self.number_of_regions,
             "03. x0_values": self.x0_values,
             "04. e_values": self.e_values,
             "05. K": self.K,
             "06. x1eq_mode": self.x1eq_mode,
             "07. connectivity": self.connectivity,
             "08. coupling": self.coupling,
             "09. monitor": self.monitor,
             "10. initial_conditions": self.initial_conditions,
             "11. noise": self.noise

             }
        return formal_repr(self, d)

    def set_params_from_tvb_model(self, model):
        for pname in ["x0", "a", "b", "d", "Iext2", "slope", "gamma", "tt", "r", "tau2", "Kvf", "Kf"]:
            self.set_parameter(pname, getattr(model, pname))

        if model._ui_name == "Epileptor":
            for pname in ["c","Iext", "aa", "tt", "Ks"]:
                self.set_parameter(pname, getattr(model, pname))
        else:
            for pname in ["yc","Iext1", "s", "tau1", "K"]:
                self.set_parameter(pname, getattr(model, pname))
            if model._ui_name == "EpileptorDPrealistic":
                for pname in ["zmode", "pmode"]:
                    self.set_parameter(pname, getattr(model, pname))
        return self

    def set_parameter(self, pname, pval):
        if pname == "tt":
            self.tau1 = pval * np.ones((self.number_of_regions,), dtype=np.float32)
        elif pname == "r":
            self.tau0 = 1.0 / pval
        elif pname == "c":
            self.yc = pval * np.ones((self.number_of_regions,), dtype=np.float32)
        elif pname == "Iext":
            self.Iext1 = pval * np.ones((self.number_of_regions,), dtype=np.float32)
        elif pname == "s":
            self.s = pval * np.ones((self.number_of_regions,), dtype=np.float32)
        elif pname == "Ks":
            self.K = -pval * np.ones((self.number_of_regions,), dtype=np.float32)
        elif pval is not None:
            setattr(self, pname, pval * np.ones((self.number_of_regions,), dtype=np.float32))
        else:
            setattr(self, pname, pval)
        return self

    def build_model_config_from_tvb(self):
        model = self.model
        del model["model_name"]
        model_config = ModelConfiguration(self.model_name, self.connectivity, self.coupling,
                                          self.monitor, self.initial_conditions, self.noise,
                                          self.x0_values, self.e_values, x1eq=None, zeq=None, Ceq=None, **model)
        if model_config.initial_conditions is None:
            model_config.initial_conditions = compute_initial_conditions_from_eq_point(model_config)
        return model_config

    def build_model_config_from_model_config(self, model_config):
        if not isinstance(model_config, dict):
            model_config_dict = model_config.__dict__
        else:
            model_config_dict = model_config
        model_configuration = ModelConfiguration()
        for attr, value in model_configuration.__dict__.items():
            value = model_config_dict.get(attr, None)
            if value is None:
                warning(attr + " not found in the input model configuration dictionary!" +
                        "\nLeaving default " + attr + ": " + str(getattr(model_configuration, attr)))
            if value is not None:
                setattr(model_configuration, attr, value)
        return model_configuration

    def set_K_unscaled(self, K_unscaled):
        self._normalize_global_coupling(K_unscaled)

    def update_K(self):
        self.set_K_unscaled(self.K * self.number_of_regions)
        return self

    @property
    def K_unscaled(self):
        # !!Very important to correct here for the sign of K!!
        return self.K * self.number_of_regions

    @property
    def model_K(self):
        return -self.K

    @property
    def Ks(self):
        # !!Very important to correct here for the sign of K!!
        return -self.K

    @property
    def c(self):
        return self.yc

    @property
    def Iext(self):
        return self.Iext1

    @property
    def aa(self):
        return self.s

    @property
    def tt(self):
        return self.tau1

    @property
    def model(self):
        return {pname: getattr(self, pname) for pname in ["model_name"] + EPILEPTOR_PARAMS}

    @property
    def nvar(self):
        return EPILEPTOR_MODEL_NVARS[self.model_name]

    def _compute_model_x0(self, x0_values, x0_indices=None):
        if x0_indices is None:
            x0_indices = np.array(range(self.number_of_regions))
        return calc_x0_val_to_model_x0(x0_values, self.yc[x0_indices], self.Iext1[x0_indices], self.a[x0_indices],
                                       self.b[x0_indices], self.d[x0_indices], self.zmode[x0_indices])

    def _ensure_equilibrum(self, x1eq, zeq):
        temp = x1eq > self.x1eq_cr - 10 ** (-3)
        if temp.any():
            x1eq[temp] = self.x1eq_cr - 10 ** (-3)
            zeq = self._compute_z_equilibrium(x1eq)

        return x1eq, zeq

    def _compute_x1_equilibrium_from_E(self, e_values):
        array_ones = np.ones((self.number_of_regions,), dtype=np.float32)
        return ((e_values - 5.0) / 3.0) * array_ones

    def _compute_z_equilibrium(self, x1eq):
        return calc_eq_z(x1eq, self.yc, self.Iext1, "2d", slope=self.slope, a=self.a, b=self.b, d=self.d)

    def _compute_critical_x0_scaling(self):
        (self.x0cr, self.rx0) = calc_x0cr_r(self.yc, self.Iext1, a=self.a, b=self.b, d=self.d, zmode=self.zmode)

    def _compute_coupling_at_equilibrium(self, x1eq, model_connectivity):
        return calc_coupling(x1eq, self.K, model_connectivity)

    def _compute_x0_values_from_x0_model(self, x0):
        return calc_model_x0_to_x0_val(x0, self.yc, self.Iext1, self.a, self.b, self.d, self.zmode)

    def _compute_x0_values(self, x1eq, zeq, model_connectivity):
        x0 = calc_x0(x1eq, zeq, self.K, model_connectivity)
        return self._compute_x0_values_from_x0_model(x0)

    def _compute_e_values(self, x1eq):
        return 3.0 * x1eq + 5.0

    def _compute_params_after_equilibration(self, x1eq, zeq, model_connectivity):
        self._compute_critical_x0_scaling()
        Ceq = self._compute_coupling_at_equilibrium(x1eq, model_connectivity)
        x0_values = self._compute_x0_values(x1eq, zeq, model_connectivity)
        e_values = self._compute_e_values(x1eq)
        x0 = self._compute_model_x0(x0_values)
        return x0, Ceq, x0_values, e_values

    def _compute_x1_and_z_equilibrium_from_E(self, e_values):
        x1EQ = self._compute_x1_equilibrium_from_E(e_values)
        zEQ = self._compute_z_equilibrium(x1EQ)
        return x1EQ, zEQ

    def _compute_x1_equilibrium(self, e_indices, x1eq, zeq, x0_values, model_connectivity):
        self._compute_critical_x0_scaling()
        x0_indices = np.delete(np.array(range(self.number_of_regions)), e_indices)
        x0 = self._compute_model_x0(x0_values, x0_indices)
        if self.x1eq_mode == "linTaylor":
            x1eq = \
                eq_x1_hypo_x0_linTaylor(x0_indices, e_indices, x1eq, zeq, x0, self.K,
                                        model_connectivity, self.yc, self.Iext1, self.a, self.b, self.d)[0]
        else:
            x1eq = \
                eq_x1_hypo_x0_optimize(x0_indices, e_indices, x1eq, zeq, x0, self.K,
                                       model_connectivity, self.yc, self.Iext1, self.a, self.b, self.d)[0]
        return x1eq

    def _normalize_global_coupling(self, K_unscaled):
        self.K = K_unscaled / self.number_of_regions

    def _configure_model_from_equilibrium(self, x1eq, zeq, model_connectivity):
        # x1eq, zeq = self._ensure_equilibrum(x1eq, zeq) # We don't this by default anymore
        x0, Ceq, x0_values, e_values = self._compute_params_after_equilibration(x1eq, zeq, model_connectivity)
        self.x0 = x0
        model = self.model
        del model["model_name"]
        model_config = ModelConfiguration(self.model_name, model_connectivity, self.coupling,
                                          self.monitor, self.initial_conditions, self.noise,
                                          x0_values, e_values, x1eq, zeq, Ceq, **model)
        if model_config.initial_conditions is None:
            model_config.initial_conditions = compute_initial_conditions_from_eq_point(model_config)
        return model_config

    def build_model_from_E_hypothesis(self, disease_hypothesis):

        # This function sets healthy regions to the default epileptogenicity.

        model_connectivity = np.array(self.connectivity)

        # Then apply connectivity disease hypothesis scaling if any:
        if len(disease_hypothesis.w_indices) > 0:
            model_connectivity *= disease_hypothesis.connectivity_disease

        # All nodes except for the diseased ones will get the default epileptogenicity:
        e_values = np.array(self.e_values)
        e_values[disease_hypothesis.e_indices] = disease_hypothesis.e_values

        # Compute equilibrium from epileptogenicity:
        x1eq, zeq = self._compute_x1_and_z_equilibrium_from_E(e_values)

        if len(disease_hypothesis.x0_values) > 0:

            # If there is also some x0 hypothesis, solve the system for the equilibrium:
            # x0_values values must have size of len(x0_indices),
            # e_indices are all regions except for the x0_indices in this case
            x1eq = self._compute_x1_equilibrium(np.delete(range(self.number_of_regions), disease_hypothesis.x0_indices),
                                                x1eq, zeq, disease_hypothesis.x0_values, model_connectivity)
            zeq = self._compute_z_equilibrium(x1eq)

        return self._configure_model_from_equilibrium(x1eq, zeq, model_connectivity)

    def build_model_from_hypothesis(self, disease_hypothesis):
        # This function sets healthy regions to the default excitability.

        model_connectivity = np.array(self.connectivity)

        # Then apply connectivity disease hypothesis scaling if any:
        if len(disease_hypothesis.w_indices) > 0:
            model_connectivity *= disease_hypothesis.connectivity_disease

        # We assume that all nodes have the default (healthy) excitability:
        x0_values = np.array(self.x0_values)
        # ...and some  excitability-diseased ones:
        x0_values[disease_hypothesis.x0_indices] = disease_hypothesis.x0_values
        # x0_values values must have size of len(x0_indices):
        x0_values = np.delete(x0_values, disease_hypothesis.e_indices)

        # There might be some epileptogenicity-diseased regions as well:
        # Initialize with the default e_values
        e_values = np.array(self.e_values)
        # and assign any diseased E_values if any
        e_values[disease_hypothesis.e_indices] = disease_hypothesis.e_values

        # Compute equilibrium only from epileptogenicity:
        x1eq, zeq = self._compute_x1_and_z_equilibrium_from_E(e_values)

        # Now, solve the system in order to compute equilibrium:
        x1eq = self._compute_x1_equilibrium(disease_hypothesis.e_indices, x1eq, zeq, x0_values,
                                            model_connectivity)
        zeq = self._compute_z_equilibrium(x1eq)

        return self._configure_model_from_equilibrium(x1eq, zeq, model_connectivity)

    # TODO: This is used from PSE for varying an attribute's value. We should find a better way, not hardcoded strings.
    def set_attributes_from_pse(self, values, paths, indices):
        for i, val in enumerate(paths):
            vals = val.split(".")
            if vals[0] == "model_configuration_builder":
                if vals[1] == "K_unscaled":
                    temp = self.K_unscaled
                    temp[indices[i]] = values[i]
                    self.set_K_unscaled(temp)
                else:
                    getattr(self, vals[1])[indices[i]] = values[i]



