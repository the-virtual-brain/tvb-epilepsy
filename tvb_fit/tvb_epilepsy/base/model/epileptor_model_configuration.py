# coding=utf-8

"""
Class to keep the model configuration values.
This will be used to populate a Model instance needed in order to launch a simulation.
"""
import numpy as np
from tvb_fit.tvb_epilepsy.base.constants.model_constants import *
from tvb_fit.base.model.model_configuration import ModelConfiguration as ModelConfigurationBase
from tvb_fit.tvb_epilepsy.base.computation_utils.equilibrium_computation \
    import compute_initial_conditions_from_eq_point
from tvb_fit.tvb_epilepsy.service.simulator.epileptor_model_factory import EPILEPTOR_MODEL_NVARS

from tvb_scripts.utils.data_structures_utils import formal_repr, dicts_of_lists_to_lists_of_dicts


EPILEPTOR_PARAMS = ["x0", "a", "b", "yc", "d", "Iext1", "Iext2", "slope", "s", "gamma", "tau1", "tau0", "tau2",
                    "zmode", "pmode", "K", "Kvf", "Kf"]


class EpileptorModelConfiguration(ModelConfigurationBase):

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

    def __init__(self, model_name="EpileptorDP", connectivity=None, coupling=None, monitor=None, initial_conditions=None,
                 noise=None, x0_values=X0_DEF, e_values=None, x1eq=None, zeq=None, Ceq=None,  **kwargs):
        super(EpileptorModelConfiguration, self).__init__(model_name, connectivity, coupling, monitor,
                                                          initial_conditions, noise)
        self.x0_values = x0_values
        self.e_values = e_values
        self.x1eq = x1eq
        self.zeq = zeq
        self.Ceq = Ceq
        for pname in EPILEPTOR_PARAMS:
            self.set_parameter(pname, kwargs.get(pname, getattr(self, pname)))

    def __repr__(self):
        d = {
            "01. model": self.model,
            "02. number_of_regions": self.number_of_regions,
            "03. Excitability": self.x0_values,
            "04. Epileptor Model Excitability": self.x0,
            "05. Epileptogenicity": self.e_values,
            "06. x1eq": self.x1eq,
            "07. zeq": self.zeq,
            "08. Ceq": self.Ceq,
            "09. connectivity": self.connectivity,
            "10. coupling": self.coupling,
            "11. monitor": self.monitor,
            "12. initial_conditions": self.initial_conditions,
            "13. noise": self.noise,

        }
        return formal_repr(self, d)

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

    def update_initial_conditions(self):
        self.initial_conditions = compute_initial_conditions_from_eq_point(self)
        return self

    def change_model(self, new_model_name):
        self.model_name = new_model_name
        return self.update_initial_conditions()

    def prepare_for_plot(self, x0_indices=[], e_indices=[], disease_indices=[]):
        names = ["Pathological Excitabilities x0_values", "Model Epileptogenicities e_values", "x1 Equilibria",
                 "z Equilibria", "Total afferent coupling \n at equilibrium"]
        data = [self.x0_values, self.e_values, self.x1eq, self.zeq, self.Ceq]
        disease_indices = np.unique(np.concatenate((x0_indices, e_indices, disease_indices), axis=0)).tolist()
        indices = [x0_indices, e_indices, disease_indices, disease_indices, disease_indices]
        plot_types = ["vector", "vector", "vector", "vector", "regions2regions"]
        return dicts_of_lists_to_lists_of_dicts({"name": names, "data": data, "focus_indices": indices,
                                                 "plot_type": plot_types})
