# coding=utf-8

"""
Class to keep the model configuration values.
This will be used to populate a Model instance needed in order to launch a simulation.
"""

from tvb_fit.tvb_epilepsy.base.constants.model_constants import *
from tvb_fit.base.utils.data_structures_utils import formal_repr, dicts_of_lists_to_lists_of_dicts, ensure_list


class ModelConfiguration(object):
    def __init__(self, yc=YC_DEF, Iext1=I_EXT1_DEF, Iext2=I_EXT2_DEF, K=K_DEF, a=A_DEF, b=B_DEF, d=D_DEF,
                 slope=SLOPE_DEF, s=S_DEF, gamma=GAMMA_DEF, tau1=TAU1_DEF, tau0=TAU0_DEF,
                 x1eq=None, zeq=None, Ceq=None, x0=None, x0_values=X0_DEF, e_values=None,
                 zmode=np.array("lin"), model_connectivity=None, number_of_regions=0):
        # These parameters are used for every Epileptor Model...
        self.x0_values = x0_values
        self.x0 = x0
        self.yc = yc
        self.Iext1 = Iext1
        self.Iext2 = Iext2
        self.K = K
        self.a = a
        self.b = b
        self.d = d
        self.s = s
        self.slope = slope
        self.gamma = gamma
        self.tau1 = tau1
        self.tau0 = tau0
        # These parameters are used only for EpileptorDP2D Model
        self.zmode = zmode

        # These parameters are not used for Epileptor Model, but are important to keep (h5 or plotting)
        self.x1eq = x1eq
        self.zeq = zeq
        self.Ceq = Ceq
        self.e_values = e_values
        self.model_connectivity = model_connectivity
        if number_of_regions == 0 and model_connectivity is not None:
            self.number_of_regions = model_connectivity.shape[0]
        else:
            self.number_of_regions = len(ensure_list(self.x1eq))

    def __repr__(self):
        d = {
            "00. number of regions": self.number_of_regions,
            "01. Excitability": self.x0_values,
            "02. Epileptor Model Excitability": self.x0,
            "03. x1eq": self.x1eq,
            "04. zeq": self.zeq,
            "05. Ceq": self.Ceq,
            "06. Epileptogenicity": self.e_values,
            "07. yc": self.yc,
            "08. Iext1": self.Iext1,
            "09. Iext2": self.Iext2,
            "10. K": self.K,
            "11. a": self.a,
            "12. b": self.b,
            "13. d": self.d,
            "14. s": self.s,
            "15. slope": self.slope,
            "16. gamma": self.gamma,
            "17. tau1": self.tau1,
            "18. tau0": self.tau0,
            "19. zmode": self.zmode,
            "20. Model connectivity": self.model_connectivity
        }
        return formal_repr(self, d)

    def __str__(self):
        return self.__repr__()

    def set_attribute(self, attr_name, data):
        setattr(self, attr_name, data)

    def prepare_for_plot(self, x0_indices=[], e_indices=[], disease_indices=[]):
        names = ["Pathological Excitabilities x0_values", "Model Epileptogenicities e_values", "x1 Equilibria",
                 "z Equilibria", "Total afferent coupling \n at equilibrium"]
        data = [self.x0_values, self.e_values, self.x1eq, self.zeq, self.Ceq]
        disease_indices = np.unique(np.concatenate((x0_indices, e_indices, disease_indices), axis=0)).tolist()
        indices = [x0_indices, e_indices, disease_indices, disease_indices, disease_indices]
        plot_types = ["vector", "vector", "vector", "vector", "regions2regions"]
        return dicts_of_lists_to_lists_of_dicts({"name": names, "data": data, "focus_indices": indices,
                                                 "plot_type": plot_types})
