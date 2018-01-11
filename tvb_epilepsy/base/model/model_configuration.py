# coding=utf-8
"""
Class to keep the model configuration values.
This will be used to populate a Model instance needed in order to launch a simulation.
"""

import numpy as np
from tvb_epilepsy.base.constants.model_constants import X0_DEF, K_DEF, YC_DEF, I_EXT1_DEF, I_EXT2_DEF, A_DEF, B_DEF, \
    D_DEF, SLOPE_DEF, S_DEF, GAMMA_DEF
from tvb_epilepsy.base.constants.configurations import FOLDER_FIGURES, VERY_LARGE_SIZE, FIG_FORMAT, SAVE_FLAG, SHOW_FLAG
from tvb_epilepsy.base.utils.data_structures_utils import formal_repr, dicts_of_lists_to_lists_of_dicts
from tvb_epilepsy.base.utils.plot_utils import plot_in_columns


class ModelConfiguration(object):
    def __init__(self, yc=YC_DEF, Iext1=I_EXT1_DEF, Iext2=I_EXT2_DEF, K=K_DEF, a=A_DEF, b=B_DEF, d=D_DEF,
                 slope=SLOPE_DEF, s=S_DEF, gamma=GAMMA_DEF, x1EQ=None, zEQ=None, Ceq=None, x0=None,
                 x0_values=X0_DEF, e_values=None, zmode=np.array("lin"), model_connectivity=None, n_regions=None):
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
        
        # These parameters are used only for EpileptorDP2D Model
        self.zmode = zmode

        # These parameters are not used for Epileptor Model, but are important to keep (h5 or plotting)
        self.x1EQ = x1EQ
        self.zEQ = zEQ
        self.Ceq = Ceq
        self.e_values = e_values
        self.model_connectivity = model_connectivity
        if n_regions is None:
            self.n_regions = model_connectivity.shape[0]
        else:
            self.n_regions = 0

    def __repr__(self):
        d = {
            "00. number of regions": self.n_regions,
            "01. Excitability": self.x0_values,
            "02. Epileptor Model Excitability": self.x0,
            "03. x1EQ": self.x1EQ,
            "04. zEQ": self.zEQ,
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
            "17. zmode": self.zmode,
            "18. Model connectivity": self.model_connectivity
        }
        return formal_repr(self, d)

    def __str__(self):
        return self.__repr__()

    def prepare_for_plot(self, x0_indices=[], e_indices=[], disease_indices=[]):
        names = ["Pathological Excitabilities x0_values", "Model Epileptogenicities e_values", "x1 Equilibria",
                 "z Equilibria", "Total afferent coupling \n at equilibrium"]
        data = [self.x0_values, self.e_values, self.x1EQ, self.zEQ, self.Ceq]
        disease_indices = np.unique(np.concatenate((x0_indices, e_indices, disease_indices), axis=0)).tolist()
        indices = [x0_indices, e_indices, disease_indices, disease_indices, disease_indices]
        plot_types = ["vector", "vector", "vector", "vector", "regions2regions"]
        return dicts_of_lists_to_lists_of_dicts({"name": names, "data": data, "focus_indices": indices,
                                                 "plot_type": plot_types})

    def plot(self, n_regions=None, regions_labels=[], x0_indices=[], e_indices=[], disease_indices=[],
             title="Model Configuration Overview", figure_name='', show_flag=SHOW_FLAG, save_flag=SAVE_FLAG,
             figure_dir=FOLDER_FIGURES, figure_format=FIG_FORMAT, figsize=VERY_LARGE_SIZE):
        if n_regions == None:
            n_regions = len(self.x0_values)
        if regions_labels == []:
            regions_labels = np.array([str(ii) for ii in range(n_regions)])
        disease_indices = np.unique(np.concatenate((x0_indices, e_indices, disease_indices), axis=0)).tolist()
        plot_dict_list = self.prepare_for_plot(x0_indices, e_indices, disease_indices)
        return plot_in_columns(plot_dict_list, regions_labels, width_ratios=[], left_ax_focus_indices=disease_indices,
                               right_ax_focus_indices=disease_indices, title=title, figure_name=figure_name,
                               show_flag=show_flag, save_flag=save_flag, figure_dir=figure_dir,
                               figure_format=figure_format, figsize=figsize)
