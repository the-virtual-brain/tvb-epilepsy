# coding=utf-8
"""
Class to keep the model configuration values.
This will be used to populate a Model instance needed in order to launch a simulation.
"""

import numpy as np

from tvb_epilepsy.base.constants import FOLDER_FIGURES, VERY_LARGE_SIZE, FIG_FORMAT, SAVE_FLAG, SHOW_FLAG
from tvb_epilepsy.base.constants import X0_DEF, K_DEF, YC_DEF, I_EXT1_DEF, A_DEF, B_DEF
from tvb_epilepsy.base.h5_model import convert_to_h5_model
from tvb_epilepsy.base.plot_utils import plot_in_columns
from tvb_epilepsy.base.utils import formal_repr, dicts_of_lists_to_lists_of_dicts


class ModelConfiguration(object):
    def __init__(self, yc=YC_DEF, Iext1=I_EXT1_DEF, K=K_DEF, a=A_DEF, b=B_DEF,
                 x0cr=None, rx0=None, x1EQ=None, zEQ=None, Ceq=None, x0_values=X0_DEF, e_values=None,
                 connectivity_matrix=None):

        # These parameters are used for every Epileptor Model...
        self.x0_values = x0_values
        self.yc = yc
        self.Iext1 = Iext1
        self.K = K
        # ...but these 2 have different values for models with more than 2 dimensions
        self.a = a
        self.b = b

        # These parameters are used only for EpileptorDP2D Model
        self.x0cr = x0cr
        self.rx0 = rx0

        # These parameters are not used for Epileptor Model, but are important to keep (h5 or plotting)
        self.x1EQ = x1EQ
        self.zEQ = zEQ
        self.Ceq = Ceq
        self.e_values = e_values

        self.connectivity_matrix = connectivity_matrix

    def __repr__(self):
        d = {
            "01. Excitability": self.x0_values,
            "02. yc": self.yc,
            "03. Iext1": self.Iext1,
            "04. K": self.K,
            "05. a": self.a,
            "06. b": self.b,
            "07. x0cr": self.x0cr,
            "08. rx0": self.rx0,
            "09. x1EQ": self.x1EQ,
            "10. zEQ": self.zEQ,
            "11. Ceq": self.Ceq,
            "12. Epileptogenicity": self.e_values
        }
        return formal_repr(self, d)

    def __str__(self):
        return self.__repr__()

    def _prepare_for_h5(self):
        h5_model = convert_to_h5_model(self)
        h5_model.add_or_update_metadata_attribute("EPI_Type", "HypothesisModel")
        h5_model.add_or_update_metadata_attribute("Number_of_nodes", len(self.x0_values))

        return h5_model

    def write_to_h5(self, folder, filename=""):
        if filename == "":
            filename = self.name + ".h5"
        h5_model = self._prepare_for_h5()
        h5_model.write_to_h5(folder, filename)

    def prepare_for_plot(self, x0_indices=[], e_indices=[], disease_indices=[]):

        names = ["Excitabilities x0", "Epileptogenicities x0", "x1 Equilibria", "z Equilibria",
                 "Total afferent coupling \n at equilibrium"]

        data = [self.x0_values, self.e_values, self.x1EQ, self.zEQ, self.Ceq]

        disease_indices = np.unique(np.concatenate((x0_indices, e_indices, disease_indices), axis=0)).tolist()
        indices = [x0_indices, e_indices, disease_indices, disease_indices, disease_indices]
        plot_types = ["vector", "vector", "vector", "vector", "regions2regions"]

        return dicts_of_lists_to_lists_of_dicts({"name": names, "data": data, "focus_indices": indices,
                                                 "plot_type": plot_types})

    def plot(self, n_regions=None, region_labels=[], x0_indices=[], e_indices=[], disease_indices=[],
             title="Model Configuration Overview", figure_name='', show_flag=SHOW_FLAG, save_flag=SAVE_FLAG,
             figure_dir=FOLDER_FIGURES, figure_format=FIG_FORMAT, figsize=VERY_LARGE_SIZE):

        if n_regions == None:
            n_regions = len(self.x0_values)

        if region_labels == []:
            regions_labels = np.array([str(ii) for ii in range(n_regions)])

        disease_indices = np.unique(np.concatenate((x0_indices, e_indices, disease_indices), axis=0)).tolist()

        plot_dict_list = self.prepare_for_plot(x0_indices, e_indices, disease_indices)

        return plot_in_columns(plot_dict_list, region_labels, width_ratios=[], left_ax_focus_indices=disease_indices,
                               right_ax_focus_indices=disease_indices, title=title, figure_name=figure_name,
                               show_flag=show_flag, save_flag=save_flag, figure_dir=figure_dir,
                               figure_format=figure_format, figsize=figsize)
