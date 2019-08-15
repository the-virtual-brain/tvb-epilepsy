# coding=utf-8
"""
Class for defining and storing the state of a hypothesis.

NOTES:
For the moment a hypothesis concerns the excitability and/or epileptogenicity of each brain region, and/or scalings of specific connectivity weights.
TODO: if needed in the future: Generate a richer disease hypothesis as a combination of hypotheses on other parameters.
"""
from copy import deepcopy
import numpy as np

from tvb_utils.log_error_utils import initialize_logger, raise_value_error
from tvb_utils.data_structures_utils import formal_repr, sort_dict, ensure_list, \
    generate_region_labels, dicts_of_lists_to_lists_of_dicts


logger = initialize_logger(__name__)


class DiseaseHypothesis(object):

    def __init__(self, number_of_regions=0, excitability_hypothesis={}, epileptogenicity_hypothesis={},
                 connectivity_hypothesis={}, lsa_propagation_indices=[], lsa_propagation_strenghts=[], name=""):
        self.number_of_regions = number_of_regions
        self.x0_indices, self.x0_values = self._sort_disease_indices_values(excitability_hypothesis)
        self.e_indices, self.e_values = self._sort_disease_indices_values(epileptogenicity_hypothesis)
        self.w_indices, self.w_values = self._sort_disease_indices_values(connectivity_hypothesis)
        self._type = []
        self._update_type()
        if name == "":
            self.name = "_".join(self._type + ["Hypothesis"])
        else:
            self.name = name
        self.lsa_propagation_indices = np.array(lsa_propagation_indices)
        self.lsa_propagation_strengths = np.array(lsa_propagation_strenghts)

    def __repr__(self):
        d = {"01. Name": self.name,
             "02. Type": self.type,
             "03. Number of regions": self.number_of_regions,
             "04. Excitability (x0) disease indices": self.x0_disease_indices,  # x0_indices,
             "05. Excitability (x0) disease values": self.x0_disease_values,  # x0_values,
             "06. Epileptogenicity (E) disease indices": self.e_disease_indices,  # e_indices,
             "07. Epileptogenicity (E) disease values": self.e_disease_values,  # e_values,
             "08. Connectivity (W) disease indices": self.w_indices,
             "09. Connectivity (W) disease values": self.w_values,
             "10. Propagation indices": self.lsa_propagation_indices,
             }
        if len(self.lsa_propagation_indices):
            d.update(
                {"11. Propagation strengths of indices": self.lsa_propagation_strengths[self.lsa_propagation_indices]})
        else:
            d.update({"11. Propagation strengths of indices": self.lsa_propagation_strengths})
        # d.update({"11. Connectivity": str(self.connectivity)})
        return formal_repr(self, sort_dict(d))

    def __str__(self):
        return self.__repr__()

    def set_attribute(self, attr_name, data):
        setattr(self, attr_name, data)

    def _update_type(self):
        self._type = []
        if len(self.x0_indices) > 0:
            self._type.append("x0")
        if len(self.e_indices) > 0:
            self._type.append("e")
        if len(self.w_indices) > 0:
            self._type.append("w")

    @property
    def type(self):
        self._update_type()
        return '_'.join(self._type)

    def _sort_disease_indices_values(self, disease_dict):
        indices = []
        values = []
        for key, value in disease_dict.items():
            value = ensure_list(value)
            key = ensure_list(key)
            n = len(key)
            if n > 0:
                indices += key
                if len(value) == n:
                    values += value
                elif len(value) == 1 and n > 1:
                    values += value * n
                else:
                    raise_value_error("Length of disease indices " + str(n) + " and values " + str(len(value)) +
                                      " do not match!")
        if len(indices) > 0:
            if isinstance(indices[0], tuple):
                arg_sort = np.ravel_multi_index(indices, (self.number_of_regions, self.number_of_regions)).argsort()
            else:
                arg_sort = np.argsort(indices)
            return np.array(indices)[arg_sort].tolist(), np.array(values)[arg_sort]
        else:
            return [], []

    # This return all e_indices and x0_indices
    @property
    def regions_indices(self):
        return np.unique(self.x0_indices + self.e_indices).astype("i").tolist()

    # The following functions return only those e/x0 indices and values that correspond to e/x0_values > 0.0
    @property
    def regions_disease_indices(self):
        return np.where(self.regions_disease > 0.0)[0].tolist()

    @property
    def x0_disease_indices(self):
        return np.intersect1d(self.regions_disease_indices, self.x0_indices).astype("i").tolist()

    @property
    def x0_disease_values(self):
        return self.regions_disease[np.intersect1d(self.regions_disease_indices, self.x0_indices).astype("i")].tolist()

    @property
    def e_disease_indices(self):
        return np.intersect1d(self.regions_disease_indices, self.e_indices).astype("i").tolist()

    @property
    def e_disease_values(self):
        return self.regions_disease[np.intersect1d(self.regions_disease_indices, self.e_indices).astype("i")].tolist()

    @property
    def regions_disease_values(self):
        return self.regions_disease[self.regions_disease_indices]

    @property
    def connectivity_disease_indices(self):
        if len(self.w_indices) > 0:
            return np.array(self.w_indices)[self.w_values != 1.0]
        else:
            return []

    @property
    def connectivity_disease_values(self):
        if len(self.w_values) > 0:
            return self.w_values[self.w_values != 1.0]
        else:
            return np.array([])

    @property
    def all_disease_indices(self):
        return np.unique(
            np.concatenate([self.regions_disease_indices,
                            np.array(self.connectivity_disease_indices).flatten()])).astype("i").tolist()

    @property
    def regions_disease(self):
        # In case we need values for all regions, we can use this and have zeros where values are not defined
        regions_disease = np.zeros(self.number_of_regions)
        regions_disease[self.x0_indices] = self.x0_values
        regions_disease[self.e_indices] = self.e_values
        return regions_disease

    @property
    def connectivity_disease(self):
        # In case we need values for all regions, we can use this and have ones where values are not defined
        connectivity_shape = (self.number_of_regions, self.number_of_regions)
        connectivity_disease = np.ones(connectivity_shape)
        if len(self.connectivity_disease_indices) > 0 and self.connectivity_disease_values.size > 0:
            indices = self.connectivity_disease_indices
            connectivity_disease[indices] = self.connectivity_disease_values
        return connectivity_disease

    @property
    def disease_propagation_strengths(self):
        return self.lsa_propagation_strengths[self.lsa_propagation_indices]

    @property
    def disease_propagation(self):
        return self.lsa_propagation_indices, self.disease_propagation_strengths

    def update(self, name=""):
        self._type = []
        default_name = "Hypothesis"
        self.x0_indices, self.x0_values = self._sort_disease_indices_values({tuple(self.x0_indices): self.x0_values})
        if len(self.x0_indices) > 0:
            self._type.append("Excitability")
            default_name = "x0_" + default_name
        self.e_indices, self.e_values = self._sort_disease_indices_values({tuple(self.e_indices): self.e_values})
        if len(self.e_indices) > 0:
            self._type.append("Epileptogenicity")
            default_name = "e_" + default_name
        self.w_indices, self.w_values = self._sort_disease_indices_values({tuple(self.w_indices): self.w_values})
        if len(self.w_indices) > 0:
            self._type.append("Connectivity")
            default_name = "w_" + default_name
        self._type = '_'.join(self._type)
        if name == "":
            self.name = default_name
        else:
            self.name = name

    def update_for_pse(self, values, paths, indices):
        for i, val in enumerate(paths):
            vals = val.split(".")
            if vals[0] == "hypothesis":
                getattr(self, vals[1])[indices[i]] = values[i]

    def prepare_for_plot(self, connectivity_matrix=None):
        plot_dict_list = []
        width_ratios = []
        if len(self.lsa_propagation_indices) > 0:
            if connectivity_matrix is None:
                width_ratios += [1]
                name = "LSA Propagation Strength"
                names = [name]
                data = [self.lsa_propagation_strengths]
                indices = [self.lsa_propagation_indices]
                plot_types = ["vector"]
            else:
                width_ratios += [1, 2]
                name = "LSA Propagation Strength"
                names = [name, "Afferent connectivity \n from seizuring regions"]
                data = [self.lsa_propagation_strengths, connectivity_matrix]
                indices = [self.lsa_propagation_indices, self.lsa_propagation_indices]
                plot_types = ["vector", "regions2regions"]
            plot_dict_list = dicts_of_lists_to_lists_of_dicts({"name": names, "data": data, "focus_indices": indices,
                                                               "plot_type": plot_types})
        return plot_dict_list

    def string_regions_disease(self, region_labels=[]):
        region_labels = generate_region_labels(self.number_of_regions, region_labels, str=". ")
        disease_values = self.regions_disease
        disease_string = ""
        for iRegion in self.regions_disease_indices:
            if iRegion in self.e_indices:
                hyp_type = "E"
            else:
                hyp_type = "x0"
            disease_string += region_labels[iRegion] + ": " + hyp_type + "=" + str(disease_values[iRegion]) + "\n"
        return disease_string[:-1]

    def string_connectivity_disease(self, region_labels=[]):
        region_labels = generate_region_labels(self.number_of_regions, region_labels, str=". ")
        disease_string = ""
        for w_ind, w_val in zip(self.w_indices, self.w_values):
            disease_string += region_labels[w_ind[0]] + " -> " + region_labels[w_ind[1]] + ": " + str(w_val) + "\n"
        return disease_string[:-1]

    def prepare_hypothesis_for_h5(self):
        e_values = np.zeros(self.number_of_regions)
        x0_values = np.zeros(self.number_of_regions)
        propagation_values = np.zeros(self.number_of_regions)
        w_values = self.w_values

        if len(self.e_indices) > 0 and self.e_values.size > 0:
            e_values[self.e_indices] = self.e_values

        if len(self.x0_indices) > 0 and self.x0_values.size > 0:
            x0_values[self.x0_indices] = self.x0_values

        if len(self.lsa_propagation_indices) > 0 and self.lsa_propagation_strengths.size > 0:
            if self.lsa_propagation_strengths.size == propagation_values.size:
                propagation_values = self.lsa_propagation_strengths
            else:
                propagation_values[self.lsa_propagation_indices] = self.lsa_propagation_strengths

        hypo = deepcopy(self)
        hypo.e_values = e_values
        hypo.x0_values = x0_values
        hypo.w_values = w_values
        hypo.lsa_propagation_strengths = propagation_values

        return hypo

    def simplify_hypothesis_from_h5(self):
        self.e_values = self.e_values[self.e_indices]
        self.x0_values = self.x0_values[self.x0_indices]
        self.lsa_propagation_strengths = self.lsa_propagation_strengths  # [list(self.lsa_propagation_indices)]
        if (self.w_values == 1).all():
            self.w_values = []
            self.w_indices = []
        else:
            self.w_values = self.w_values[self.w_indices]
