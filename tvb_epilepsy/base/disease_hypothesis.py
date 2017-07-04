# coding=utf-8
"""
Class for defining and storing the state of a hypothesis.
"""
from collections import OrderedDict

import numpy

from tvb_epilepsy.base.h5_model import prepare_for_h5
from tvb_epilepsy.base.utils import formal_repr, ensure_list, linear_index_to_coordinate_tuples

# NOTES:
#  For the moment a hypothesis concerns the excitability and/or epileptogenicity of each brain region,
#  and/or scalings of specific connectivity weights.
# TODO if needed in the future: Generate a richer disease hypothesis as a combination of hypotheses on other parameters.


class DiseaseHypothesis(object):
    def __init__(self, connectivity, excitability_hypothesis={}, epileptogenicity_hypothesis={},
                 connectivity_hypothesis={}, propagation_indices=[], propagation_strenghts=[], name=""):

        self.connectivity = connectivity

        self.type = []
        self.x0_indices, self.x0_values = self.sort_disease_indices_values(excitability_hypothesis)
        if len(self.x0_indices) > 0:
            self.type.append("Excitability")
        self.e_indices, self.e_values = self.sort_disease_indices_values(epileptogenicity_hypothesis)
        if len(self.e_indices) > 0:
            self.type.append("Epileptogenicity")
        self.w_indices, self.w_values = self.sort_disease_indices_values(connectivity_hypothesis)
        if len(self.w_indices) > 0:
            self.type.append("Connectivity")
        self.type = '_'.join(self.type)
        if name == "":
            self.name = self.type + "_Hypothesis"
        else:
            self.name = name

        self.propagation_indices = propagation_indices
        self.propagation_strenghts = propagation_strenghts

    def __repr__(self):
        d = {"01. Name": self.name,
             "02. Type": self.type,
             "03. X0 disease indices": self.x0_indices,
             "04. X0 disease values": self.x0_values,
             "05. E disease indices": self.e_indices,
             "06. E disease indices": self.e_values,
             "07. Connectivity disease indices":
                 linear_index_to_coordinate_tuples(self.w_indices, self.connectivity.weights.shape),
             "08. Connectivity disease values": self.w_values,
             "09. Propagation indices": self.propagation_indices,
             "10. Propagation strengths of indices": self.propagation_strenghts[self.propagation_indices]
             }
        return formal_repr(self, OrderedDict(sorted(d.items(), key=lambda t: t[0])))

    def __str__(self):
        return self.__repr__()

    def prepare_for_h5(self):
        h5_model = prepare_for_h5(self)
        h5_model.add_or_update_metadata_attribute("EPI_Type", "HypothesisModel")
        h5_model.add_or_update_metadata_attribute("Number_of_nodes", self.get_number_of_regions())

        all_indices_for_propagation = numpy.zeros(self.get_number_of_regions())
        all_indices_for_propagation[self.propagation_indices] = 1

        h5_model.add_or_update_datasets_attribute("propagation_indices", all_indices_for_propagation)

        return h5_model

    def sort_disease_indices_values(self, disease_dict):
        indices = []
        values = []
        for key, value in disease_dict.iteritems():
            key = ensure_list(key)
            value = ensure_list(value)
            n = len(key)
            indices += key
            if len(value) == n:
                values += value
            elif len(value) == 1 and n>1:
                values += value * n
            else:
                raise ValueError("Length of disease indices " + str(len(key)) + " and values " + str(len(value)) +
                                 " do not match!")
        arg_sort = numpy.argsort(indices)
        return numpy.array(indices)[arg_sort].tolist(), numpy.array(values)[arg_sort]

    def update(self, name=""):
        self.type = []
        self.x0_indices, self.x0_values = self.sort_disease_indices_values({tuple(self.x0_indices): self.x0_values})
        if len(self.x0_indices) > 0:
            self.type.append("Excitability")
        self.e_indices, self.e_values = self.sort_disease_indices_values({tuple(self.e_indices): self.e_values})
        if len(self.e_indices) > 0:
            self.type.append("Epileptogenicity")
        self.w_indices, self.w_values = self.sort_disease_indices_values({tuple(self.w_indices): self.w_values})
        if len(self.w_indices) > 0:
            self.type.append("Connectivity")
        self.type = '_'.join(self.type)
        if name == "":
            self.name = self.type + "_Hypothesis"
        else:
            self.name = name

    def get_regions_disease_indices(self):
        return numpy.unique(self.x0_indices + self.e_indices).astype("i").tolist()

    def get_connectivity_disease_indices(self):
        return self.w_indices

    def get_connectivity_regions_disease_indices(self):
        indexes = numpy.unravel_index(self.get_connectivity_disease_indices(),
                                      (self.get_number_of_regions(), self.get_number_of_regions()))
        indexes = numpy.unique(numpy.concatenate(indexes)).astype("i")
        return indexes.tolist()

    def get_all_disease_indices(self):
        return numpy.unique(numpy.concatenate((self.get_regions_disease_indices(),
                                               self.get_connectivity_disease_indices()))).astype("i").tolist()

    def get_regions_disease(self):
        # In case we need values for all regions, we can use this and have zeros where values are not defined
        regions_disease = numpy.zeros(self.get_number_of_regions())
        regions_disease[self.x0_indices] = self.x0_values
        regions_disease[self.e_indices] = self.e_values

        return regions_disease

    def get_connectivity_disease(self):
        # In case we need values for all regions, we can use this and have zeros where values are not defined
        connectivity_shape = (self.get_number_of_regions(), self.get_number_of_regions())
        connectivity_disease = numpy.ones(connectivity_shape)
        indexes = numpy.unravel_index(self.get_connectivity_disease_indices(), connectivity_shape)
        connectivity_disease[indexes[0], indexes[1]] = self.w_values
        connectivity_disease[indexes[1], indexes[0]] = self.w_values

        return connectivity_disease

    def get_number_of_regions(self):
        return self.connectivity.number_of_regions

    def get_weights(self):
        return self.connectivity.normalized_weights

    def get_region_labels(self):
        return self.connectivity.region_labels

    # Do we really need those two?:
    def get_e_values_for_all_regions(self):
        return self.get_regions_disease()[self.e_indices]

    def get_x0_values_for_all_regions(self):
        return self.get_regions_disease()[self.x0_indices]
