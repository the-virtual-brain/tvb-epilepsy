# coding=utf-8
"""
Class for defining and storing the state of a hypothesis.
"""
from collections import OrderedDict

import numpy

from tvb_epilepsy.base.h5_model import prepare_for_h5
from tvb_epilepsy.base.utils import formal_repr


class DiseaseHypothesis(object):
    def __init__(self, type, connectivity, disease_values, x0_indices, e_indices, propagation_indices,
                 propagation_strenghts,
                 name="E_Hypothesis"):
        self.type = type
        self.connectivity = connectivity
        self.disease_values = disease_values
        self.x0_indices = x0_indices
        self.e_indices = e_indices
        self.propagation_indices = propagation_indices
        self.propagation_strenghts = propagation_strenghts
        self.name = name

    def __repr__(self):
        d = {"01. Type": self.type,
             "02. Weights of x0 nodes": self.get_weights()[:, self.x0_indices],
             "03. Weights of e nodes": self.get_weights()[:, self.e_indices],
             "04. X0 disease indices": self.x0_indices,
             "05. E disease indices": self.e_indices,
             "06. Disease values": self.disease_values,
             "07. Propagation indices": self.propagation_indices,
             "08. Propagation strengths of indices": self.propagation_strenghts[self.propagation_indices],
             "09. Name": self.name
             }
        return formal_repr(self, OrderedDict(sorted(d.items(), key=lambda t: t[0])))

    def __str__(self):
        return self.__repr__()

    def prepare_for_h5(self):
        h5_model = prepare_for_h5(self)
        h5_model.add_or_update_metadata_attribute("EPI_Type", "HypothesisModel")
        h5_model.add_or_update_metadata_attribute("Number_of_nodes", self.get_number_of_regions())

        h5_model.add_or_update_datasets_attribute("disease_indices", (self.get_regions_disease() != 0).astype(float))
        h5_model.add_or_update_datasets_attribute("disease_values", self.get_regions_disease())

        all_indices_for_propagation = numpy.zeros(self.get_number_of_regions())
        all_indices_for_propagation[self.propagation_indices] = 1

        h5_model.add_or_update_datasets_attribute("propagation_indices", all_indices_for_propagation)

        return h5_model

    def get_regions_disease(self):
        # In case we need values for all regions, we can use this and have zeros where values are not defined
        regions_disease = numpy.zeros(self.get_number_of_regions())
        regions_disease[sorted(self.x0_indices + self.e_indices)] = self.disease_values

        return regions_disease

    def get_number_of_regions(self):
        return self.connectivity.number_of_regions

    def get_weights(self):
        return self.connectivity.normalized_weights

    def get_region_labels(self):
        return self.connectivity.region_labels

    def get_seizure_indices(self, seizure_threshold):
        seizure_indices, = numpy.where(self.get_regions_disease() > seizure_threshold)
        return seizure_indices
