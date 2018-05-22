# coding=utf-8

import numpy
from tvb_epilepsy.base.constants.config import Config
from tvb_epilepsy.base.model.disease_hypothesis import DiseaseHypothesis
from tvb_epilepsy.base.utils.log_error_utils import initialize_logger, raise_value_error
from tvb_epilepsy.base.utils.data_structures_utils import ensure_list
from tvb_epilepsy.base.computations.analyzers_utils import interval_scaling
from tvb_epilepsy.io.h5_reader import H5Reader

# TODO: In the future we should allow for various not 0 healthy values.
# In this case x0 could take any value, and only knowing e_indices would make some difference


class HypothesisBuilder(object):
    """
    Builder that will create instances of DiseaseHypothesis type in different ways.
    Building a DiseaseHypothesis object is based on the user preferences/choices.
    The most popular ways to define a hypothesis are implemented bellow.

    Attributes that can be configured are listed below, as class attributes.
    """

    # Attributes specific to a DiseaseHypothesis
    number_of_regions = 0
    diseased_regions_values = numpy.zeros((number_of_regions,))
    name = ""
    type = []
    e_indices = []
    w_indices = []
    w_values = []
    lsa_propagation_indices = []
    lsa_propagation_strengths = []

    normalize_values = ensure_list(0.99)

    def __init__(self, number_of_regions=0, config=Config()):
        self.config = config
        self.logger = initialize_logger(__name__, config.out.FOLDER_LOGS)
        self.number_of_regions = number_of_regions
        self.diseased_regions_values = numpy.zeros((self.number_of_regions,))

    def set_nr_of_regions(self, nr_of_regions):
        self.number_of_regions = nr_of_regions
        self.diseased_regions_values = numpy.zeros((self.number_of_regions,))
        return self

    def set_diseased_regions_values(self, disease_values):
        n = len(disease_values)
        if n != self.number_of_regions:
            raise_value_error("Diseased region values size (" + str(n) +
                              ") doesn't match the number of regions (" + str(self.number_of_regions) + ")!")
        self.diseased_regions_values = disease_values
        return self

    def set_name(self, name):
        self.name = name
        return self

    def _check_regions_inds_range(self, indices, type):
        if numpy.any(numpy.array(indices) < 0) or numpy.any(numpy.array(indices) >= self.number_of_regions):
            raise_value_error(type + "_indices out of range! " +
                              "\nThe maximum indice is " + str(self.number_of_regions)
                              + " for number of brain regions " + str(self.number_of_regions) + " but"
                              "\n" + type + "_indices = " + str(indices))
        return indices

    def _check_indices_vals_sizes(self, indices, values, type):
        n_inds = len(indices)
        n_vals = len(values)
        if n_inds != n_vals:
            if n_vals != 1:
                values *= numpy.ones(numpy.array(indices).shape)
            else:
                raise_value_error("Sizes of " + type + "_indices (" + str(n_inds) + ") " +
                                  "and " + type + "_values (" + str(n_vals) + ") do not match!")
        return values

    def _check_overwritting(self, indices, type):
        for ind in indices:
            if self.diseased_regions_values[ind] > 0.0:
                if ind in self.e_indices:
                    previous_hyp = "epileptogenicity"
                else:
                    previous_hyp = "excitability"
                if type == "e":
                    current_hyp = "epileptogenicity"
                else:
                    current_hyp = "excitability"
                self.logger.warning("Overwritting hypothesis for region " + str(ind)
                                    + " from the previous " + previous_hyp +
                                    " hypothesis to a current " + current_hyp + " one!")

    def set_x0_hypothesis(self, x0_indices, x0_values):
        idx = self._check_regions_inds_range(list(x0_indices), "x0")
        self.diseased_regions_values[idx] = self._check_indices_vals_sizes(idx, list(x0_values), "x0")
        return self

    def set_e_hypothesis(self, e_indices, e_values):
        self.e_indices = self._check_regions_inds_range(list(e_indices), "e")
        self.diseased_regions_values[self.e_indices] = self._check_indices_vals_sizes(self.e_indices,
                                                                                      list(e_values), "e")
        return self

    def set_e_indices(self, e_indices):
        self.e_indices = self._check_regions_inds_range(list(e_indices), "e")
        return self

    def set_w_hypothesis(self, w_indices, w_values):
        w_indices = list(w_indices)
        for ind, w_ind in enumerate(w_indices):
            w_indices[ind] = tuple(w_ind)
        self.w_indices = self._check_regions_inds_range(w_indices, "w")
        self.w_values = self._check_indices_vals_sizes(w_indices, list(w_values), "w")
        return self

    def set_lsa_propagation(self, lsa_propagation_indices, lsa_propagation_strengths):
        self.lsa_propagation_indices = numpy.array(self._check_regions_inds_range(lsa_propagation_indices, "lsa_propagation"))
        # self.lsa_propagation_strengths = numpy.array(
        #                                         self._check_indices_vals_sizes(self.lsa_propagation_indices,
        #                                                                        lsa_propagation_strengths,
        #                                                                        "lsa_propagation"))
        self.lsa_propagation_strengths = numpy.array(lsa_propagation_strengths)
        return self

    def set_normalize(self, values):
        values = ensure_list(values)
        n_vals = len(values)
        if n_vals > 2:
            raise_value_error("Invalid disease hypothesis normalization values!: " + str(values) +
                              "\nThey cannot be more than 2!")
        else:
            if n_vals < 2:
                values = [numpy.min(self.diseased_regions_values)] + values
            self.normalize_values = values
        return self

    def _normalize_disease_values(self):
        disease_values = numpy.array(self.diseased_regions_values)
        self.set_normalize(self.normalize_values)
        self.diseased_regions_values = \
            interval_scaling(disease_values, self.normalize_values[0], self.normalize_values[1])
        return self

    def set_attributes_based_on_hypothesis(self, disease_hypothesis):
        self.set_nr_of_regions(disease_hypothesis.number_of_regions). \
                set_name(disease_hypothesis.name). \
                    set_e_hypothesis(disease_hypothesis.e_indices, disease_hypothesis.e_values). \
                        set_x0_hypothesis(disease_hypothesis.x0_indices, disease_hypothesis.x0_values). \
                            set_w_hypothesis(disease_hypothesis.w_indices, disease_hypothesis.w_values)
        return self

    def build_lsa_hypothesis(self):
        return self.build_hypothesis()

    def build_hypothesis(self):
        if self.normalize_values:
            self._normalize_disease_values()
        disease_indices, = numpy.where(self.diseased_regions_values != 0.0)
        x0_indices = numpy.setdiff1d(disease_indices, self.e_indices)
        return DiseaseHypothesis(self.number_of_regions,
                                 excitability_hypothesis={tuple(x0_indices):
                                                              self.diseased_regions_values[x0_indices]},
                                 epileptogenicity_hypothesis={tuple(self.e_indices):
                                                                  self.diseased_regions_values[self.e_indices]},
                                 connectivity_hypothesis={tuple(self.w_indices): self.w_values},
                                 lsa_propagation_indices=self.lsa_propagation_indices,
                                 lsa_propagation_strenghts=self.lsa_propagation_strengths, name=self.name)

    def build_hypothesis_from_file(self, hyp_file, e_indices=None):
        self.set_diseased_regions_values(H5Reader().read_epileptogenicity(self.config.input.HEAD, name=hyp_file))
        if e_indices:
            self.set_e_indices(e_indices)
        return self.build_hypothesis()
