# coding=utf-8

import numpy
from tvb_epilepsy.base.constants.config import Config
from tvb_epilepsy.base.model.disease_hypothesis import DiseaseHypothesis
from tvb_epilepsy.base.utils.log_error_utils import initialize_logger, raise_value_error
from tvb_epilepsy.io.h5_reader import H5Reader


class HypothesisBuilder(object):
    """
    Builder that will create instances of DiseaseHypothesis type in different ways.
    Building a DiseaseHypothesis object is based on the user preferences/choices.
    The most popular ways to define a hypothesis are implemented bellow.

    Attributes that can be configured are listed below, as class attributes.
    """

    # Attributes specific to a DiseaseHypothesis
    nr_of_regions = 0
    diseased_regions_values = numpy.zeros((nr_of_regions, ))
    name = ""
    type = []
    e_indices = []
    w_indices = []
    w_values = []
    lsa_propagation_indices = []
    lsa_propagation_strengths = []

    normalize_value = 0.95

    def __init__(self, number_of_regions, config=Config()):
        self.config = config
        self.logger = initialize_logger(__name__, config.out.FOLDER_LOGS)
        self.nr_of_regions = number_of_regions
        self.diseased_regions_values = numpy.zeros((self.nr_of_regions, ))

    def set_nr_of_regions(self, nr_of_regions):
        self.nr_of_regions = nr_of_regions
        return self

    def set_diseased_regions_values(self, disease_values):
        n = len(disease_values)
        if n != self.nr_of_regions:
            raise_value_error("Diseased region values size (" + str(n) +
                              ") doesn't match the number of regions (" + str(self.nr_of_regions) + ")!")
        self.diseased_regions_values = disease_values
        return self

    def set_name(self, name):
        self.name = name
        return self

    def _check_region_inds_range(self, indices, type):
        if numpy.array(indices) < 0 or  numpy.array(indices) >= self.nr_of_regions:
            raise_value_error(type + "_indices out of range! " +
                              "\nnumber of brain regions = " + str(self.nr_of_regions) +
                              "\n" + type + "_indices = " + str(indices))

    def set_x0_hypothesis(self, x0_indices, x0_values):
        self._check_region_inds_range(x0_indices, "x0")
        overwritting = numpy.where(self.diseased_regions_values[x0_indices] > 0)[0].tolist()
        for ind in overwritting:
            self.logger.warning("Overwritting diseased region " + ind  + " with excitability hypothesis " + str())
        return self

    def set_e_hypothesis(self, e_indices, e_values):
        self._check_region_inds_range(e_indices, "e")
        return self

    def set_e_indices(self, e_indices):
        self._check_region_inds_range(e_indices, "e")
        return self

    def set_w_hypothesis(self, w_indices, w_values):
        self._check_set_inds_vals(w_indices, w_values, "w")
        return self

    def set_lsa_propagation_indices(self, lsa_propagation_indices):
        self.lsa_propagation_indices = lsa_propagation_indices
        return self

    def set_lsa_propagation_strengths(self, lsa_propagation_strengths):
        self.lsa_propagation_strengths = lsa_propagation_strengths
        return self

    def set_normalize(self, value):
        self.normalize_value = value
        return self

    def set_sort_disease_values(self, value):
        self.sort_disease_values = value
        return self

    def set_attributes_based_on_hypothesis(self, disease_hypothesis):
        self.set_nr_of_regions(disease_hypothesis.number_of_regions). \
                set_e_hypothesis(disease_hypothesis.e_indices, disease_hypothesis.e_values). \
                    set_x0_hypothesis(disease_hypothesis.x0_indices, disease_hypothesis.x0_values). \
                        set_w_hypothesis(disease_hypothesis.w_indices, disease_hypothesis.w_values). \
                            set_name(disease_hypothesis.name + "LSA")
        return self

    def _normalize_disease_values(self):
        disease_values = self.diseased_regions_values[self.diseased_regions_values > 0]
        if self.normalize_value:
            disease_values += (self.normalize_value - numpy.max(disease_values))
        self.diseased_regions_values[self.diseased_regions_values > 0] = disease_values
        return self

    def build_lsa_hypothesis(self):
        return self.build_hypothesis()

    def build_hypothesis(self):
        self._normalize_disease_values()
        disease_indices, = numpy.where(self.diseased_regions_values > 0)
        exc_indices = numpy.setdiff1d(disease_indices, self.e_indices)

        return DiseaseHypothesis(self.nr_of_regions,
                                 excitability_hypothesis={tuple(exc_indices):
                                                                        self.diseased_regions_values[exc_indices]},
                                 epileptogenicity_hypothesis={tuple(self.e_indices):
                                                                        self.diseased_regions_values[self.e_indices]},
                                 connectivity_hypothesis={tuple(self.w_indices): self.w_values},
                                 lsa_propagation_indices=self.lsa_propagation_indices,
                                 lsa_propagation_strenghts=self.lsa_propagation_strengths, name=self.name)


    def build_hypothesis_from_file(self, hyp_file):
        self.set_diseased_regions_values(H5Reader().read_epileptogenicity(self.config.input.HEAD, name=hyp_file))
        return self.build_hypothesis()

