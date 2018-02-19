# coding=utf-8

import numpy
from tvb_epilepsy.base.constants.config import Config
from tvb_epilepsy.base.model.disease_hypothesis import DiseaseHypothesis
from tvb_epilepsy.base.utils.log_error_utils import initialize_logger
from tvb_epilepsy.io.h5_reader import H5Reader


class HypothesisBuilder(object):
    """
    Builder that will create instances of DiseaseHypothesis type in different ways.
    Building a DiseaseHypothesis object is based on the user preferences/choices.
    The most popular ways to define a hypothesis are implemented bellow.

    Attributes that can be configured are listed below, as class attributes.
    """
    # config = Config()
    # logger = initialize_logger(__name__, config.out.FOLDER_LOGS)

    # Attributes specific to a DiseaseHypothesis
    nr_of_regions = 0
    name = ""
    type = []
    x0_indices = []
    x0_values = []
    e_indices = []
    e_values = []
    w_indices = []
    w_values = []
    lsa_propagation_indices = []
    lsa_propagation_strengths = []

    normalize_value = 0.95
    sort_disease_values = False

    def __init__(self, config=Config()):
        self.config = config
        self.logger = initialize_logger(__name__, config.out.FOLDER_LOGS)

    def set_nr_of_regions(self, nr_of_regions):
        self.nr_of_regions = nr_of_regions
        return self

    def set_name(self, name):
        self.name = name
        return self

    def set_x0_indices(self, x0_indices):
        self.x0_indices = x0_indices
        return self

    def set_x0_values(self, x0_values):
        self.x0_values = x0_values
        return self

    def set_e_indices(self, e_indices):
        self.e_indices = e_indices
        return self

    def set_e_values(self, e_values):
        self.e_values = e_values
        return self

    def set_w_indices(self, w_indices):
        self.w_indices = w_indices
        return self

    def set_w_values(self, w_values):
        self.w_values = w_values
        return self

    def set_lsa_propagation_indices(self, lsa_propagtion_indices):
        self.lsa_propagation_indices = lsa_propagtion_indices
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
        self.set_nr_of_regions(disease_hypothesis.number_of_regions).set_e_indices(
            disease_hypothesis.e_indices).set_e_values(disease_hypothesis.e_values).set_x0_indices(
            disease_hypothesis.x0_indices).set_x0_values(disease_hypothesis.x0_values).set_w_indices(
            disease_hypothesis.w_indices).set_w_values(disease_hypothesis.w_values).set_name(
            disease_hypothesis.name + "LSA")
        return self

    def _build_hypothesis(self):

        return DiseaseHypothesis(self.nr_of_regions, excitability_hypothesis={tuple(self.x0_indices): self.x0_values},
                                 epileptogenicity_hypothesis={tuple(self.e_indices): self.e_values},
                                 connectivity_hypothesis={tuple(self.w_indices): self.w_values},
                                 lsa_propagation_indices=self.lsa_propagation_indices,
                                 lsa_propagation_strenghts=self.lsa_propagation_strengths, name=self.name)

    def _build_epileptogenicity_hypothesis(self, values=None, indices=None):
        if values is None or indices is None:
            hypo = self._build_hypothesis()
            self.logger.warning(
                "Since values or indices are None, the DiseaseHypothesis will be build with default values: %s", hypo)

            return hypo

        return DiseaseHypothesis(number_of_regions=self.nr_of_regions,
                                 epileptogenicity_hypothesis={tuple(indices): values}, name=self.name)

    def _build_excitability_hypothesis(self, values=None, indices=None):
        if values is None or indices is None:
            hypo = self._build_hypothesis()
            self.logger.warning(
                "Since values or indices are None, the DiseaseHypothesis will be build with default values: %s", hypo)

            return hypo

        return DiseaseHypothesis(number_of_regions=self.nr_of_regions, excitability_hypothesis={tuple(indices): values},
                                 name=self.name)

    def _build_mixed_hypothesis(self, e_values=None, e_indices=None, exc_values=None, exc_indices=None):
        if e_values is None or exc_indices is None or e_values is None or exc_indices is None:
            hypo = self._build_hypothesis()
            self.logger.warning(
                "Since values or indices are None, the DiseaseHypothesis will be build with default values: %s", hypo)

            return hypo

        return DiseaseHypothesis(number_of_regions=self.nr_of_regions,
                                 epileptogenicity_hypothesis={tuple(e_indices): e_values},
                                 excitability_hypothesis={tuple(exc_indices): exc_values}, name=self.name)

    def _normalize_disease_values(self, values):
        # TODO: something smarter to normalize better disease values
        values += (self.normalize_value - numpy.max(values))

        return values

    def _ensure_normalization_or_sorting(self, disease_values, disease_indices):
        disease_values = self._normalize_disease_values(disease_values)

        if self.sort_disease_values:
            inds = numpy.argsort(disease_values)
            disease_values = disease_values[inds]
            disease_indices = disease_indices[inds]

        return disease_values, disease_indices

    def build_lsa_hypothesis(self):
        return self._build_hypothesis()

    def build_hypothesis(self, epi_values, e_indices=None):
        disease_indices, = numpy.where(epi_values > 0)
        disease_values = epi_values[disease_indices]
        disease_values, disease_indices = self._ensure_normalization_or_sorting(disease_values, disease_indices)

        if not e_indices:
            self.logger.info("An excitability hypothesis will be created with values: %s on indices: %s",
                             disease_values, disease_indices)
            return self._build_excitability_hypothesis(disease_values, disease_indices)

        if set(disease_indices) == set(e_indices):
            self.logger.info("An epileptogenicity hypothesis will be created with values: %s on indices: %s",
                             disease_values, disease_indices)
            return self._build_epileptogenicity_hypothesis(disease_values, disease_indices)

        e_values = epi_values[e_indices]
        exc_indices = numpy.setdiff1d(disease_indices, e_indices)
        exc_values = epi_values[exc_indices]
        self.logger.info("A mixed hypothesis will be created with x0 values: %s on x0 indices: %s "
                         "and ep values: %s on ep indices: %s", exc_values, exc_indices, e_values, e_indices)
        return self._build_mixed_hypothesis(e_values, e_indices, exc_values, exc_indices)

    def build_hypothesis_from_manual_input(self, e_values=None, e_indices=None, exc_values=None, exc_indices=None):
        epi_values = numpy.zeros((self.nr_of_regions,))
        if e_indices is not None and e_values is not None:
            e_indices = list(e_indices)
            epi_values[e_indices] = e_values
        else:
            e_indices = []
        if exc_indices is not None and exc_values is not None:
            epi_values[exc_indices] = exc_values
            common_indices = list(numpy.intersect1d(e_indices, exc_indices))
            if len(common_indices) > 0:
                e_indices = numpy.setdiff1d(e_indices, common_indices)
                self.logger.warning("Overwriting e_indices that are common to exc_indices!: " + str(common_indices))
        if len(e_indices) == 0:
            e_indices = None
        return self.build_hypothesis(epi_values, e_indices)

    def build_hypothesis_from_file(self, hyp_file, e_indices=None):
        epi_values = H5Reader().read_epileptogenicity(self.config.input.HEAD, name=hyp_file)
        return self.build_hypothesis(epi_values, e_indices)

