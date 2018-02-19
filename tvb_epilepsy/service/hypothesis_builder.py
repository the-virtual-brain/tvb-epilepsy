import numpy
from tvb_epilepsy.base.constants.configurations import IN_HEAD
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
    logger = initialize_logger(__name__)

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

    # Flags specific to the process of choosing values for a DiseaseHypothesis
    normalize = False
    sort_disease_values = False

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
        self.normalize = value
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

    def _build_hypothesis_with_all_attributes(self):
        return DiseaseHypothesis(self.nr_of_regions, excitability_hypothesis={tuple(self.x0_indices): self.x0_values},
                                 epileptogenicity_hypothesis={tuple(self.e_indices): self.e_values},
                                 connectivity_hypothesis={tuple(self.w_indices): self.w_values},
                                 lsa_propagation_indices=self.lsa_propagation_indices,
                                 lsa_propagation_strenghts=self.lsa_propagation_strengths, name=self.name)

    def build_epileptogenicity_hypothesis(self, values=None, indices=None):
        if values is None or indices is None:
            hypo = self._build_hypothesis_with_all_attributes()
            self.logger.warning(
                "Since values or indices are None, the DiseaseHypothesis will be build with default values: %s", hypo)

            return hypo

        return DiseaseHypothesis(number_of_regions=self.nr_of_regions,
                                 epileptogenicity_hypothesis={tuple(indices): values})

    def build_excitability_hypothesis(self, values=None, indices=None):
        if values is None or indices is None:
            hypo = self._build_hypothesis_with_all_attributes()
            self.logger.warning(
                "Since values or indices are None, the DiseaseHypothesis will be build with default values: %s", hypo)

            return hypo

        return DiseaseHypothesis(number_of_regions=self.nr_of_regions, excitability_hypothesis={tuple(indices): values})

    def build_mixed_hypothesis(self, ep_values=None, ep_indices=None, exc_values=None, exc_indices=None):
        if ep_values is None or exc_indices is None or ep_values is None or exc_indices is None:
            hypo = self._build_hypothesis_with_all_attributes()
            self.logger.warning(
                "Since values or indices are None, the DiseaseHypothesis will be build with default values: %s", hypo)

            return hypo

        return DiseaseHypothesis(number_of_regions=self.nr_of_regions,
                                 epileptogenicity_hypothesis={tuple(ep_indices): ep_values},
                                 excitability_hypothesis={tuple(exc_indices): exc_values}, name=self.name)

    def _normalize_disease_values(self, values):
        # TODO: something smarter to normalize better disease values
        values += (0.95 - numpy.max(values))

        return values

    def _ensure_normalization_or_sorting(self, disease_values, disease_indices):
        if self.normalize:
            disease_values = self._normalize_disease_values(disease_values)

        if self.sort_disease_values:
            inds = numpy.argsort(disease_values)
            disease_values = disease_values[inds]
            disease_indices = disease_indices[inds]

        return disease_values, disease_indices

    # deprecated
    def build_epileptogenicity_hypothesis_based_on_threshold(self, values, threshold):
        disease_indices, = numpy.where(values > threshold)
        disease_values = values[disease_indices]
        disease_values, disease_indices = self._ensure_normalization_or_sorting(disease_values, disease_indices)

        return self.build_epileptogenicity_hypothesis(disease_values, list(disease_indices))

    # deprecated
    def build_excitability_hypothesis_based_on_threshold(self, values, threshold):
        disease_indices, = numpy.where(values > threshold)
        disease_values = values[disease_indices]
        disease_values, disease_indices = self._ensure_normalization_or_sorting(disease_values, disease_indices)

        return self.build_excitability_hypothesis(disease_values, list(disease_indices))

    # deprecated
    def _compute_e_x0_values_based_on_threshold(self, values, threshold):
        disease_indices, = numpy.where(values > threshold)
        disease_values = values[disease_indices]
        if disease_values.size > 1:
            inds_split = numpy.ceil(disease_values.size * 1.0 / 2).astype("int")
            x0_indices = disease_indices[:inds_split].tolist()
            e_indices = disease_indices[inds_split:].tolist()
            x0_values = disease_values[:inds_split].tolist()
            e_values = disease_values[inds_split:].tolist()
        else:
            x0_indices = disease_indices.tolist()
            x0_values = disease_values.tolist()
            e_indices = []
            e_values = []

        return e_indices, e_values, x0_indices, x0_values

    # deprecated
    def build_mixed_hypothesis_based_on_threshold(self, values, threshold):
        e_indices, e_values, x0_indices, x0_values = self._compute_e_x0_values_based_on_threshold(values, threshold)
        return self.build_mixed_hypothesis(e_values, e_indices, x0_values, x0_indices)

    # deprecated
    def build_mixed_hypothesis_with_x0_having_max_values(self, values, threshold):
        disease_indices, = numpy.where(values > threshold)
        disease_values = values[disease_indices]
        disease_values, disease_indices = self._ensure_normalization_or_sorting(disease_values, disease_indices)

        x0_indices = [disease_indices[-1]]
        x0_values = [disease_values[-1]]
        e_indices = disease_indices[0:-1].tolist()
        e_values = disease_values[0:-1].tolist()

        return self.build_mixed_hypothesis(e_values, e_indices, x0_values, x0_indices)

    def build_lsa_hypothesis(self):
        return self._build_hypothesis_with_all_attributes()

    def build_hypothesis_from_file(self, file, ep_indices=None):
        epi_values = H5Reader().read_epileptogenicity(IN_HEAD, name=file)
        disease_indices, = numpy.where(epi_values > 0)
        disease_values = epi_values[disease_indices]
        disease_values, disease_indices = self._ensure_normalization_or_sorting(disease_values, disease_indices)

        if not ep_indices:
            self.logger.info("An excitability hypothesis will be created with values: %s on indices: %s",
                             disease_values, disease_indices)
            return self.build_excitability_hypothesis(disease_values, disease_indices)

        if set(disease_indices) == set(ep_indices):
            self.logger.info("An epileptogenicity hypothesis will be created with values: %s on indices: %s",
                             disease_values, disease_indices)
            return self.build_epileptogenicity_hypothesis(disease_values, disease_indices)

        ep_values = epi_values[ep_indices]
        exc_indices = numpy.setdiff1d(disease_indices, ep_indices)
        exc_values = epi_values[exc_indices]
        self.logger.info(
            "A mixed hypothesis will be created with x0 values: %s on x0 indices: %s and ep values: %s on ep indices: %s",
            exc_values, exc_indices, ep_values, ep_indices)
        return self.build_mixed_hypothesis(ep_values, ep_indices, exc_values, exc_indices)
