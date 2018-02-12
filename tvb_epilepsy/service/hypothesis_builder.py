import numpy
from tvb_epilepsy.base.model.disease_hypothesis import DiseaseHypothesis


class HypothesisBuilder(object):
    """
    Builder that will create instances of DiseaseHypothesis type in different ways.
    Building a DiseaseHypothesis object is based on the user preferences/choices.
    The most popular ways to define a hypothesis are implemented bellow.

    Attributes that can be configured are listed below, as class attributes.
    """
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
    normalize = False
    sort_disease_values = False

    def set_nr_of_regions(self, nr_of_regions):
        self.nr_of_regions = nr_of_regions
        return self

    def set_normalize(self, value):
        self.normalize = value
        return self

    def set_sort_disease_values(self, value):
        self.sort_disease_values = value
        return self

    def build_epileptogenicity_hypothesis(self, values, indices):
        return DiseaseHypothesis(number_of_regions=self.nr_of_regions,
                                 epileptogenicity_hypothesis={tuple(indices): values})

    def build_excitability_hypothesis(self, values, indices):
        return DiseaseHypothesis(number_of_regions=self.nr_of_regions, excitability_hypothesis={tuple(indices): values})

    def build_mixed_hypothesis(self, ep_values, ep_indices, exc_values, exc_indices):
        return DiseaseHypothesis(number_of_regions=self.nr_of_regions,
                                 epileptogenicity_hypothesis={tuple(ep_indices): ep_values},
                                 excitability_hypothesis={tuple(exc_indices): exc_values})

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

    def build_epileptogenicity_hypothesis_based_on_threshold(self, values, threshold):
        disease_indices, = numpy.where(values > threshold)
        disease_values = values[disease_indices]
        disease_values, disease_indices = self._ensure_normalization_or_sorting(disease_values, disease_indices)

        return self.build_epileptogenicity_hypothesis(disease_values, list(disease_indices))

    def build_excitability_hypothesis_based_on_threshold(self, values, threshold):
        disease_indices, = numpy.where(values > threshold)
        disease_values = values[disease_indices]
        disease_values, disease_indices = self._ensure_normalization_or_sorting(disease_values, disease_indices)

        return self.build_excitability_hypothesis(disease_values, list(disease_indices))

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

    def build_mixed_hypothesis_based_on_threshold(self, values, threshold):
        e_indices, e_values, x0_indices, x0_values = self._compute_e_x0_values_based_on_threshold(values, threshold)
        return self.build_mixed_hypothesis(e_values, e_indices, x0_values, x0_indices)
