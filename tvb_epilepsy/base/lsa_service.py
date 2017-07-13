# coding=utf-8
"""
Service to do LSA computation.
"""
import numpy
from collections import OrderedDict

from tvb.basic.logger.builder import get_logger
from tvb_epilepsy.base.constants import EIGENVECTORS_NUMBER_SELECTION, WEIGHTED_EIGENVECTOR_SUM
from tvb_epilepsy.base.utils import formal_repr, weighted_vector_sum
from tvb_epilepsy.base.h5_model import object_to_h5_model
from tvb_epilepsy.base.calculations import calc_fz_jac_square_taylor
from tvb_epilepsy.base.utils import curve_elbow_point
from tvb_epilepsy.base.disease_hypothesis import DiseaseHypothesis
from tvb_epilepsy.base.model_configuration import ModelConfiguration

LOG = get_logger(__name__)

# TODO: it might be useful to store eigenvalues and eigenvectors, as well as the parameters of the computation, such as
# eigen_vectors_number and LSAService in a h5 file

# NOTES: currently the disease_hypothesis (after it has configured a model) is needed only for the connectivity weights.
# In the future this could be part of the model configuration. Disease hypothesis should hold only specific hypotheses,
# on the connectivity matrix (changes, lesions, etc)


class LSAService(object):

    def __init__(self, eigen_vectors_number_selection=EIGENVECTORS_NUMBER_SELECTION, eigen_vectors_number=None,
                 weighted_eigenvector_sum=WEIGHTED_EIGENVECTOR_SUM):
        self.eigen_vectors_number_selection = eigen_vectors_number_selection
        self.eigen_values = []
        self.eigen_vectors = []
        self.eigen_vectors_number = eigen_vectors_number
        self.weighted_eigenvector_sum=weighted_eigenvector_sum

    def __repr__(self):
        d = {"01. Eigenvectors' number selection mode": self.eigen_vectors_number_selection,
             "02. Eigenvectors' number": self.eigen_vectors_number_selection,
             "03. Eigen values": self.eigen_values,
             "04. Eigenvectors": self.eigen_vectors,
             "05. Eigenvectors' number": self.eigen_vectors_number,
             "06. Weighted eigenvector's sum flag": str(self.weighted_eigenvector_sum)
             }
        return formal_repr(self, d)

    def __str__(self):
        return self.__repr__()

    def _prepare_for_h5(self):
        h5_model = object_to_h5_model(self)
        h5_model.add_or_update_metadata_attribute("EPI_Type", "HypothesisModel")
        return h5_model

    def write_to_h5(self, folder, filename=""):
        if filename == "":
            filename = self.name + ".h5"
        h5_model = self._prepare_for_h5()
        h5_model.write_to_h5(folder, filename)

    def get_curve_elbow_point(self, values_array):
        return curve_elbow_point(values_array)

    def _ensure_eigen_vectors_number(self, eigen_values, e_values, x0_values, disease_indices):
        if self.eigen_vectors_number is None:
            if self.eigen_vectors_number_selection is "auto_eigenvals":
                self.eigen_vectors_number = self.get_curve_elbow_point(numpy.abs(eigen_values)) + 1

            elif self.eigen_vectors_number_selection is "auto_disease":
                self.eigen_vectors_number = len(disease_indices)

            elif self.eigen_vectors_number_selection is "auto_epileptogenicity":
                self.eigen_vectors_number = self.get_curve_elbow_point(e_values) + 1

            elif self.eigen_vectors_number_selection is "auto_excitability":
                self.eigen_vectors_number = self.get_curve_elbow_point(x0_values) + 1

            else:
                raise ValueError("\n" + self.eigen_vectors_number_selection +
                                 "is not a valid option when for automatic computation of self.eigen_vectors_number")
        else:
            self.eigen_vectors_number_selection = "user_defined"

    def _compute_jacobian(self, model_configuration):
        fz_jacobian = calc_fz_jac_square_taylor(model_configuration.zEQ, model_configuration.yc,
                                                model_configuration.Iext1, model_configuration.K,
                                                model_configuration.connectivity_matrix,
                                                model_configuration.a, model_configuration.b)

        if numpy.any([numpy.any(numpy.isnan(fz_jacobian.flatten())), numpy.any(numpy.isinf(fz_jacobian.flatten()))]):
            raise ValueError("nan or inf values in dfz")

        return fz_jacobian

    def run_lsa(self, disease_hypothesis, model_configuration):

        jacobian = self._compute_jacobian(model_configuration)

        # Perform eigenvalue decomposition
        eigen_values, eigen_vectors = numpy.linalg.eig(jacobian)

        sorted_indices = numpy.argsort(eigen_values, kind='mergesort')
        self.eigen_values = eigen_values[sorted_indices]
        self.eigen_vectors = eigen_vectors[:, sorted_indices]

        self._ensure_eigen_vectors_number(self.eigen_values, model_configuration.e_values,
                                          model_configuration.x0_values, disease_hypothesis.get_all_disease_indices())

        if self.eigen_vectors_number == disease_hypothesis.get_number_of_regions():
            # Calculate the propagation strength index by summing all eigenvectors
            lsa_propagation_strength = numpy.abs(numpy.sum(self.eigen_vectors, axis=1))
            lsa_propagation_strength /= numpy.max(lsa_propagation_strength)

        else:
            sorted_indices = max(self.eigen_vectors_number, 1)
            # Calculate the propagation strength index by summing the first n eigenvectors (minimum 1)
            if self.weighted_eigenvector_sum:
                lsa_propagation_strength = numpy.abs(weighted_vector_sum(self.eigen_values[:sorted_indices],
                                                           self.eigen_vectors[:, :sorted_indices], normalize=True))
            else:
                lsa_propagation_strength = numpy.abs(numpy.sum(self.eigen_vectors[:, :sorted_indices], axis=1))
            lsa_propagation_strength /= numpy.max(lsa_propagation_strength)


        propagation_strength_elbow = self.get_curve_elbow_point(lsa_propagation_strength)
        propagation_indices = lsa_propagation_strength.argsort()[-propagation_strength_elbow:]

        return DiseaseHypothesis(disease_hypothesis.connectivity,
                                 {tuple(disease_hypothesis.x0_indices): disease_hypothesis.x0_values},
                                 {tuple(disease_hypothesis.e_indices): disease_hypothesis.e_values},
                                 {tuple(disease_hypothesis.w_indices): disease_hypothesis.w_values},
                                 propagation_indices, lsa_propagation_strength, "LSA_" + disease_hypothesis.name)
