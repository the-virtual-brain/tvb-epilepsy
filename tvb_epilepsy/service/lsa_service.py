# coding=utf-8
"""
Service to do LSA computation.

TODO: it might be useful to store eigenvalues and eigenvectors, as well as the parameters of the computation, such as eigen_vectors_number and LSAService in a h5 file
"""
import numpy
from tvb_epilepsy.base.constants.module_constants import EIGENVECTORS_NUMBER_SELECTION, WEIGHTED_EIGENVECTOR_SUM
from tvb_epilepsy.base.constants.model_constants import X1_EQ_CR_DEF
from tvb_epilepsy.base.utils.log_error_utils import initialize_logger, raise_value_error
from tvb_epilepsy.base.utils.data_structures_utils import formal_repr
from tvb_epilepsy.base.computations.calculations_utils import calc_fz_jac_square_taylor
from tvb_epilepsy.base.computations.equilibrium_computation import calc_eq_z
from tvb_epilepsy.base.utils.math_utils import weighted_vector_sum, curve_elbow_point
from tvb_epilepsy.service.hypothesis_builder import HypothesisBuilder


class LSAService(object):
    logger = initialize_logger(__name__)

    def __init__(self, eigen_vectors_number_selection=EIGENVECTORS_NUMBER_SELECTION, eigen_vectors_number=None,
                 weighted_eigenvector_sum=WEIGHTED_EIGENVECTOR_SUM, normalize_propagation_strength=False):
        self.eigen_vectors_number_selection = eigen_vectors_number_selection
        self.eigen_values = []
        self.eigen_vectors = []
        self.eigen_vectors_number = eigen_vectors_number
        self.weighted_eigenvector_sum = weighted_eigenvector_sum
        self.normalize_propagation_strength = normalize_propagation_strength

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

    def set_attribute(self, attr_name, data):
        setattr(self, attr_name, data)

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
                raise_value_error("\n" + self.eigen_vectors_number_selection +
                                  "is not a valid option when for automatic computation of self.eigen_vectors_number")
        else:
            self.eigen_vectors_number_selection = "user_defined"

    def _compute_jacobian(self, model_configuration):

        # Check if any of the equilibria are in the supercritical regime (beyond the separatrix) and set it right before
        # the bifurcation.
        zEQ = model_configuration.zEQ
        temp = model_configuration.x1EQ > X1_EQ_CR_DEF - 10 ** (-3)
        if temp.any():
            correction_value = X1_EQ_CR_DEF - 10 ** (-3)
            self.logger.warning("Equibria x1EQ[" + str(numpy.where(temp)[0]) + "]  = " + str(model_configuration.x1EQ[temp]) +
                    "\nwere corrected for LSA to value: X1_EQ_CR_DEF - 10 ** (-3) = " + str(correction_value)
                    + " to be sub-critical!")
            model_configuration.x1EQ[temp] = correction_value
            i_temp = numpy.ones(model_configuration.x1EQ.shape)
            zEQ[temp] = calc_eq_z(model_configuration.x1EQ[temp], model_configuration.yc * i_temp[temp],
                                  model_configuration.Iext1 * i_temp[temp], "2d", 0.0,
                                  model_configuration.slope * i_temp[temp],
                                  model_configuration.a * i_temp[temp], model_configuration.b * i_temp[temp],
                                  model_configuration.d * i_temp[temp])

        fz_jacobian = calc_fz_jac_square_taylor(model_configuration.zEQ, model_configuration.yc,
                                                model_configuration.Iext1, model_configuration.K,
                                                model_configuration.model_connectivity,
                                                model_configuration.a, model_configuration.b, model_configuration.d)

        if numpy.any([numpy.any(numpy.isnan(fz_jacobian.flatten())), numpy.any(numpy.isinf(fz_jacobian.flatten()))]):
            raise_value_error("nan or inf values in dfz")

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

        if self.eigen_vectors_number == disease_hypothesis.number_of_regions:
            # Calculate the propagation strength index by summing all eigenvectors
            lsa_propagation_strength = numpy.abs(numpy.sum(self.eigen_vectors, axis=1))

        else:
            sorted_indices = max(self.eigen_vectors_number, 1)
            # Calculate the propagation strength index by summing the first n eigenvectors (minimum 1)
            if self.weighted_eigenvector_sum:
                lsa_propagation_strength = numpy.abs(weighted_vector_sum(self.eigen_values[:sorted_indices],
                                                                         self.eigen_vectors[:, :sorted_indices],
                                                                         normalize=True))
            else:
                lsa_propagation_strength = numpy.abs(numpy.sum(self.eigen_vectors[:, :sorted_indices], axis=1))

        if self.normalize_propagation_strength:
            # Normalize by the maximum
            lsa_propagation_strength /= numpy.max(lsa_propagation_strength)

        # # TODO: this has to be corrected
        # if self.eigen_vectors_number < 0.2 * disease_hypothesis.number_of_regions:
        #     propagation_strength_elbow = numpy.max([self.get_curve_elbow_point(lsa_propagation_strength),
        #                                     self.eigen_vectors_number])
        # else:
        propagation_strength_elbow = self.get_curve_elbow_point(lsa_propagation_strength)
        propagation_indices = lsa_propagation_strength.argsort()[-propagation_strength_elbow:]

        hypothesis_builder = HypothesisBuilder().set_attributes_based_on_hypothesis(
            disease_hypothesis).set_lsa_propagation_indices(propagation_indices).set_lsa_propagation_strengths(
            lsa_propagation_strength)

        return hypothesis_builder.build_lsa_hypothesis()

    def update_for_pse(self, values, paths, indices):
        for i, val in enumerate(paths):
            vals = val.split(".")
            if vals[0] == "lsa_service":
                getattr(self, vals[1])[indices[i]] = values[i]
