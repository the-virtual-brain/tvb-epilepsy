# coding=utf-8
"""
Service to do LSA computation.
"""
import numpy
from tvb.basic.logger.builder import get_logger

from tvb_epilepsy.base.calculations import calc_fz_jac_square_taylor
from tvb_epilepsy.base.constants import EIGENVECTORS_NUMBER_SELECTION
from tvb_epilepsy.base.utils import curve_elbow_point

LOG = get_logger(__name__)


class LSAService(object):
    def get_curve_elbow_point(self, values_array):
        return curve_elbow_point(values_array)

    def _ensure_eigen_vectors_number(self, eigen_vectors_number, eigen_values, e_values, x0_values):
        if eigen_vectors_number is None:
            if EIGENVECTORS_NUMBER_SELECTION is "auto_eigenvals":
                eigen_vectors_number = self.get_curve_elbow_point(numpy.abs(eigen_values))

            elif EIGENVECTORS_NUMBER_SELECTION is "auto_epileptogenicity":
                eigen_vectors_number = self.get_curve_elbow_point(e_values)

            elif EIGENVECTORS_NUMBER_SELECTION is "auto_x0":
                eigen_vectors_number = self.get_curve_elbow_point(x0_values)

        return eigen_vectors_number

    def _compute_jacobian(self, model, weights, z_eq_point):
        fz_jacobian = calc_fz_jac_square_taylor(z_eq_point, model.yc, model.Iext1, model.K, weights)

        if numpy.any([numpy.any(numpy.isnan(fz_jacobian.flatten())), numpy.any(numpy.isinf(fz_jacobian.flatten()))]):
            raise ValueError("nan or inf values in dfz")

        return fz_jacobian

    def run_lsa(self, disease_hypothesis, model, eigen_vectors_number, z_eq_point):
        jacobian = self._compute_jacobian(model, disease_hypothesis.get_weights(), z_eq_point)
        # Perform eigenvalue decomposition
        eigen_values, eigen_vectors = numpy.linalg.eig(jacobian)

        sorted_indices = numpy.argsort(eigen_values, kind='mergesort')
        sorted_eigen_vectors = eigen_vectors[:, sorted_indices]

        # Calculate the propagation strength index by summing all eigenvectors
        propagation_strength_all = numpy.abs(numpy.sum(sorted_eigen_vectors, axis=1))
        propagation_strength_all /= numpy.max(propagation_strength_all)

        eigen_vectors_number = self._ensure_eigen_vectors_number(eigen_vectors_number, propagation_strength_all,
                                                                 disease_hypothesis.get_e_values_for_all_regions(),
                                                                 disease_hypothesis.get_x0_values_for_all_regions())

        if eigen_vectors_number == disease_hypothesis.get_number_of_regions():
            return propagation_strength_all

        sorted_indices = max(eigen_vectors_number, 1)
        # Calculate the propagation strength index by summing the first n eigenvectors (minimum 1)
        propagation_strength = numpy.abs(numpy.sum(sorted_eigen_vectors[:, :sorted_indices], axis=1))
        propagation_strength /= numpy.max(propagation_strength)

        return propagation_strength
