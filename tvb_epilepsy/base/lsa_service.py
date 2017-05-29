# coding=utf-8
"""
Service to do LSA computation.
"""
import numpy
from tvb.basic.logger.builder import get_logger

from tvb_epilepsy.base.calculations import calc_fz_jac_square_taylor
from tvb_epilepsy.base.utils import curve_elbow_point

LOG = get_logger(__name__)


class LSAService(object):
    def _ensure_eigen_vectors_number(self, eigen_vectors_number, propagation_strength_all):
        if eigen_vectors_number is None:
            elbow = curve_elbow_point(propagation_strength_all, interactive=False)
            eigen_vectors_number = elbow + 1

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

        eigen_vectors_number = self._ensure_eigen_vectors_number(eigen_vectors_number, propagation_strength_all)

        if eigen_vectors_number == disease_hypothesis.get_number_of_regions():
            return propagation_strength_all

        sorted_indices = max(eigen_vectors_number, 1)
        # Calculate the propagation strength index by summing the first n eigenvectors (minimum 1)
        propagation_strength = numpy.abs(numpy.sum(sorted_eigen_vectors[:, :sorted_indices], axis=1))
        propagation_strength /= numpy.max(propagation_strength)

        return propagation_strength, eigen_vectors_number
