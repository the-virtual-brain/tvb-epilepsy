# coding=utf-8
"""
Service to do LSA computation.

TODO: it might be useful to store eigenvalues and eigenvectors, as well as the parameters of the computation,
such as eigen_vectors_number and LSAService in a h5 file
"""
import numpy
from tvb_fit.base.config import CalculusConfig
from tvb_fit.tvb_epilepsy.base.constants.model_constants import X1EQ_CR_DEF
from tvb_fit.base.utils.log_error_utils import initialize_logger, raise_value_error, warning
from tvb_fit.base.utils.data_structures_utils import formal_repr
from tvb_fit.base.computations.analyzers_utils import interval_scaling
from tvb_fit.tvb_epilepsy.base.computation_utils.calculations_utils import calc_fz_jac_square_taylor, calc_jac
from tvb_fit.tvb_epilepsy.base.computation_utils.equilibrium_computation import calc_eq_z
from tvb_fit.base.computations.math_utils import weighted_vector_sum, curve_elbow_point
from tvb_fit.tvb_epilepsy.service.hypothesis_builder import HypothesisBuilder


class LSAService(object):
    logger = initialize_logger(__name__)

    def __init__(self, lsa_method=CalculusConfig.LSA_METHOD,
                 eigen_vectors_number_selection=CalculusConfig.EIGENVECTORS_NUMBER_SELECTION,
                 eigen_vectors_number=None, weighted_eigenvector_sum=CalculusConfig.WEIGHTED_EIGENVECTOR_SUM,
                 normalize_propagation_strength=False):
        self.eigen_vectors_number_selection = eigen_vectors_number_selection
        self.eigen_values = numpy.array([])
        self.eigen_vectors = numpy.array([])
        self.eigen_vectors_number = eigen_vectors_number
        self.weighted_eigenvector_sum = weighted_eigenvector_sum
        self.normalize_propagation_strength = normalize_propagation_strength
        # lsa_method = "1D" (default), "2D"
        # or "auto" in which case "2D" is selected only if there is an unstable fixed point...
        self.lsa_method = lsa_method

    def __repr__(self):
        d = {"01. LSA method": self.lsa_method,
             "02. Eigenvectors' number selection mode": self.eigen_vectors_number_selection,
             "03. Eigenvectors' number": self.eigen_vectors_number_selection,
             "04. Eigen values": self.eigen_values,
             "05. Eigenvectors": self.eigen_vectors,
             "06. Eigenvectors' number": self.eigen_vectors_number,
             "07. Weighted eigenvector's sum flag": str(self.weighted_eigenvector_sum)
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
                self.eigen_vectors_number = self.get_curve_elbow_point(numpy.abs(eigen_values))

            elif self.eigen_vectors_number_selection is "auto_disease":
                self.eigen_vectors_number = len(disease_indices)

            elif self.eigen_vectors_number_selection is "auto_epileptogenicity":
                self.eigen_vectors_number = self.get_curve_elbow_point(e_values)

            elif self.eigen_vectors_number_selection is "auto_excitability":
                self.eigen_vectors_number = self.get_curve_elbow_point(x0_values)

            else:
                raise_value_error("\n" + self.eigen_vectors_number_selection +
                                  "is not a valid option when for automatic computation of self.eigen_vectors_number")
        else:
            self.eigen_vectors_number_selection = "user_defined"

    def _compute_jacobian(self, model_configuration):

        if self.lsa_method == "2D":
            fz_jacobian = calc_jac(model_configuration.x1eq, model_configuration.zeq, model_configuration.yc,
                                   model_configuration.Iext1, model_configuration.x0, model_configuration.K,
                                   model_configuration.connectivity, model_vars=2, zmode=model_configuration.zmode,
                                   a=model_configuration.a, b=model_configuration.b, d=model_configuration.d,
                                   tau1= model_configuration.tau1, tau0=model_configuration.tau0)
        else:
            # Check if any of the equilibria are in the supercritical regime (beyond the separatrix)
            # and set it right before the bifurcation.
            x1eq = numpy.array(model_configuration.x1eq)
            zeq = numpy.array(model_configuration.zeq)
            correction_value = X1EQ_CR_DEF - 10 ** (-3)
            if numpy.any(x1eq > correction_value):
                x1eq_min = numpy.min(x1eq)
                x1eq = interval_scaling(x1eq, min_targ=x1eq_min, max_targ=correction_value,
                                              min_orig=x1eq_min, max_orig=numpy.max(x1eq))
                self.logger.warning("Equilibria x1eq are rescaled for LSA to value: X1EQ_CR_DEF - 10 ** (-3) = "
                                    + str(correction_value) + " to be sub-critical!")
                zeq = calc_eq_z(x1eq, model_configuration.yc, model_configuration.Iext1,
                                      "2d", numpy.zeros(model_configuration.x1eq.shape),
                                      model_configuration.slope, model_configuration.a,
                                      model_configuration.b, model_configuration.d)
            fz_jacobian = calc_fz_jac_square_taylor(zeq, model_configuration.yc, model_configuration.Iext1,
                                                    model_configuration.K, model_configuration.connectivity,
                                                    model_configuration.a, model_configuration.b, model_configuration.d)

        if numpy.any([numpy.any(numpy.isnan(fz_jacobian.flatten())), numpy.any(numpy.isinf(fz_jacobian.flatten()))]):
            raise_value_error("nan or inf values in dfz")

        return fz_jacobian

    def run_lsa(self, disease_hypothesis, model_configuration):

        if self.lsa_method == "auto":
            if numpy.any(model_configuration.x1eq > X1EQ_CR_DEF):
                self.lsa_method = "2D"
            else:
                self.lsa_method = "1D"

        if self.lsa_method == "2D" and numpy.all(model_configuration.x1eq <= X1EQ_CR_DEF):
            warning("LSA with the '2D' method (on the 2D Epileptor model) will not produce interpretable results when"
                    " the equilibrium point of the system is not supercritical (unstable)!")

        jacobian = self._compute_jacobian(model_configuration)

        # Perform eigenvalue decomposition
        eigen_values, eigen_vectors = numpy.linalg.eig(jacobian)
        eigen_values = numpy.real(eigen_values)
        eigen_vectors = numpy.real(eigen_vectors)
        sorted_indices = numpy.argsort(eigen_values, kind='mergesort')
        if self.lsa_method == "2D":
            sorted_indices = sorted_indices[::-1]
        self.eigen_vectors = eigen_vectors[:, sorted_indices]
        self.eigen_values = eigen_values[sorted_indices]

        self._ensure_eigen_vectors_number(self.eigen_values[:disease_hypothesis.number_of_regions],
                                          model_configuration.e_values, model_configuration.x0_values,
                                          disease_hypothesis.regions_disease_indices)

        if self.eigen_vectors_number == disease_hypothesis.number_of_regions:
            # Calculate the propagation strength index by summing all eigenvectors
            lsa_propagation_strength = numpy.abs(numpy.sum(self.eigen_vectors, axis=1))

        else:
            sorted_indices = max(self.eigen_vectors_number, 1)
            # Calculate the propagation strength index by summing the first n eigenvectors (minimum 1)
            if self.weighted_eigenvector_sum:
                lsa_propagation_strength = \
                    numpy.abs(weighted_vector_sum(numpy.array(self.eigen_values[:self.eigen_vectors_number]),
                                                  numpy.array(self.eigen_vectors[:, :self.eigen_vectors_number]),
                                                              normalize=True))
            else:
                lsa_propagation_strength = \
                    numpy.abs(numpy.sum(self.eigen_vectors[:, :self.eigen_vectors_number], axis=1))

        if self.lsa_method == "2D":
            # lsa_propagation_strength = lsa_propagation_strength[:disease_hypothesis.number_of_regions]
            # or
            # lsa_propagation_strength = numpy.where(lsa_propagation_strength[:disease_hypothesis.number_of_regions] >=
            #                                        lsa_propagation_strength[disease_hypothesis.number_of_regions:],
            #                                        lsa_propagation_strength[:disease_hypothesis.number_of_regions],
            #                                        lsa_propagation_strength[disease_hypothesis.number_of_regions:])
            # or
            lsa_propagation_strength = numpy.sqrt(lsa_propagation_strength[:disease_hypothesis.number_of_regions]**2 +
                                                  lsa_propagation_strength[disease_hypothesis.number_of_regions:]**2)
            lsa_propagation_strength = numpy.log10(lsa_propagation_strength)
            lsa_propagation_strength -= lsa_propagation_strength.min()


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

        hypothesis_builder = HypothesisBuilder(disease_hypothesis.number_of_regions). \
                                set_attributes_based_on_hypothesis(disease_hypothesis). \
                                    set_name(disease_hypothesis.name + "_LSA"). \
                                        set_lsa_propagation(propagation_indices, lsa_propagation_strength)

        return hypothesis_builder.build_lsa_hypothesis()

    def update_for_pse(self, values, paths, indices):
        for i, val in enumerate(paths):
            vals = val.split(".")
            if vals[0] == "lsa_service":
                getattr(self, vals[1])[indices[i]] = values[i]
