# coding=utf-8
"""
Service to do X0/E Hypothesis configuration.
"""
import numpy

from tvb_epilepsy.base.calculations import calc_x0cr_r, calc_coupling, calc_x0
from tvb_epilepsy.base.constants import X1_EQ_CR_DEF, E_DEF
from tvb_epilepsy.base.disease_hypothesis import DiseaseHypothesis
from tvb_epilepsy.base.equilibrium_computation import calc_eq_z_2d, eq_x1_hypo_x0_linTaylor, eq_x1_hypo_x0_optimize
from tvb_epilepsy.base.lsa_service import LSAService
from tvb_epilepsy.base.model_configuration import ModelConfiguration


class EquilibrumComputationService(object):
    lsa_service = LSAService()
    x1EQcr = X1_EQ_CR_DEF
    x1eq_mode = "optimize"

    def __init__(self, disease_hypothesis, epileptor_model):
        self.disease_hypothesis = disease_hypothesis
        self.epileptor_model = epileptor_model

    def _ensure_equilibrum(self, x1EQ, zEQ):
        temp = x1EQ > self.x1EQcr - 10 ** (-3)
        if temp.any():
            x1EQ[temp] = self.x1EQcr - 10 ** (-3)
            zEQ = self._compute_z_equilibrium(x1EQ)

        return x1EQ, zEQ

    def _compute_x1_equilibrium(self, E_values):
        array_ones = numpy.ones((self.disease_hypothesis.get_number_of_regions(),), dtype=numpy.float32)
        return ((E_values - 5.0) / 3.0) * array_ones

    def _compute_z_equilibrium(self, x1EQ):
        return calc_eq_z_2d(x1EQ, self.epileptor_model.yc, self.epileptor_model.Iext1)

    def _compute_critical_x0_scaling(self):
        return calc_x0cr_r(self.epileptor_model.yc, self.epileptor_model.Iext1)

    def _compute_coupling_at_equilibrium(self, x1EQ):
        return calc_coupling(x1EQ, self.epileptor_model.K, self.disease_hypothesis.get_weights())

    def _compute_x0(self, x1EQ, zEQ, x0cr, rx0):
        return calc_x0(x1EQ, zEQ, self.epileptor_model.K, self.disease_hypothesis.get_weights(), x0cr, rx0, model="2d",
                       zmode=numpy.array("lin"), z_pos=True)

    def _compute_E_values(self, x1EQ):
        return 3.0 * x1EQ + 5.0

    def _compute_params_after_equilibration(self, x1EQ, zEQ):
        (x0cr, rx0) = self._compute_critical_x0_scaling()
        Ceq = self._compute_coupling_at_equilibrium(x1EQ)
        x0_values = self._compute_x0(x1EQ, zEQ, x0cr, rx0)
        E_values = self._compute_E_values(x1EQ)

        return x0cr, rx0, Ceq, x0_values, E_values

    def _compute_x1_and_z_equilibrum(self, E_values):
        x1EQ = self._compute_x1_equilibrium(E_values)
        zEQ = self._compute_z_equilibrium(x1EQ)

        return x1EQ, zEQ

    def configure_model_from_E_hypothesis(self):
        x1EQ_temp, zEQ_temp = self._compute_x1_and_z_equilibrum(self.disease_hypothesis.get_regions_disease())
        x1EQ, zEQ = self._ensure_equilibrum(x1EQ_temp, zEQ_temp)
        x0cr, rx0, Ceq, x0_values, E_values = self._compute_params_after_equilibration(x1EQ, zEQ)

        model_configuration = ModelConfiguration(x0_values, self.epileptor_model.yc, self.epileptor_model.Iext1,
                                                 self.epileptor_model.K, x0cr, rx0, x1EQ, zEQ, Ceq, E_values)

        lsa_propagation_strength, eigen_vectors_number = self.lsa_service.run_lsa(self.disease_hypothesis,
                                                                                  self.epileptor_model, None, zEQ)

        propagation_indices = lsa_propagation_strength.argsort()[-eigen_vectors_number:]

        lsa_hypothesis = DiseaseHypothesis("E", self.disease_hypothesis.get_connectivity(),
                                           self.disease_hypothesis.get_disease_indices(),
                                           self.disease_hypothesis.get_disease_values(), propagation_indices,
                                           lsa_propagation_strength, "LSA_" + self.disease_hypothesis.get_name())

        return model_configuration, lsa_hypothesis

    def configure_model_from_x0_hypothesis(self):
        # TODO: how to handle x0 and E indices for x1EQ computation?

        (x0cr, rx0) = self._compute_critical_x0_scaling()
        E_values = E_DEF * numpy.ones((self.disease_hypothesis.get_number_of_regions(),), dtype=numpy.float32)
        x1EQ_temp, zEQ_temp = self._compute_x1_and_z_equilibrum(E_values)
        E_indices = self.disease_hypothesis.get_E_indices_when_x0_are_defined()

        # Convert x0 to an array of (1,len(ix0)) shape
        x0_values = numpy.expand_dims(numpy.array(self.disease_hypothesis.get_regions_disease()), 1).T

        if self.x1eq_mode == "linTaylor":
            x1EQ = \
                eq_x1_hypo_x0_linTaylor(self.disease_hypothesis.get_disease_indices(), E_indices, x1EQ_temp, zEQ_temp,
                                        x0_values, x0cr, rx0, self.epileptor_model.yc, self.epileptor_model.Iext1,
                                        self.epileptor_model.K, self.disease_hypothesis.get_weights())[0]
        else:
            x1EQ = \
                eq_x1_hypo_x0_optimize(self.disease_hypothesis.get_disease_indices(), E_indices, x1EQ_temp, zEQ_temp,
                                       x0_values, x0cr, rx0, self.epileptor_model.yc, self.epileptor_model.Iext1,
                                       self.epileptor_model.K, self.disease_hypothesis.get_weights())[0]

        zEQ = self._compute_z_equilibrium(x1EQ)

        x1EQ_final, zEQ_final = self._ensure_equilibrum(x1EQ, zEQ)
        x0cr, rx0, Ceq, eq_x0_values, E_values = self._compute_params_after_equilibration(x1EQ_final, zEQ_final)

        model_configuration = ModelConfiguration(eq_x0_values, self.epileptor_model.yc, self.epileptor_model.Iext1,
                                                 self.epileptor_model.K, x0cr, rx0, x1EQ_final, zEQ_final, Ceq,
                                                 E_values)

        lsa_propagation_strength, eigen_vectors_number = self.lsa_service.run_lsa(self.disease_hypothesis,
                                                                                  self.epileptor_model, None,
                                                                                  zEQ_final)

        propagation_indices = lsa_propagation_strength.argsort()[-eigen_vectors_number:]

        lsa_hypothesis = DiseaseHypothesis("x0", self.disease_hypothesis.get_connectivity(),
                                           self.disease_hypothesis.get_disease_indices(),
                                           self.disease_hypothesis.get_disease_values(), propagation_indices,
                                           lsa_propagation_strength, "LSA_" + self.disease_hypothesis.get_name())

        return model_configuration, lsa_hypothesis
