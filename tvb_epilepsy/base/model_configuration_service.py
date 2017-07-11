# coding=utf-8
"""
Service to do X0/E Hypothesis configuration.
"""
from collections import OrderedDict

import numpy

from tvb.basic.logger.builder import get_logger
from tvb_epilepsy.base.constants import X1_EQ_CR_DEF, E_DEF, X0_DEF, K_DEF, YC_DEF, I_EXT1_DEF, A_DEF, B_DEF
from tvb_epilepsy.base.utils import formal_repr
from tvb_epilepsy.base.h5_model import object_to_h5_model
from tvb_epilepsy.base.calculations import calc_x0cr_r, calc_coupling, calc_x0
from tvb_epilepsy.base.equilibrium_computation import calc_eq_z_2d, eq_x1_hypo_x0_linTaylor, eq_x1_hypo_x0_optimize
from tvb_epilepsy.base.model_configuration import ModelConfiguration

# NOTES:
# In the future all the related to model configuration parameters might be part of the disease hypothesis:
# yc=YC_DEF, Iext1=I_EXT1_DEF, K=K_DEF, a=A_DEF, b=B_DEF
# For now, we assume default values, or externally set

LOG = get_logger(__name__)


class ModelConfigurationService(object):

    x1EQcr = X1_EQ_CR_DEF

    def __init__(self, number_of_regions, x0=X0_DEF,yc=YC_DEF, Iext1=I_EXT1_DEF, K=K_DEF, a=A_DEF, b=B_DEF, E=E_DEF, 
                 x1eq_mode="optimize"):
        self.number_of_regions = number_of_regions
        self.x0 = x0 * numpy.ones((self.number_of_regions,), dtype=numpy.float32)
        self.yc = yc
        self.Iext1 = Iext1
        self.a = a
        self.b = b
        self.x1eq_mode = x1eq_mode
        self.K_unscaled = K * numpy.ones((self.number_of_regions,), dtype=numpy.float32)
        self.K = None
        self._normalize_global_coupling()
        self.E = E * numpy.ones((self.number_of_regions,), dtype=numpy.float32)

    def __repr__(self):
        d = {"01. Number of regions": self.number_of_regions,
             "02. x0": self.x0,
             "03. Iext1": self.Iext1,
             "04. a": self.a,
             "05. b": self.b,
             "06. x1eq_mode": self.x1eq_mode,
             "07. K_unscaled": self.K_unscaled,
             "08. K": self.K,
             "09. E": self.E,
             }
        return formal_repr(self, OrderedDict(sorted(d.items(), key=lambda t: t[0])))

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

    def _ensure_equilibrum(self, x1EQ, zEQ):
        temp = x1EQ > self.x1EQcr - 10 ** (-3)
        if temp.any():
            x1EQ[temp] = self.x1EQcr - 10 ** (-3)
            zEQ = self._compute_z_equilibrium(x1EQ)

        return x1EQ, zEQ

    def _compute_x1_equilibrium_from_E(self, e_values):
        array_ones = numpy.ones((self.number_of_regions,), dtype=numpy.float32)
        return ((e_values - 5.0) / 3.0) * array_ones

    def _compute_z_equilibrium(self, x1EQ):
        return calc_eq_z_2d(x1EQ, self.yc, self.Iext1)

    def _compute_critical_x0_scaling(self):
        return calc_x0cr_r(self.yc, self.Iext1, a=self.a, b=self.b)

    def _compute_coupling_at_equilibrium(self, x1EQ, weights):
        return calc_coupling(x1EQ, self.K, weights)

    def _compute_x0(self, x1EQ, zEQ, x0cr, rx0, weights):
        return calc_x0(x1EQ, zEQ, self.K, weights, x0cr, rx0)

    def _compute_e_values(self, x1EQ):
        return 3.0 * x1EQ + 5.0

    def _compute_params_after_equilibration(self, x1EQ, zEQ, weights):
        (x0cr, rx0) = self._compute_critical_x0_scaling()
        Ceq = self._compute_coupling_at_equilibrium(x1EQ, weights)
        x0_values = self._compute_x0(x1EQ, zEQ, x0cr, rx0, weights)
        e_values = self._compute_e_values(x1EQ)
        return x0cr, rx0, Ceq, x0_values, e_values

    def _compute_x1_and_z_equilibrium_from_E(self, e_values):
        x1EQ = self._compute_x1_equilibrium_from_E(e_values)
        zEQ = self._compute_z_equilibrium(x1EQ)
        return x1EQ, zEQ

    def _compute_x1_equilibrium(self, e_indices, x1EQ, zEQ, x0_values, x0cr, rx0, weights):
        x0_indices = numpy.delete(numpy.array(range(weights.shape[0])), e_indices)
        if self.x1eq_mode == "linTaylor":
            x1EQ = \
                eq_x1_hypo_x0_linTaylor(x0_indices, e_indices, x1EQ, zEQ, x0_values, x0cr, rx0,
                                        self.yc, self.Iext1, self.K, weights)[0]
        else:
            x1EQ = \
                eq_x1_hypo_x0_optimize(x0_indices, e_indices, x1EQ, zEQ, x0_values, x0cr, rx0,
                                       self.yc, self.Iext1, self.K, weights)[0]
        return x1EQ

    def _normalize_global_coupling(self):
        self.K = self.K_unscaled / self.number_of_regions

    def configure_model_from_equilibrium(self, x1EQ, zEQ, weights):
        x1EQ, zEQ = self._ensure_equilibrum(x1EQ, zEQ)
        x0cr, rx0, Ceq, x0_values, e_values = self._compute_params_after_equilibration(x1EQ, zEQ, weights)
        model_configuration = ModelConfiguration(self.yc, self.Iext1, self.K, self.a, self.b,
                                                 x0cr, rx0, x1EQ, zEQ, Ceq, x0_values, e_values, weights)
        return model_configuration

    def configure_model_from_E_hypothesis(self, disease_hypothesis):
        # Always normalize K first
        self._normalize_global_coupling()

        # Then apply connectivity disease hypothesis scaling if any:
        connectivity = disease_hypothesis.get_weights()
        if len(disease_hypothesis.w_indices) > 0:
            connectivity *= disease_hypothesis.get_connectivity_disease()

        # All nodes except for the diseased ones will get the default epileptogenicity:
        e_values = numpy.array(self.E)
        e_values[disease_hypothesis.e_indices] = disease_hypothesis.e_values

        # Compute equilibrium from epileptogenicity:
        x1EQ, zEQ = self._compute_x1_and_z_equilibrium_from_E(e_values)
        x1EQ, zEQ = self._ensure_equilibrum(x1EQ, zEQ)

        return self.configure_model_from_equilibrium(x1EQ, zEQ, connectivity)

    def configure_model_from_hypothesis(self, disease_hypothesis):
        # Always normalize K first
        self._normalize_global_coupling()

        # Then apply connectivity disease hypothesis scaling if any:
        connectivity = disease_hypothesis.get_weights()
        if len(disease_hypothesis.w_indices) > 0:
            connectivity *= disease_hypothesis.get_connectivity_disease()

        # We assume that all nodes have the default (healthy) excitability:
        x0_values = numpy.array(self.x0)
        # ...and some  excitability-diseased ones:
        x0_values[disease_hypothesis.x0_indices] = disease_hypothesis.x0_values
        # x0 values must have size of len(x0_indices):
        x0_values = numpy.delete(x0_values, disease_hypothesis.e_indices)

        # There might be some epileptogenicity-diseased regions as well:
        # Initialize with the default E
        e_values = numpy.array(self.E)
        # and assign any diseased E_values if any
        e_values[disease_hypothesis.e_indices] = disease_hypothesis.e_values

        # Compute equilibrium from epileptogenicity:
        x1EQ_temp, zEQ_temp = self._compute_x1_and_z_equilibrium_from_E(e_values)

        (x0cr, rx0) = self._compute_critical_x0_scaling()

        # Now, solve the system in order to compute equilibrium:
        x1EQ = self._compute_x1_equilibrium(disease_hypothesis.e_indices, x1EQ_temp, zEQ_temp, x0_values, x0cr, rx0,
                                            connectivity)
        zEQ = self._compute_z_equilibrium(x1EQ)

        return self.configure_model_from_equilibrium(x1EQ, zEQ, connectivity)

