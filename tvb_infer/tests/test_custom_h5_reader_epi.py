import os
import numpy
from tvb_infer.base.config import InputConfig
from tvb_infer.tvb_epilepsy.base.model.model_configuration import ModelConfiguration
from tvb_infer.tvb_epilepsy.base.model.simulation_settings import SimulationSettings
from tvb_infer.tvb_epilepsy.io.h5_reader import H5Reader
from tvb_infer.tvb_epilepsy.io.h5_writer import H5Writer
from tvb_infer.tvb_epilepsy.service.hypothesis_builder import HypothesisBuilder
from tvb_infer.tvb_lsa.lsa_service import LSAService
from tvb_infer.tvb_epilepsy.service.model_configuration_builder import ModelConfigurationBuilder
from tvb_infer.tests.base import BaseTest


class TestCustomH5Reader(BaseTest):
    reader = H5Reader()
    writer = H5Writer()
    in_head = InputConfig().HEAD
    not_existent_file = "NotExistent.h5"

    def test_read_hypothesis(self):
        test_file = os.path.join(self.config.out.FOLDER_TEMP, "TestHypothesis.h5")
        hypothesis_builder = HypothesisBuilder(3, self.config)
        dummy_hypothesis = hypothesis_builder.set_e_hypothesis([0], [0.6]).build_hypothesis()

        self.writer.write_hypothesis(dummy_hypothesis, test_file)
        hypothesis = self.reader.read_hypothesis(test_file)

        assert dummy_hypothesis.number_of_regions == hypothesis.number_of_regions
        assert numpy.array_equal(dummy_hypothesis.x0_values, hypothesis.x0_values)
        assert dummy_hypothesis.x0_indices == hypothesis.x0_indices
        assert numpy.array_equal(dummy_hypothesis.e_values, hypothesis.e_values)
        assert dummy_hypothesis.e_indices == hypothesis.e_indices
        assert numpy.array_equal(dummy_hypothesis.w_values, hypothesis.w_values)
        assert dummy_hypothesis.w_indices == hypothesis.w_indices
        assert numpy.array_equal(dummy_hypothesis.lsa_propagation_indices, hypothesis.lsa_propagation_indices)
        if len(dummy_hypothesis.lsa_propagation_indices) == 0:
            assert numpy.array_equal([0, 0, 0], hypothesis.lsa_propagation_strengths)
        else:
            assert numpy.array_equal(dummy_hypothesis.lsa_propagation_strengths, hypothesis.lsa_propagation_strengths)

    def test_read_model_configuration(self):
        test_file = os.path.join(self.config.out.FOLDER_TEMP, "TestModelConfiguration.h5")
        dummy_mc = ModelConfiguration(x1eq=numpy.array([2.0, 3.0, 1.0]), zmode=None, zeq=numpy.array([3.0, 2.0, 1.0]),
                                      model_connectivity=numpy.array(
                                          [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [2.0, 2.0, 2.0]]),
                                      Ceq=numpy.array([1.0, 2.0, 3.0]))
        self.writer.write_model_configuration(dummy_mc, test_file)
        mc = self.reader.read_model_configuration(test_file)

        assert numpy.array_equal(dummy_mc.x1eq, mc.x1eq)
        assert numpy.array_equal(dummy_mc.zeq, mc.zeq)
        assert numpy.array_equal(dummy_mc.Ceq, mc.Ceq)
        assert numpy.array_equal(dummy_mc.model_connectivity, mc.model_connectivity)

    def test_read_model_configuration_builder(self):
        test_file = os.path.join(self.config.out.FOLDER_TEMP, "TestModelConfigService.h5")
        dummy_mc_service = ModelConfigurationBuilder(3)
        self.writer.write_model_configuration_builder(dummy_mc_service, test_file)

        mc_service = self.reader.read_model_configuration_builder(test_file)

        assert dummy_mc_service.number_of_regions == mc_service.number_of_regions
        assert numpy.array_equal(dummy_mc_service.x0_values, mc_service.x0_values)
        assert numpy.array_equal(dummy_mc_service.K_unscaled, mc_service.K_unscaled)
        assert numpy.array_equal(dummy_mc_service.e_values, mc_service.e_values)
        assert dummy_mc_service.yc == mc_service.yc
        assert dummy_mc_service.Iext1 == mc_service.Iext1
        assert dummy_mc_service.Iext2 == mc_service.Iext2
        assert dummy_mc_service.a == mc_service.a
        assert dummy_mc_service.b == mc_service.b
        assert dummy_mc_service.d == mc_service.d
        assert dummy_mc_service.slope == mc_service.slope
        assert dummy_mc_service.s == mc_service.s
        assert dummy_mc_service.gamma == mc_service.gamma
        assert dummy_mc_service.tau1 == mc_service.tau1
        assert dummy_mc_service.tau0 == mc_service.tau0
        assert dummy_mc_service.zmode == mc_service.zmode
        assert dummy_mc_service.x1eq_mode == mc_service.x1eq_mode
        assert dummy_mc_service.K.all() == mc_service.K.all()
        assert dummy_mc_service.x0cr == mc_service.x0cr
        assert dummy_mc_service.rx0 == mc_service.rx0

    def test_read_simulation_settigs(self):
        test_file = os.path.join(self.config.out.FOLDER_TEMP, "TestSimSettings.h5")
        dummy_sim_settings = SimulationSettings()
        self.writer.write_simulation_settings(dummy_sim_settings, test_file)

        sim_settings = self.reader.read_simulation_settings(test_file)

        assert dummy_sim_settings.integration_step == sim_settings.integration_step
        assert dummy_sim_settings.simulated_period == sim_settings.simulated_period
        assert dummy_sim_settings.integrator_type == sim_settings.integrator_type
        assert dummy_sim_settings.noise_type == sim_settings.noise_type
        assert dummy_sim_settings.noise_ntau == sim_settings.noise_ntau
        assert dummy_sim_settings.noise_intensity == sim_settings.noise_intensity
        assert dummy_sim_settings.noise_seed == sim_settings.noise_seed
        assert dummy_sim_settings.monitor_type == sim_settings.monitor_type
        assert dummy_sim_settings.monitor_sampling_period == sim_settings.monitor_sampling_period
        assert dummy_sim_settings.monitor_expressions == sim_settings.monitor_expressions
        assert numpy.array_equal(dummy_sim_settings.initial_conditions, sim_settings.initial_conditions)
