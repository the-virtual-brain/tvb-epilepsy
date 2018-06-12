# coding=utf-8

import os
import numpy as np
from tvb_epilepsy.io.h5_reader import H5Reader
from tvb_epilepsy.service.hypothesis_builder import HypothesisBuilder
from tvb_epilepsy.service.model_configuration_builder import ModelConfigurationBuilder
from tvb_epilepsy.service.simulator.simulator_builder import SimulatorBuilder
from tvb_epilepsy.io.tvb_data_reader import TVBReader
from tvb_epilepsy.tests.base import BaseTest


class TestSimulationRun(BaseTest):
    fs = 2 * 4096.0
    time_length = 30.0
    report_every_n_monitor_steps = 10.0

    zmode = np.array("lin")
    epileptor_model = "EpileptorDP2D"
    noise_intensity = 10 ** -8

    @staticmethod
    def _prepare_model_for_simulation(connectivity):
        hypothesis = HypothesisBuilder(connectivity.number_of_regions).set_e_hypothesis([1, 1],
                                                                                        [0, 10]).build_hypothesis()
        model_configuration_builder = ModelConfigurationBuilder(connectivity.number_of_regions)
        model_configuration = model_configuration_builder.build_model_from_hypothesis(hypothesis,
                                                                                      connectivity.normalized_weights)
        return model_configuration

    def test_tvb_simulation(self):
        reader = TVBReader()
        connectivity = reader.read_connectivity("connectivity_76.zip")
        model_configuration = self._prepare_model_for_simulation(connectivity)

        simulator_builder = SimulatorBuilder()
        simulator, _, _ = simulator_builder.build_simulator(model_configuration, connectivity)
        ts, status = simulator.launch_simulation(100)
        assert status

    # This can be ran only locally for the moment

    # def test_custom_simulation(self):
    #     reader = H5Reader()
    #     conn = reader.read_connectivity(os.path.join(self.config.input.HEAD, "Connectivity.h5"))
    #     model_configuration = self._prepare_model_for_simulation(conn)
    #
    #     builder = SimulatorBuilder("java").set_simulated_period(self.time_length)
    #     simulator, _, _ = builder.build_simulator_java_from_model_configuration(model_configuration, conn,
    #                                                                             noise_intensity=1e-6
    #                                                                             # noise_intensity=np.array(
    #                                                                             #     [0., 0., 5e-6, 0.0, 5e-6, 0.])
    #                                                                             # noise_intensity=np.zeros(
    #                                                                             #     conn.number_of_regions * 6, )
    #                                                                             )
    #
    #     simulator.config_simulation()
    #     ttavg, tavg_data, status = simulator.launch_simulation()
    #
    #     assert status == 0
    #
    # @classmethod
    # def teardown_class(cls):
    #     os.remove(os.path.join(cls.config.input.HEAD, "SimulationConfiguration.json"))
    #     os.remove(os.path.join(cls.config.input.HEAD, "full-configuration", "full-configuration.h5"))
    #     os.remove(os.path.join(cls.config.input.HEAD, "full-configuration", "ts.h5"))
    #     os.removedirs(os.path.join(cls.config.input.HEAD, "full-configuration"))
