# coding=utf-8

import numpy as np
from tvb_epilepsy.service.hypothesis_builder import HypothesisBuilder
from tvb_epilepsy.service.model_configuration_builder import ModelConfigurationBuilder
from tvb_epilepsy.service.simulator_builder import SimulatorBuilder
from tvb_epilepsy.io.tvb_data_reader import TVBReader
from tvb_epilepsy.tests.base import BaseTest


class TestSimulationRun(BaseTest):
    fs = 2 * 4096.0
    time_length = 30.0
    report_every_n_monitor_steps = 10.0

    zmode = np.array("lin")
    epileptor_model = "EpileptorDP2D"
    noise_intensity = 10 ** -8

    def _prepare_model_for_simulation(self, connectivity):
        hypothesis = HypothesisBuilder().set_nr_of_regions(
            connectivity.number_of_regions).build_excitability_hypothesis([1, 1], [0, 10])
        model_configuration_builder = ModelConfigurationBuilder(connectivity.number_of_regions)
        model_configuration = \
            model_configuration_builder.build_model_from_hypothesis(hypothesis, connectivity.normalized_weights)
        return model_configuration

    def test_tvb_simulation(self):
        reader = TVBReader()
        connectivity = reader.read_connectivity("connectivity_76.zip")
        model_configuration = self._prepare_model_for_simulation(connectivity)

        simulator_builder = SimulatorBuilder().set_time_length(self.time_length)
        simulator = simulator_builder.build_simulator_TVB_fitting(
            model_configuration, connectivity)
        simulator.config_simulation(initial_conditions=None)
        ttavg, tavg_data, status = simulator.launch_simulation(simulator_builder.n_report_blocks)
        assert status == True

    # This can be ran only locally for the moment

    # def test_custom_simulation(self):
    #     reader = H5Reader()
    #     connectivity = reader.read_connectivity(os.path.join(IN_HEAD, "Connectivity.h5"))
    #     model_configuration = self._prepare_model_for_simulation(connectivity)
    #
    #     simulator_builder = SimulatorBuilder().set_time_length(self.time_length)
    #     simulator = simulator_builder.build_simulator_java_from_model_configuration(model_configuration, connectivity)
    #
    #     simulator.config_simulation()
    #     ttavg, tavg_data, status = simulator.launch_simulation(simulator_builder.n_report_blocks)
    #
    #     assert status == 0
    #
    # @classmethod
    # def teardown_class(cls):
    #     os.remove(os.path.join(DATA_TEST, head_dir, "SimulationConfiguration.json"))
    #     os.remove(os.path.join(DATA_TEST, head_dir, "full-configuration", "full-configuration.h5"))
    #     os.remove(os.path.join(DATA_TEST, head_dir, "full-configuration", "ts.h5"))
    #     os.removedirs(os.path.join(DATA_TEST, head_dir, "full-configuration"))
