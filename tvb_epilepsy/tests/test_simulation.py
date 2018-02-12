import numpy as np
from tvb_epilepsy.service.hypothesis_builder import HypothesisBuilder
from tvb_epilepsy.service.model_configuration_service import ModelConfigurationService
from tvb_epilepsy.top.scripts.simulation_scripts import setup_TVB_simulation_from_model_configuration, set_time_scales
from tvb_epilepsy.io.tvb_data_reader import TVBReader

head_dir = "head2"


class TestSimulationRun(object):
    fs = 2 * 4096.0
    time_length = 30.0
    report_every_n_monitor_steps = 10.0
    (dt, fsAVG, sim_length, monitor_period, n_report_blocks) \
        = set_time_scales(fs=fs, time_length=time_length, scale_fsavg=None,
                          report_every_n_monitor_steps=report_every_n_monitor_steps)
    zmode = np.array("lin")
    epileptor_model = "EpileptorDP2D"
    noise_intensity = 10 ** -8

    def _prepare_model_for_simulation(self, connectivity):
        hypothesis = HypothesisBuilder().set_nr_of_regions(
            connectivity.number_of_regions).build_excitability_hypothesis([1, 1], [0, 10])
        model_configuration_service = ModelConfigurationService(connectivity.number_of_regions)
        model_configuration = \
            model_configuration_service.build_model_from_hypothesis(hypothesis, connectivity.normalized_weights)
        return model_configuration

    def test_tvb_simulation(self):
        reader = TVBReader()
        connectivity = reader.read_connectivity("connectivity_76.zip")
        model_configuration = self._prepare_model_for_simulation(connectivity)

        simulator = setup_TVB_simulation_from_model_configuration(model_configuration, connectivity, self.dt,
                                                                  self.sim_length, self.monitor_period,
                                                                  self.epileptor_model, zmode=self.zmode,
                                                                  noise_instance=None,
                                                                  noise_intensity=self.noise_intensity,
                                                                  monitor_expressions=None)
        simulator.config_simulation(initial_conditions=None)
        ttavg, tavg_data, status = simulator.launch_simulation(self.n_report_blocks)
        assert status == True

    # This can be ran only locally for the moment

    # def test_custom_simulation(self):
    #     reader = H5Reader()
    #     connectivity = reader.read_connectivity(os.path.join(DATA_TEST, head_dir, "Connectivity.h5"))
    #     model_configuration = self._prepare_model_for_simulation(connectivity)
    #
    #     simulator = setup_custom_simulation_from_model_configuration(model_configuration, connectivity, self.dt,
    #                                                                  self.sim_length,
    #                                                                  self.monitor_period, "CustomEpileptor",
    #                                                                  self.noise_intensity)
    #
    #     simulator.config_simulation()
    #     ttavg, tavg_data, status = simulator.launch_simulation(self.n_report_blocks)
    #
    #     assert status == 0

    # @classmethod
    # def teardown_class(cls):
    #     os.remove(os.path.join(os.path.abspath(data_dir), "SimulationConfiguration.json"))
    #     os.remove(os.path.join(os.path.abspath(data_dir), "full-configuration", "full-configuration.h5"))
    #     os.remove(os.path.join(os.path.abspath(data_dir), "full-configuration", "ts.h5"))
    #     os.removedirs(os.path.join(os.path.abspath(data_dir), "full-configuration"))
