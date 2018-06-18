import os
import numpy
from tvb_epilepsy.base.constants.model_constants import X1EQ_CR_DEF
from tvb_epilepsy.base.model.model_configuration import ModelConfiguration
from tvb_epilepsy.base.simulation_settings import SimulationSettings
from tvb_epilepsy.io.h5_writer import H5Writer
from tvb_epilepsy.service.hypothesis_builder import HypothesisBuilder
from tvb_epilepsy.service.lsa_service import LSAService
from tvb_epilepsy.service.model_configuration_builder import ModelConfigurationBuilder
from tvb_epilepsy.service.model_inversion.model_inversion_services import ModelInversionService
from tvb_epilepsy.service.pse.lsa_pse_service import LSAPSEService
from tvb_epilepsy.service.sensitivity_analysis_service import SensitivityAnalysisService
from tvb_epilepsy.tests.base import BaseTest


class TestCustomH5writer(BaseTest):
    writer = H5Writer()

    def test_write_connectivity(self):
        test_file = os.path.join(self.config.out.FOLDER_TEMP, "TestConnectivity.h5")

        assert not os.path.exists(test_file)

        self.writer.write_connectivity(self.dummy_connectivity, test_file)

        assert os.path.exists(test_file)

    def test_write_connectivity_with_normalized_weigths(self):
        test_file = os.path.join(self.config.out.FOLDER_TEMP, "TestConnectivityNorm.h5")

        assert not os.path.exists(test_file)

        connectivity = self.dummy_connectivity
        self.writer.write_connectivity(connectivity, test_file)

        assert os.path.exists(test_file)

    def test_write_surface(self):
        test_file = os.path.join(self.config.out.FOLDER_TEMP, "TestSurface.h5")

        assert not os.path.exists(test_file)

        self.writer.write_surface(self.dummy_surface, test_file)

        assert os.path.exists(test_file)

    def test_write_sensors(self):
        test_file = os.path.join(self.config.out.FOLDER_TEMP, "TestSensors.h5")

        assert not os.path.exists(test_file)

        self.writer.write_sensors(self.dummy_sensors, test_file)

        assert os.path.exists(test_file)

    def test_write_head(self):
        test_folder = os.path.join(self.config.out.FOLDER_TEMP, "test_head")

        assert not os.path.exists(test_folder)

        head = self._prepare_dummy_head_from_dummy_attrs()
        self.writer.write_head(head, test_folder)

        assert os.path.exists(test_folder)
        assert len(os.listdir(test_folder)) >= 3

    def test_write_hypothesis(self):
        test_file = os.path.join(self.config.out.FOLDER_TEMP, "TestHypothesis.h5")
        dummy_hypothesis = HypothesisBuilder(3).set_e_hypothesis([0], [0.6]).build_hypothesis()

        assert not os.path.exists(test_file)

        self.writer.write_hypothesis(dummy_hypothesis, test_file)

        assert os.path.exists(test_file)

    def test_write_model_configuration(self):
        test_file = os.path.join(self.config.out.FOLDER_TEMP, "TestModelConfiguration.h5")
        dummy_mc = ModelConfiguration(x1eq=numpy.array([2.0, 3.0, 1.0]), zmode=None,
                                      zeq=numpy.array([3.0, 2.0, 1.0]), Ceq=numpy.array([1.0, 2.0, 3.0]),
                                      model_connectivity=self.dummy_connectivity.normalized_weights)

        assert not os.path.exists(test_file)

        self.writer.write_model_configuration(dummy_mc, test_file)

        assert os.path.exists(test_file)

    def test_write_model_configuration_builder(self):
        test_file = os.path.join(self.config.out.FOLDER_TEMP, "TestModelConfigurationService.h5")
        dummy_mc_service = ModelConfigurationBuilder(3)

        assert not os.path.exists(test_file)

        self.writer.write_model_configuration_builder(dummy_mc_service, test_file)

        assert os.path.exists(test_file)

    def test_write_lsa_service(self):
        test_file = os.path.join(self.config.out.FOLDER_TEMP, "TestLSAService.h5")
        dummy_lsa_service = LSAService()

        assert not os.path.exists(test_file)

        self.writer.write_lsa_service(dummy_lsa_service, test_file)

        assert os.path.exists(test_file)

    def test_write_model_inversion_service(self):
        test_file = os.path.join(self.config.out.FOLDER_TEMP, "TestModelInversionService.h5")
        dummy_model_inversion_service = ModelInversionService()

        assert not os.path.exists(test_file)

        self.writer.write_model_inversion_service(dummy_model_inversion_service, test_file)

        assert os.path.exists(test_file)

    def test_write_pse_service(self):
        test_file = os.path.join(self.config.out.FOLDER_TEMP, "TestPSEService.h5")
        hypothesis = HypothesisBuilder(3).set_e_hypothesis([0], [0.6]).build_hypothesis()
        dummy_pse_service = LSAPSEService(hypothesis=hypothesis,
                                          params_pse={"path": [], "indices": [], "name": [], "bounds": []})

        assert not os.path.exists(test_file)

        self.writer.write_pse_service(dummy_pse_service, test_file)

        assert os.path.exists(test_file)

    def test_write_sensitivity_analysis_service(self):
        test_file = os.path.join(self.config.out.FOLDER_TEMP, "TestSensitivityAnalysisService.h5")
        dummy_sa_service = SensitivityAnalysisService(
            [{"name": "test1", "samples": [1, 2], "bounds": []}],
            [{"names": ["LSA Propagation Strength"], "values": numpy.array([1, 2])}])

        assert not os.path.exists(test_file)

        self.writer.write_sensitivity_analysis_service(dummy_sa_service, test_file)

        assert os.path.exists(test_file)

    def test_write_dictionary(self):
        test_file = os.path.join(self.config.out.FOLDER_TEMP, "TestDict.h5")
        dummy_dict = dict(
            {"n_loops": 96, "params_indices": numpy.array([0, 1, 2]), "params_names": numpy.array(["x0", "z", "x0"]),
             "params_samples": numpy.array([[0.0, 0.1, 0.2], [0.3, 0.0, 0.1], [0.2, 0.3, 0.0]]), "task": "LSA"})

        assert not os.path.exists(test_file)

        self.writer.write_dictionary(dummy_dict, test_file)

        assert os.path.exists(test_file)

    def test_write_simulation_settings(self):
        test_file = os.path.join(self.config.out.FOLDER_TEMP, "TestSimSettings.h5")
        dummy_sim_settings = SimulationSettings()

        assert not os.path.exists(test_file)

        self.writer.write_simulation_settings(dummy_sim_settings, test_file)

        assert os.path.exists(test_file)

    @classmethod
    def teardown_class(cls):
        head_dir = os.path.join(cls.config.out.FOLDER_TEMP, "test_head")
        if os.path.exists(head_dir):
            for dir_file in os.listdir(head_dir):
                os.remove(os.path.join(os.path.abspath(head_dir), dir_file))
            os.rmdir(head_dir)
        BaseTest.teardown_class()
