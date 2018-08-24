import os
import numpy

from tvb_fit.tests.base import BaseTest
from tvb_fit.base.model.simulation_settings import SimulationSettings
from tvb_fit.service.sensitivity_analysis_service import SensitivityAnalysisService

from tvb_fit.tvb_epilepsy.base.model.epileptor_model_configuration \
    import EpileptorModelConfiguration as ModelConfiguration
from tvb_fit.tvb_epilepsy.service.hypothesis_builder import HypothesisBuilder
from tvb_fit.tvb_epilepsy.service.model_configuration_builder import ModelConfigurationBuilder
from tvb_fit.tvb_epilepsy.service.model_inversion_services import ModelInversionService
from tvb_fit.tvb_epilepsy.service.lsa_service import LSAService
from tvb_fit.tvb_epilepsy.service.pse.lsa_pse_service import LSAPSEService
from tvb_fit.tvb_epilepsy.io.h5_writer import H5Writer


class TestCustomH5writer(BaseTest):
    writer = H5Writer()

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
                                      connectivity=self.dummy_connectivity.normalized_weights)

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

    def test_write_simulation_settings(self):
        test_file = os.path.join(self.config.out.FOLDER_TEMP, "TestSimSettings.h5")
        dummy_sim_settings = SimulationSettings()

        assert not os.path.exists(test_file)

        self.writer.write_simulation_settings(dummy_sim_settings, test_file)

        assert os.path.exists(test_file)

