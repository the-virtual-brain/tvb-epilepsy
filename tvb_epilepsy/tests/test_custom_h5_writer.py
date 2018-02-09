import os
import numpy
from tvb_epilepsy.base.constants.model_constants import X1_EQ_CR_DEF
from tvb_epilepsy.base.model.disease_hypothesis import DiseaseHypothesis
from tvb_epilepsy.base.model.model_configuration import ModelConfiguration
from tvb_epilepsy.base.model.vep.connectivity import Connectivity
from tvb_epilepsy.base.model.vep.head import Head
from tvb_epilepsy.base.model.vep.sensors import Sensors
from tvb_epilepsy.base.model.vep.surface import Surface
from tvb_epilepsy.base.simulation_settings import SimulationSettings
from tvb_epilepsy.io.h5_writer import H5Writer
from tvb_epilepsy.service.lsa_service import LSAService
from tvb_epilepsy.service.model_configuration_service import ModelConfigurationService
from tvb_epilepsy.service.model_inversion.model_inversion_service import ModelInversionService
from tvb_epilepsy.service.pse.lsa_pse_service import LSAPSEService
from tvb_epilepsy.service.sensitivity_analysis_service import SensitivityAnalysisService
from tvb_epilepsy.tests.base import remove_temporary_test_files, get_temporary_folder


class TestCustomH5writer(object):
    writer = H5Writer()

    dummy_connectivity = Connectivity("", numpy.array([[1.0, 2.0, 3.0], [2.0, 3.0, 1.0], [3.0, 2.0, 1.0]]),
                                      numpy.array([[4, 5, 6], [5, 6, 4], [6, 4, 5]]), labels=["a", "b", "c"],
                                      centres=numpy.array([1.0, 2.0, 3.0]), normalized_weights=numpy.array(
            [[1.0, 2.0, 3.0], [2.0, 3.0, 1.0], [3.0, 2.0, 1.0]]))
    dummy_surface = Surface(numpy.array([[1, 2, 3], [2, 3, 1], [3, 1, 2]]), numpy.array([[0, 1, 2]]))
    dummy_sensors = Sensors(numpy.array(["sens1", "sens2"]), numpy.array([[0, 0, 0], [0, 1, 0]]),
                            gain_matrix=numpy.array([[1, 2, 3], [2, 3, 4]]))

    def _prepare_dummy_head(self):
        return Head(self.dummy_connectivity, self.dummy_surface, sensorsSEEG=[self.dummy_sensors])

    def test_write_connectivity(self):
        test_file = os.path.join(get_temporary_folder(), "TestConnectivity.h5")

        assert not os.path.exists(test_file)

        self.writer.write_connectivity(self.dummy_connectivity, test_file)

        assert os.path.exists(test_file)

    def test_write_connectivity_with_normalized_weigths(self):
        test_file = os.path.join(get_temporary_folder(), "TestConnectivityNorm.h5")

        assert not os.path.exists(test_file)

        connectivity = self.dummy_connectivity
        self.writer.write_connectivity(connectivity, test_file)

        assert os.path.exists(test_file)

    def test_write_surface(self):
        test_file = os.path.join(get_temporary_folder(), "TestSurface.h5")

        assert not os.path.exists(test_file)

        self.writer.write_surface(self.dummy_surface, test_file)

        assert os.path.exists(test_file)

    def test_write_sensors(self):
        test_file = os.path.join(get_temporary_folder(), "TestSensors.h5")

        assert not os.path.exists(test_file)

        self.writer.write_sensors(self.dummy_sensors, test_file)

        assert os.path.exists(test_file)

    def test_write_head(self):
        test_folder = os.path.join(get_temporary_folder(), "test_head")

        assert not os.path.exists(test_folder)

        head = self._prepare_dummy_head()
        self.writer.write_head(head, test_folder)

        assert os.path.exists(test_folder)
        assert len(os.listdir(test_folder)) >= 3

    def test_write_hypothesis(self):
        test_file = os.path.join(get_temporary_folder(), "TestHypothesis.h5")
        dummy_hypothesis = DiseaseHypothesis(3, excitability_hypothesis={tuple([0]): numpy.array([0.6])},
                                             epileptogenicity_hypothesis={})

        assert not os.path.exists(test_file)

        self.writer.write_hypothesis(dummy_hypothesis, test_file)

        assert os.path.exists(test_file)

    def test_write_model_configuration(self):
        test_file = os.path.join(get_temporary_folder(), "TestModelConfiguration.h5")
        dummy_mc = ModelConfiguration(x1EQ=numpy.array([2.0, 3.0, 1.0]), zmode=None,
                                      zEQ=numpy.array([3.0, 2.0, 1.0]), Ceq=numpy.array([1.0, 2.0, 3.0]),
                                      model_connectivity=self.dummy_connectivity.normalized_weights)

        assert not os.path.exists(test_file)

        self.writer.write_model_configuration(dummy_mc, test_file)

        assert os.path.exists(test_file)

    def test_write_model_configuration_service(self):
        test_file = os.path.join(get_temporary_folder(), "TestModelConfigurationService.h5")
        dummy_mc_service = ModelConfigurationService(3)

        assert not os.path.exists(test_file)

        self.writer.write_model_configuration_service(dummy_mc_service, test_file)

        assert os.path.exists(test_file)

    def test_write_lsa_service(self):
        test_file = os.path.join(get_temporary_folder(), "TestLSAService.h5")
        dummy_lsa_service = LSAService()

        assert not os.path.exists(test_file)

        self.writer.write_lsa_service(dummy_lsa_service, test_file)

        assert os.path.exists(test_file)

    def test_write_model_inversion_service(self):
        test_file = os.path.join(get_temporary_folder(), "TestModelInversionService.h5")
        dummy_model_inversion_service = ModelInversionService(
            ModelConfiguration(model_connectivity=self.dummy_connectivity.normalized_weights, x1EQ=X1_EQ_CR_DEF),
            dynamical_model="Epileptor", sig_eq=(-4.0 / 3.0 - -5.0 / 3.0) / 10.0)

        assert not os.path.exists(test_file)

        self.writer.write_model_inversion_service(dummy_model_inversion_service, test_file)

        assert os.path.exists(test_file)

    def test_write_pse_service(self):
        test_file = os.path.join(get_temporary_folder(), "TestPSEService.h5")
        dummy_pse_service = LSAPSEService(
            hypothesis=DiseaseHypothesis(3, excitability_hypothesis={tuple([0]): numpy.array([0.6])},
                                         epileptogenicity_hypothesis={}),
            params_pse={"path": [], "indices": [], "name": [], "bounds": []})

        assert not os.path.exists(test_file)

        self.writer.write_pse_service(dummy_pse_service, test_file)

        assert os.path.exists(test_file)

    def test_write_sensitivity_analysis_service(self):
        test_file = os.path.join(get_temporary_folder(), "TestSensitivityAnalysisService.h5")
        dummy_sa_service = SensitivityAnalysisService(
            [{"name": "test1", "samples": [1, 2], "bounds": []}],
            [{"names": ["LSA Propagation Strength"], "values": numpy.array([1, 2])}])

        assert not os.path.exists(test_file)

        self.writer.write_sensitivity_analysis_service(dummy_sa_service, test_file)

        assert os.path.exists(test_file)

    def test_write_dictionary(self):
        test_file = os.path.join(get_temporary_folder(), "TestDict.h5")
        dummy_dict = dict(
            {"n_loops": 96, "params_indices": numpy.array([0, 1, 2]), "params_names": numpy.array(["x0", "z", "x0"]),
             "params_samples": numpy.array([[0.0, 0.1, 0.2], [0.3, 0.0, 0.1], [0.2, 0.3, 0.0]]), "task": "LSA"})

        assert not os.path.exists(test_file)

        self.writer.write_dictionary(dummy_dict, test_file)

        assert os.path.exists(test_file)

    def test_write_simulation_settings(self):
        test_file = os.path.join(get_temporary_folder(), "TestSimSettings.h5")
        dummy_sim_settings = SimulationSettings()

        assert not os.path.exists(test_file)

        self.writer.write_simulation_settings(dummy_sim_settings, test_file)

        assert os.path.exists(test_file)

    @classmethod
    def teardown_class(cls):
        remove_temporary_test_files()
