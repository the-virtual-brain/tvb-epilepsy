import os
import numpy
from tvb_epilepsy.base.constants.configurations import DATA_TEST
from tvb_epilepsy.base.model.disease_hypothesis import DiseaseHypothesis
from tvb_epilepsy.base.model.model_configuration import ModelConfiguration
from tvb_epilepsy.base.model.vep.sensors import Sensors
from tvb_epilepsy.io.h5.reader_custom import CustomH5Reader
from tvb_epilepsy.io.h5.writer_custom import CustomH5Writer
from tvb_epilepsy.tests.base import get_temporary_folder, remove_temporary_test_files

head_dir = "head2"

class TestCustomH5Reader():
    reader = CustomH5Reader()
    writer = CustomH5Writer()
    not_existent_file = "NotExistent.h5"

    def test_read_connectivity(self):
        connectivity = self.reader.read_connectivity(os.path.join(DATA_TEST, head_dir, "Connectivity.h5"))

        assert connectivity is not None
        assert connectivity.number_of_regions == 76

    def test_read_surface(self):
        surface = self.reader.read_surface(os.path.join(DATA_TEST, head_dir, "CorticalSurface.h5"))

        assert surface is not None
        assert surface.vertices.shape[0] == 16

    def test_read_surface_not_existent_file(self):
        surface = self.reader.read_surface(os.path.join(DATA_TEST, head_dir, self.not_existent_file))

        assert surface is None

    def test_read_region_mapping(self):
        region_mapping = self.reader.read_region_mapping(os.path.join(DATA_TEST, head_dir, "RegionMapping.h5"))

        assert isinstance(region_mapping, numpy.ndarray)
        assert region_mapping.size == 16

    def test_read_region_mapping_not_existent_file(self):
        region_mapping = self.reader.read_region_mapping(os.path.join(DATA_TEST, head_dir, self.not_existent_file))

        assert isinstance(region_mapping, numpy.ndarray)
        assert region_mapping.size == 0

    def test_read_volume_mapping(self):
        volume_mapping = self.reader.read_volume_mapping(os.path.join(DATA_TEST, head_dir, "VolumeMapping.h5"))

        assert isinstance(volume_mapping, numpy.ndarray)
        assert volume_mapping.shape == (6, 5, 4)

    def test_read_volume_mapping_not_existent_file(self):
        volume_mapping = self.reader.read_volume_mapping(os.path.join(DATA_TEST, head_dir, self.not_existent_file))

        assert isinstance(volume_mapping, numpy.ndarray)
        assert volume_mapping.shape == (0,)

    def test_sensors_of_type(self):
        sensors_file = os.path.join(DATA_TEST, head_dir, "SensorsSEEG_20.h5")
        sensors = self.reader.read_sensors_of_type(sensors_file, Sensors.TYPE_SEEG)

        assert sensors is not None
        assert sensors.number_of_sensors == 20

    def test_sensors_of_type_not_existent_file(self):
        sensors_file = os.path.join(DATA_TEST, head_dir, self.not_existent_file)
        sensors = self.reader.read_sensors_of_type(sensors_file, Sensors.TYPE_SEEG)

        assert sensors is None

    def test_read_sensors(self):
        sensors_seeg, sensors_eeg, sensors_meg = self.reader.read_sensors(os.path.join(DATA_TEST, head_dir))

        assert len(sensors_seeg) > 0
        assert len(sensors_eeg) == 0
        assert len(sensors_meg) == 0
        assert sensors_seeg[0] is not None
        assert sensors_seeg[0].number_of_sensors == 20

    def test_read_hypothesis(self):
        test_file = os.path.join(get_temporary_folder(), "TestHypothesis.h5")
        dummy_hypothesis = DiseaseHypothesis(3, excitability_hypothesis={tuple([0]): numpy.array([0.6])},
                                             epileptogenicity_hypothesis={})

        self.writer.write_hypothesis(dummy_hypothesis, test_file)
        hypothesis = self.reader.read_hypothesis(test_file)

        assert dummy_hypothesis.number_of_regions == hypothesis.number_of_regions
        assert numpy.array_equal(dummy_hypothesis.x0_values, hypothesis.x0_values)
        assert dummy_hypothesis.x0_indices == hypothesis.x0_indices
        assert numpy.array_equal(dummy_hypothesis.e_values, hypothesis.e_values)
        assert dummy_hypothesis.e_indices == hypothesis.e_indices
        assert numpy.array_equal(dummy_hypothesis.w_values, hypothesis.w_values)
        assert dummy_hypothesis.w_indices == hypothesis.w_indices
        assert numpy.array_equal(dummy_hypothesis.lsa_propagation_strengths, hypothesis.lsa_propagation_strengths)
        assert numpy.array_equal(dummy_hypothesis.lsa_propagation_indices, hypothesis.lsa_propagation_indices)

    def test_read_model_configuration(self):
        test_file = os.path.join(get_temporary_folder(), "TestModelConfiguration.h5")
        dummy_mc = ModelConfiguration(x1EQ=numpy.array([2.0, 3.0, 1.0]), zmode=None, zEQ=numpy.array([3.0, 2.0, 1.0]),
                                      model_connectivity=numpy.array(
                                          [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [2.0, 2.0, 2.0]]),
                                      Ceq=numpy.array([1.0, 2.0, 3.0]))
        self.writer.write_model_configuration(dummy_mc, test_file)
        mc = self.reader.read_model_configuration(test_file)

        assert numpy.array_equal(dummy_mc.x1EQ, mc.x1EQ)
        assert numpy.array_equal(dummy_mc.zEQ, mc.zEQ)
        assert numpy.array_equal(dummy_mc.Ceq, mc.Ceq)
        assert numpy.array_equal(dummy_mc.model_connectivity, mc.model_connectivity)

    @classmethod
    def teardown_class(cls):
        remove_temporary_test_files()
