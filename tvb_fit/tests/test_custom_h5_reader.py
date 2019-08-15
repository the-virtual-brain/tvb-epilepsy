import os
import numpy

from tvb_fit.tests.base import BaseTest
from tvb_fit.base.config import InputConfig
from tvb_fit.base.model.simulation_settings import SimulationSettings

from tvb_head.model.sensors import SensorTypes

from tvb_io.h5_reader import H5Reader
from tvb_io.h5_writer import H5Writer


class TestCustomH5Reader(BaseTest):
    reader = H5Reader()
    writer = H5Writer()
    in_head = InputConfig().HEAD
    not_existent_file = "NotExistent.h5"

    def test_read_connectivity(self):
        connectivity = self.reader.read_connectivity(os.path.join(self.in_head, "Connectivity.h5"))

        assert connectivity is not None
        assert connectivity.number_of_regions == 76

    def test_read_surface(self):
        surface = self.reader.read_surface(os.path.join(self.in_head, "CorticalSurface.h5"))

        assert surface is not None
        assert surface.vertices.shape[0] == 16

    def test_read_surface_not_existent_file(self):
        surface = self.reader.read_surface(os.path.join(self.in_head, self.not_existent_file))

        assert surface is None

    def test_read_region_mapping(self):
        region_mapping = self.reader.read_region_mapping(os.path.join(self.in_head, "RegionMapping.h5"))

        assert isinstance(region_mapping, numpy.ndarray)
        assert region_mapping.size == 16

    def test_read_region_mapping_not_existent_file(self):
        region_mapping = self.reader.read_region_mapping(os.path.join(self.in_head, self.not_existent_file))

        assert isinstance(region_mapping, numpy.ndarray)
        assert region_mapping.size == 0

    def test_read_volume_mapping(self):
        volume_mapping = self.reader.read_volume_mapping(os.path.join(self.in_head, "VolumeMapping.h5"))

        assert isinstance(volume_mapping, numpy.ndarray)
        assert volume_mapping.shape == (6, 5, 4)

    def test_read_volume_mapping_not_existent_file(self):
        volume_mapping = self.reader.read_volume_mapping(os.path.join(self.in_head, self.not_existent_file))

        assert isinstance(volume_mapping, numpy.ndarray)
        assert volume_mapping.shape == (0,)

    def test_sensors_of_type(self):
        sensors_file = os.path.join(self.in_head, "SensorsSEEG_20.h5")
        sensors = self.reader.read_sensors_of_type(sensors_file, SensorTypes.TYPE_SEEG, "SEEG")

        assert sensors is not None
        assert sensors.number_of_sensors == 20

    def test_sensors_of_type_not_existent_file(self):
        sensors_file = os.path.join(self.in_head, self.not_existent_file)
        sensors = self.reader.read_sensors_of_type(sensors_file, SensorTypes.TYPE_SEEG, "SEEG")

        assert len(sensors) is 0

    def test_read_sensors(self):
        sensors_seeg, sensors_eeg, sensors_meg = self.reader.read_sensors(self.in_head)

        assert len(sensors_seeg) > 0
        assert len(sensors_eeg) == 0
        assert len(sensors_meg) == 0
        assert sensors_seeg["SensorsSEEG_20"] is not None
        assert sensors_seeg["SensorsSEEG_20"].number_of_sensors == 20

    def test_read_simulation_settings(self):
        test_file = os.path.join(self.config.out.FOLDER_TEMP, "TestSimSettings.h5")
        dummy_sim_settings = SimulationSettings()
        self.writer.write_simulation_settings(dummy_sim_settings, test_file)

        sim_settings = self.reader.read_simulation_settings(test_file)

        assert dummy_sim_settings.integration_step == sim_settings.integration_step
        assert dummy_sim_settings.simulation_length == sim_settings.simulation_length
        assert dummy_sim_settings.integrator_type == sim_settings.integrator_type
        assert dummy_sim_settings.noise_type == sim_settings.noise_type
        assert dummy_sim_settings.noise_ntau == sim_settings.noise_ntau
        assert dummy_sim_settings.noise_intensity == sim_settings.noise_intensity
        assert dummy_sim_settings.noise_seed == sim_settings.noise_seed
        assert dummy_sim_settings.monitor_type == sim_settings.monitor_type
        assert dummy_sim_settings.monitor_sampling_period == sim_settings.monitor_sampling_period
        assert dummy_sim_settings.monitor_vois.size == sim_settings.monitor_vois.size
