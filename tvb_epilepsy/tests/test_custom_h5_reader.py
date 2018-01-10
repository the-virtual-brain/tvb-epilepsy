import os

import numpy

from tvb_epilepsy.base.constants.configurations import DATA_TEST
from tvb_epilepsy.base.model.vep.sensors import Sensors
from tvb_epilepsy.io.h5.reader_custom import CustomH5Reader

head_dir = "head2"


class TestCustomH5Reader():
    reader = CustomH5Reader()
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
