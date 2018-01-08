import os
from tvb_epilepsy.base.model.vep.sensors import Sensors
from tvb_epilepsy.custom.readers_custom import CustomReader

data_dir = "data"


class TestReadersCustom():
    reader = CustomReader()

    def test_read_sensors_of_type_seeg(self):
        sensors_file = os.path.join(data_dir, "SensorsSEEG_20.h5")
        sensors = self.reader.read_sensors_of_type(sensors_file, Sensors.TYPE_SEEG)

        assert sensors is not None
        assert sensors.number_of_sensors == 20

    def test_read_all_sensors(self):
        sensors_seeg, sensors_eeg, sensors_meg = self.reader.read_sensors(data_dir)

        assert len(sensors_seeg) > 0
        assert len(sensors_eeg) == 0
        assert len(sensors_meg) == 0
        assert sensors_seeg[0] is not None
        assert sensors_seeg[0].number_of_sensors == 20

    def test_read_head(self):
        head = self.reader.read_head(data_dir)

        assert head.connectivity.number_of_regions == 76
        assert len(head.sensorsEEG) == 0
        assert len(head.sensorsMEG) == 0
        assert len(head.sensorsSEEG) > 0
        assert head.sensorsSEEG[0] is not None
        assert head.sensorsSEEG[0].number_of_sensors == 20
