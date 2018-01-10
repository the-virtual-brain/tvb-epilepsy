import os
import numpy
from tvb_epilepsy.base.model.vep.connectivity import Connectivity
from tvb_epilepsy.base.model.vep.head import Head
from tvb_epilepsy.base.model.vep.sensors import Sensors
from tvb_epilepsy.base.model.vep.surface import Surface
from tvb_epilepsy.io.h5.writer_custom import CustomH5Writer
from tvb_epilepsy.tests.base import remove_temporary_test_files, get_temporary_folder


class TestCustomH5writer(object):
    writer = CustomH5Writer()

    dummy_connectivity = Connectivity("", numpy.array([[1.0, 2.0, 3.0], [2.0, 3.0, 1.0], [3.0, 2.0, 1.0]]),
                                      numpy.array([[4, 5, 6], [5, 6, 4], [6, 4, 5]]), labels=["a", "b", "c"],
                                      centres=numpy.array([1.0, 2.0, 3.0]))
    dummy_surface = Surface(numpy.array([[1, 2, 3], [2, 3, 1], [3, 1, 2]]), numpy.array([[0, 1, 2]]))
    dummy_sensors = Sensors(numpy.array(["sens1", "sens2"]), numpy.array([[0, 0, 0], [0, 1, 0]]),
                            gain_matrix=numpy.array([[1, 2, 3], [2, 3, 4]]))

    def _prepare_dummy_head(self):
        return Head(self.dummy_connectivity, self.dummy_surface, sensorsSEEG=[self.dummy_sensors])

    def test_write_connectivity(self):
        test_file = os.path.join(get_temporary_folder(), "TestConnectivity.h5")

        self.writer.write_connectivity(self.dummy_connectivity, test_file)

        assert os.path.exists(test_file)

    def test_write_connectivity_with_normalized_weigths(self):
        test_file = os.path.join(get_temporary_folder(), "TestConnectivityNorm.h5")

        connectivity = self.dummy_connectivity
        connectivity.normalized_weights = numpy.array([1, 2, 3])
        self.writer.write_connectivity(connectivity, test_file)

        assert os.path.exists(test_file)

    def test_write_surface(self):
        test_file = os.path.join(get_temporary_folder(), "TestSurface.h5")

        self.writer.write_surface(self.dummy_surface, test_file)

        assert os.path.exists(test_file)

    def test_write_sensors(self):
        test_file = os.path.join(get_temporary_folder(), "TestSensors.h5")

        self.writer.write_sensors(self.dummy_sensors, test_file)

        assert os.path.exists(test_file)

    def test_write_head(self):
        test_folder = os.path.join(get_temporary_folder(), "test_head")

        head = self._prepare_dummy_head()
        self.writer.write_head(head, test_folder)

        assert os.path.exists(test_folder)
        assert len(os.listdir(test_folder)) >= 3

    @classmethod
    def teardown_class(cls):
        remove_temporary_test_files()
