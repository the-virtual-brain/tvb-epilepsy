import os
import numpy

from tvb_fit.tests.base import BaseTest
from tvb_io.h5_writer import H5Writer


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

    def test_write_dictionary(self):
        test_file = os.path.join(self.config.out.FOLDER_TEMP, "TestDict.h5")
        dummy_dict = dict(
            {"n_loops": 96, "params_indices": numpy.array([0, 1, 2]), "params_names": numpy.array(["x0", "z", "x0"]),
             "params_samples": numpy.array([[0.0, 0.1, 0.2], [0.3, 0.0, 0.1], [0.2, 0.3, 0.0]]), "task": "LSA"})

        assert not os.path.exists(test_file)

        self.writer.write_dictionary(dummy_dict, test_file)

        assert os.path.exists(test_file)

    @classmethod
    def teardown_class(cls):
        head_dir = os.path.join(cls.config.out.FOLDER_TEMP, "test_head")
        if os.path.exists(head_dir):
            for dir_file in os.listdir(head_dir):
                os.remove(os.path.join(os.path.abspath(head_dir), dir_file))
            os.rmdir(head_dir)
        BaseTest.teardown_class()
