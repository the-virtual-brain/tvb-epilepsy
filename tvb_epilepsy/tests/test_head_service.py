import os
import numpy
from tvb_epilepsy.io.h5_reader import H5Reader
from tvb_epilepsy.base.model.vep.head import Head
from tvb_epilepsy.base.model.vep.sensors import Sensors
from tvb_epilepsy.base.model.vep.surface import Surface
from tvb_epilepsy.service.head_service import HeadService
from tvb_epilepsy.base.constants.configurations import FOLDER_LOGS, FOLDER_RES, FOLDER_FIGURES, IN_HEAD


class TestHeadService(object):
    head_service = HeadService()

    @classmethod
    def setup_class(cls):
        for direc in (FOLDER_LOGS, FOLDER_RES, FOLDER_FIGURES):
            if not os.path.exists(direc):
                os.makedirs(direc)

    @staticmethod
    def _prepare_dummy_head():
        reader = H5Reader()
        connectivity = reader.read_connectivity(os.path.join(IN_HEAD, "Connectivity.h5"))
        cort_surface = Surface([], [])
        seeg_sensors = Sensors(numpy.array(["sens1", "sens2"]), numpy.array([[0, 0, 0], [0, 1, 0]]))
        head = Head(connectivity, cort_surface, sensorsSEEG=seeg_sensors)

        return head

    def test_select_sensors_power(self):
        head = self._prepare_dummy_head()
        selected = self.head_service.select_sensors_power(head.sensorsSEEG[0], 0.4)

        # TODO: better checks
        assert isinstance(selected, list)

    def test_select_sensors_rois(self):
        head = self._prepare_dummy_head()
        selected = self.head_service.select_sensors_rois(head.sensorsSEEG[0], [0])

        # TODO: better checks
        assert isinstance(selected, list)

    def test_select_sensors_corr(self):
        head = self._prepare_dummy_head()
        selected = self.head_service.select_sensors_corr(head.sensorsSEEG[0], 0.1)

        # TODO: better checks
        assert isinstance(selected, list)

    @classmethod
    def teardown_class(cls):
        for direc in (FOLDER_LOGS, FOLDER_RES, FOLDER_FIGURES):
            for dir_file in os.listdir(direc):
                os.remove(os.path.join(os.path.abspath(direc), dir_file))
            os.removedirs(direc)
