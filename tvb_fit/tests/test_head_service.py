from tvb_fit.service.head_service import HeadService
from tvb_fit.tests.base import BaseTest


class TestHeadService(BaseTest):
    head_service = HeadService()
    #TODO: lots of methods in HeadService not used
    # def test_select_sensors_power(self):
    #     head = self._prepare_dummy_head()
    #     print(head.sensorsSEEG.keys())
    #     selected = self.head_service.select_sensors_power(head.sensorsSEEG["SensorsSEEG_20"], 0.4)
    #
    #     # TODO: better checks
    #     assert isinstance(selected, list)
    #
    # def test_select_sensors_rois(self):
    #     head = self._prepare_dummy_head()
    #     selected = self.head_service.select_sensors_rois(head.sensorsSEEG["SensorsSEEG_20"], [0])
    #
    #     # TODO: better checks
    #     assert isinstance(selected, list)
    #
    # def test_select_sensors_corr(self):
    #     head = self._prepare_dummy_head()
    #     selected = self.head_service.select_sensors_corr(head.sensorsSEEG["SensorsSEEG_20"], 0.1)
    #
    #     # TODO: better checks
    #     assert isinstance(selected, list)
