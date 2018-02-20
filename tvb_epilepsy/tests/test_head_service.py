from tvb_epilepsy.service.head_service import HeadService
from tvb_epilepsy.base.constants.config import Config
from tvb_epilepsy.tests.base import BaseTest


class TestHeadService(BaseTest):
    head_service = HeadService()
    config = Config()

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
