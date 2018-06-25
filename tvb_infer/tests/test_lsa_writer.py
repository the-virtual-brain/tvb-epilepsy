import os
from tvb_infer.tvb_lsa.lsa_writer import LSAH5Writer
from tvb_infer.tvb_lsa.lsa_service import LSAService
from tvb_infer.tests.base import BaseTest


class TestCustomH5writer(BaseTest):
    writer = LSAH5Writer()

    def test_write_lsa_service(self):
        test_file = os.path.join(self.config.out.FOLDER_TEMP, "TestLSAService.h5")
        dummy_lsa_service = LSAService()

        assert not os.path.exists(test_file)

        self.writer.write_lsa_service(dummy_lsa_service, test_file)

        assert os.path.exists(test_file)

