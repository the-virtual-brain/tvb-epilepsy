import os
from tvb_infer.base.config import InputConfig
from tvb_infer.tvb_lsa.lsa_reader import LSAH5Reader
from tvb_infer.tvb_lsa.lsa_writer import LSAH5Writer
from tvb_infer.tvb_lsa.lsa_service import LSAService
from tvb_infer.tests.base import BaseTest


class TestCustomH5Reader(BaseTest):
    reader = LSAH5Reader()
    writer = LSAH5Writer()
    in_head = InputConfig().HEAD
    not_existent_file = "NotExistent.h5"

    def test_read_lsa_service(self):
        test_file = os.path.join(self.config.out.FOLDER_TEMP, "TestLSAService.h5")
        dummy_lsa_service = LSAService()
        self.writer.write_lsa_service(dummy_lsa_service, test_file)

        lsa_service = self.reader.read_lsa_service(test_file)

        assert dummy_lsa_service.eigen_vectors_number_selection == lsa_service.eigen_vectors_number_selection
        assert dummy_lsa_service.eigen_vectors_number == lsa_service.eigen_vectors_number
        assert dummy_lsa_service.eigen_values == lsa_service.eigen_values
        assert dummy_lsa_service.eigen_vectors == lsa_service.eigen_vectors
        assert dummy_lsa_service.weighted_eigenvector_sum == lsa_service.weighted_eigenvector_sum
        assert dummy_lsa_service.normalize_propagation_strength == lsa_service.normalize_propagation_strength
