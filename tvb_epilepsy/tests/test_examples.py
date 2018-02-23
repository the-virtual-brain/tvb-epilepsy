from tvb_epilepsy.tests.base import BaseTest
from tvb_epilepsy.top.examples.main_sensitivity_analysis import main_sensitivity_analysis
from tvb_epilepsy.top.examples.main_pse import main_pse
from tvb_epilepsy.top.examples.main_h5_model import main_h5_model
from tvb_epilepsy.top.examples.main_sampling_service import main_sampling_service


class TestExamples(BaseTest):

    def test_main_sensitivity_analysis(self):
        main_sensitivity_analysis(self.config)

    # def test_main_pse(self):
    #     main_pse(self.config)

    def test_h5_model(self):
        main_h5_model(self.config)

    # def test_main_sampling_service(self):
    #     main_sampling_service(self.config)
