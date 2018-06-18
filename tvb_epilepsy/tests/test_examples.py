import os
from tvb_epilepsy.tests.base import BaseTest
from tvb_epilepsy.top.examples.vep_study import vep_study
from tvb_epilepsy.top.examples.main_vep import main_vep
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

    def test_main_vep_default(self):
        main_vep(self.config, ep_name="ep_l_frontal_complex", ep_indices=[1])

    # def test_main_vep_everything(self):
    #     main_vep(self.config, sa_pse_flag=True)

    def test_vep_study(self):
        vep_study()

    # def test_main_fitting_default(self):
    #     main_fit_sim_hyplsa()

    @classmethod
    def teardown_class(cls):
        head_dir = os.path.join(cls.config.out.FOLDER_RES, "Head")
        if os.path.exists(head_dir):
            for dir_file in os.listdir(head_dir):
                os.remove(os.path.join(os.path.abspath(head_dir), dir_file))
            os.rmdir(head_dir)
        BaseTest.teardown_class()
