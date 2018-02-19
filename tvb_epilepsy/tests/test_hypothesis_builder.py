import os

from tvb_epilepsy.base.constants.config import Config
from tvb_epilepsy.service.hypothesis_builder import HypothesisBuilder


class TestHypothesisBuilder(object):
    config = Config()
    ep = "ep_l_frontal_complex"

    @classmethod
    def setup_class(cls):
        for direc in (cls.config.out.FOLDER_LOGS, cls.config.out.FOLDER_RES, cls.config.out.FOLDER_FIGURES):
            if not os.path.exists(direc):
                os.makedirs(direc)

    def test_build_empty_hypothesis(self):
        hypo_builder = HypothesisBuilder()
        hypo = hypo_builder._build_hypothesis()

        assert hypo.name == "_Hypothesis"
        assert hypo.number_of_regions == 0
        assert len(hypo.x0_indices) == 0
        assert len(hypo.x0_values) == 0
        assert len(hypo.e_indices) == 0
        assert len(hypo.e_values) == 0
        assert len(hypo.w_indices) == 0
        assert len(hypo.w_values) == 0
        assert len(hypo.lsa_propagation_indices) == 0
        assert len(hypo.lsa_propagation_strengths) == 0

    def test_build_hypothesis_by_user_preferences(self):
        hypo_builder = HypothesisBuilder().set_nr_of_regions(76).set_x0_indices([1, 2, 3]).set_x0_values(
            [1, 1, 1]).set_e_indices([10, 11]).set_e_values([1, 1]).set_normalize(0.90)
        hypo = hypo_builder._build_hypothesis()

        assert hypo.name == "Excitability_Epileptogenicity_Hypothesis"
        assert hypo.number_of_regions == 76
        assert len(hypo.x0_indices) == 3
        assert len(hypo.x0_values) == 3
        assert len(hypo.e_indices) == 2
        assert len(hypo.e_values) == 2
        assert len(hypo.w_indices) == 0
        assert len(hypo.w_values) == 0
        assert len(hypo.lsa_propagation_indices) == 0
        assert len(hypo.lsa_propagation_strengths) == 0

    def test_build_lsa_hypothesis(self):
        hypo_builder = HypothesisBuilder().set_nr_of_regions(76).set_x0_indices([1, 2]).set_x0_values([1, 1])
        hypo = hypo_builder._build_hypothesis()

        lsa_hypo = hypo_builder.set_attributes_based_on_hypothesis(hypo).set_lsa_propagation_indices(
            [3, 4]).set_lsa_propagation_strengths([0.5, 1])._build_hypothesis()

        assert lsa_hypo.name == "Excitability_HypothesisLSA"
        assert lsa_hypo.number_of_regions == 76
        assert len(lsa_hypo.x0_indices) == 2
        assert len(lsa_hypo.x0_values) == 2
        assert len(lsa_hypo.e_indices) == 0
        assert len(lsa_hypo.e_values) == 0
        assert len(lsa_hypo.w_indices) == 0
        assert len(lsa_hypo.w_values) == 0
        assert len(lsa_hypo.lsa_propagation_indices) == 2
        assert len(lsa_hypo.lsa_propagation_strengths) == 2

    def test_build_hypothesis_from_file_excitability(self):
        hypo_builder = HypothesisBuilder().set_nr_of_regions(76)
        hypo = hypo_builder.build_hypothesis_from_file(self.ep)

        assert hypo.name == "Excitability_Hypothesis"
        assert not len(hypo.x0_indices) == 0
        assert not len(hypo.x0_values) == 0
        assert len(hypo.e_indices) == 0
        assert len(hypo.e_values) == 0
        assert len(hypo.w_indices) == 0
        assert len(hypo.w_values) == 0
        assert len(hypo.lsa_propagation_indices) == 0
        assert len(hypo.lsa_propagation_strengths) == 0

    def test_build_hypothesis_from_file_epileptogenicity(self):
        hypo_builder = HypothesisBuilder().set_nr_of_regions(76)
        hypo = hypo_builder.build_hypothesis_from_file(self.ep, [55, 56, 57, 58, 59, 60, 61])

        assert hypo.name == "Epileptogenicity_Hypothesis"
        assert not len(hypo.e_indices) == 0
        assert not len(hypo.e_values) == 0
        assert len(hypo.x0_indices) == 0
        assert len(hypo.x0_values) == 0
        assert len(hypo.w_indices) == 0
        assert len(hypo.w_values) == 0
        assert len(hypo.lsa_propagation_indices) == 0
        assert len(hypo.lsa_propagation_strengths) == 0

    def test_build_hypothesis_from_file_mixed(self):
        hypo_builder = HypothesisBuilder().set_nr_of_regions(76)
        hypo = hypo_builder.build_hypothesis_from_file(self.ep, [55, 56])

        assert hypo.name == "Excitability_Epileptogenicity_Hypothesis"
        assert not len(hypo.e_indices) == 0
        assert not len(hypo.e_values) == 0
        assert not len(hypo.x0_indices) == 0
        assert not len(hypo.x0_values) == 0
        assert len(hypo.w_indices) == 0
        assert len(hypo.w_values) == 0
        assert len(hypo.lsa_propagation_indices) == 0
        assert len(hypo.lsa_propagation_strengths) == 0

    @classmethod
    def teardown_class(cls):
        for direc in (cls.config.out.FOLDER_LOGS, cls.config.out.FOLDER_RES, cls.config.out.FOLDER_FIGURES):
            for dir_file in os.listdir(direc):
                os.remove(os.path.join(os.path.abspath(direc), dir_file))
            os.removedirs(direc)
