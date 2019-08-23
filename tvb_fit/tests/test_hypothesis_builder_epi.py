
from tvb_fit.tests.base import BaseTest
from tvb_fit.tvb_epilepsy.service.hypothesis_builder import HypothesisBuilder


class TestHypothesisBuilder(BaseTest):
    ep = "ep_l_frontal_complex"

    def test_build_empty_hypothesis(self):
        hypo_builder = HypothesisBuilder(config=self.config)
        hypo = hypo_builder.build_hypothesis()

        assert hypo.name == "Hypothesis"
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
        hypo_builder = HypothesisBuilder(76, self.config).set_x0_hypothesis(
            [1, 2, 3], [1, 1, 1]).set_e_hypothesis([10, 11], [1, 1]).set_normalize(0.90)
        hypo = hypo_builder.build_hypothesis()

        assert hypo.name == "e_x0_Hypothesis"
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
        hypo_builder = HypothesisBuilder(76, self.config).set_x0_hypothesis([1, 2], [1, 1])
        hypo = hypo_builder.build_hypothesis()

        lsa_hypo = hypo_builder.set_attributes_based_on_hypothesis(hypo).set_lsa_propagation(
            [3, 4], [0.5, 1]).build_lsa_hypothesis()

        assert lsa_hypo.name == "x0_Hypothesis"
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
        hypo_builder = HypothesisBuilder(76, self.config)
        hypo = hypo_builder.build_hypothesis_from_file(self.ep)

        assert hypo.name == "x0_Hypothesis"
        assert not len(hypo.x0_indices) == 0
        assert not len(hypo.x0_values) == 0
        assert len(hypo.e_indices) == 0
        assert len(hypo.e_values) == 0
        assert len(hypo.w_indices) == 0
        assert len(hypo.w_values) == 0
        assert len(hypo.lsa_propagation_indices) == 0
        assert len(hypo.lsa_propagation_strengths) == 0

    def test_build_hypothesis_from_file_epileptogenicity(self):
        hypo_builder = HypothesisBuilder(76, self.config)
        hypo = hypo_builder.build_hypothesis_from_file(self.ep, [55, 56, 57, 58, 59, 60, 61])

        assert hypo.name == "e_Hypothesis"
        assert not len(hypo.e_indices) == 0
        assert not len(hypo.e_values) == 0
        assert len(hypo.x0_indices) == 0
        assert len(hypo.x0_values) == 0
        assert len(hypo.w_indices) == 0
        assert len(hypo.w_values) == 0
        assert len(hypo.lsa_propagation_indices) == 0
        assert len(hypo.lsa_propagation_strengths) == 0

    def test_build_hypothesis_from_file_mixed(self):
        hypo_builder = HypothesisBuilder(76, self.config)
        hypo = hypo_builder.build_hypothesis_from_file(self.ep, [55, 56])

        assert hypo.name == "e_x0_Hypothesis"
        assert not len(hypo.e_indices) == 0
        assert not len(hypo.e_values) == 0
        assert not len(hypo.x0_indices) == 0
        assert not len(hypo.x0_values) == 0
        assert len(hypo.w_indices) == 0
        assert len(hypo.w_values) == 0
        assert len(hypo.lsa_propagation_indices) == 0
        assert len(hypo.lsa_propagation_strengths) == 0
