import os
from tvb_epilepsy.base.constants.config import Config
from tvb_epilepsy.service.hypothesis_builder import HypothesisBuilder


class TestHypothesis(object):
    config = Config()

    @classmethod
    def setup_class(cls):
        for direc in (cls.config.out.FOLDER_LOGS, cls.config.out.FOLDER_RES, cls.config.out.FOLDER_FIGURES):
            if not os.path.exists(direc):
                os.makedirs(direc)

    def test_create(self):
        x0_indices = [20]
        x0_values = [0.9]
        hyp = HypothesisBuilder().set_nr_of_regions(76).build_excitability_hypothesis(x0_values, x0_indices)
        assert x0_indices == hyp.x0_indices
        assert x0_values == hyp.x0_values

    @classmethod
    def teardown_class(cls):
        for direc in (cls.config.out.FOLDER_LOGS, cls.config.out.FOLDER_RES, cls.config.out.FOLDER_FIGURES):
            for dir_file in os.listdir(direc):
                os.remove(os.path.join(os.path.abspath(direc), dir_file))
            os.removedirs(direc)
