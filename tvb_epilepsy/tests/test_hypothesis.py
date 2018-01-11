import os
from tvb_epilepsy.base.constants.configurations import FOLDER_RES, FOLDER_LOGS, FOLDER_FIGURES
from tvb_epilepsy.base.model.disease_hypothesis import DiseaseHypothesis


class TestHypothesis():
    @classmethod
    def setup_class(cls):
        for direc in (FOLDER_LOGS, FOLDER_RES, FOLDER_FIGURES):
            if not os.path.exists(direc):
                os.makedirs(direc)

    def test_create(self):
        x0_indices = [20]
        x0_values = [0.9]
        hyp = DiseaseHypothesis(76, excitability_hypothesis={tuple(x0_indices): x0_values},
                                epileptogenicity_hypothesis={}, connectivity_hypothesis={})
        assert x0_indices == hyp.x0_indices
        assert x0_values == hyp.x0_values

    @classmethod
    def teardown_class(cls):
        for direc in (FOLDER_LOGS, FOLDER_RES, FOLDER_FIGURES):
            for dir_file in os.listdir(direc):
                os.remove(os.path.join(os.path.abspath(direc), dir_file))
            os.removedirs(direc)
