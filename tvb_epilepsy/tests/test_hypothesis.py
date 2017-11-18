import os
from copy import deepcopy

from tvb_epilepsy.base.constants.configurations import FOLDER_RES, FOLDER_LOGS, FOLDER_FIGURES
from tvb_epilepsy.base.h5_model import read_h5_model
from tvb_epilepsy.base.utils.data_structures_utils import assert_equal_objects
from tvb_epilepsy.base.model.disease_hypothesis import DiseaseHypothesis
from tvb_epilepsy.tests.base import get_temporary_files_path, remove_temporary_test_files


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

    def test_h5_conversion(self):
        nr_of_regions = 76
        x0_indices = [20]
        x0_values = [0.9]

        lsa_hypothesis = DiseaseHypothesis(nr_of_regions, excitability_hypothesis={tuple(x0_indices): x0_values},
                                           epileptogenicity_hypothesis={}, connectivity_hypothesis={},
                                           propagation_indices=[0], propagation_strenghts=[18])

        hypothesis_template = DiseaseHypothesis(nr_of_regions)

        folder = get_temporary_files_path()
        file_name = "hypo.h5"

        lsa_hypothesis.write_to_h5(folder, file_name)

        lsa_hypothesis1 = read_h5_model(os.path.join(folder, file_name)).convert_from_h5_model(
            obj=deepcopy(lsa_hypothesis))
        assert_equal_objects(lsa_hypothesis, lsa_hypothesis1)

        lsa_hypothesis2 = read_h5_model(os.path.join(folder, file_name)).convert_from_h5_model(
            children_dict=hypothesis_template)
        assert_equal_objects(lsa_hypothesis, lsa_hypothesis2)

    @classmethod
    def teardown_class(cls):
        remove_temporary_test_files()
        for direc in (FOLDER_LOGS, FOLDER_RES, FOLDER_FIGURES):
            for dir_file in os.listdir(direc):
                os.remove(os.path.join(os.path.abspath(direc), dir_file))
            os.removedirs(direc)
