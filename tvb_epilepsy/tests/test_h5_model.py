import numpy
from copy import deepcopy

from tvb_epilepsy.base.utils.data_structures_utils import assert_equal_objects
from tvb_epilepsy.io.h5_model import read_h5_model
from tvb_epilepsy.io.h5_reader import H5Reader
from tvb_epilepsy.io.h5_writer import H5Writer
from tvb_epilepsy.io.tvb_data_reader import TVBReader
from tvb_epilepsy.service.hypothesis_builder import HypothesisBuilder
from tvb_epilepsy.tests.base import BaseTest


class TestH5Model(BaseTest):
    def test_h5_model_write_read(self):
        reader = TVBReader() if self.config.input.IS_TVB_MODE else H5Reader()
        writer = H5Writer()

        head = reader.read_head(self.config.input.HEAD)

        empty_hypothesis = HypothesisBuilder()._build_mixed_hypothesis()
        x0_indices = [20]
        x0_values = [0.9]
        e_indices = [70]
        e_values = [0.9]

        hyp_x0_E = HypothesisBuilder().set_nr_of_regions(head.connectivity.number_of_regions
                                                         )._build_mixed_hypothesis(e_values, e_indices,
                                                                                   x0_values, x0_indices)

        obj = {"hyp_x0_E": hyp_x0_E,
               "test_dict": {"list0": ["l00", 1,
                                       {"d020": "a",
                                        "d021": [True, False, numpy.inf, numpy.nan, None, [], (), {}, ""]}]}}

        writer.write_generic(obj, self.config.out.FOLDER_RES, "test_h5_model.h5")

        h5_model1 = read_h5_model(self.config.out.FOLDER_RES + "/test_h5_model.h5")
        obj1 = h5_model1.convert_from_h5_model(deepcopy(obj))

        assert_equal_objects(obj, obj1)

        h5_model2 = read_h5_model(self.config.out.FOLDER_RES + "/test_h5_model.h5")
        obj2 = h5_model2.convert_from_h5_model(obj={"DiseaseHypothesis": empty_hypothesis})

        assert_equal_objects(obj, obj2)
