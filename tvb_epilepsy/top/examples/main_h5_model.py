# coding=utf-8

import os
import numpy as np
from copy import deepcopy
from tvb_epilepsy.base.constants.config import Config
from tvb_epilepsy.io.h5_model import read_h5_model
from tvb_epilepsy.base.utils.data_structures_utils import assert_equal_objects
from tvb_epilepsy.base.utils.log_error_utils import initialize_logger
from tvb_epilepsy.io.h5_writer import H5Writer
from tvb_epilepsy.service.hypothesis_builder import HypothesisBuilder
from tvb_epilepsy.io.tvb_data_reader import TVBReader
from tvb_epilepsy.io.h5_reader import H5Reader

logger = initialize_logger(__name__)


def main_h5_model(config=Config()):
    # -------------------------------Reading data-----------------------------------
    reader = TVBReader() if config.input.IS_TVB_MODE else H5Reader()
    writer = H5Writer()
    logger.info("Reading from: %s", config.input.HEAD)
    head = reader.read_head(config.input.HEAD)

    empty_hypothesis = HypothesisBuilder().build_hypothesis()
    x0_indices = [20]
    x0_values = [0.9]
    e_indices = [70]
    e_values = [0.9]

    hyp_x0_E = HypothesisBuilder(head.connectivity.number_of_regions).set_x0_hypothesis(x0_indices,
                                                                                        x0_values).set_e_hypothesis(
        e_indices, e_values).build_hypothesis()

    obj = {"hyp_x0_E": hyp_x0_E,
           "test_dict": {"list0": ["l00", 1,
                                   {"d020": "a",
                                    "d021": [True, False, np.inf, np.nan, None, [], (), {}, ""]}]}}
    logger.info("Original object: %s", obj)

    logger.info("Writing object to h5 file...")
    writer.write_generic(obj, os.path.join(config.out.FOLDER_RES, "test_h5_model.h5"))

    h5_model1 = read_h5_model(config.out.FOLDER_RES + "/test_h5_model.h5")
    obj1 = h5_model1.convert_from_h5_model(deepcopy(obj))

    if assert_equal_objects(obj, obj1):
        logger.info("Read identical object: %s", obj1)
    else:
        logger.error("Comparison failed!: %s", obj1)

    h5_model2 = read_h5_model(config.out.FOLDER_RES + "/test_h5_model.h5")
    obj2 = h5_model2.convert_from_h5_model(obj={"DiseaseHypothesis": empty_hypothesis})

    if assert_equal_objects(obj, obj2):
        logger.info("Read object as dictionary: %s", obj2)
    else:
        logger.error("Comparison failed!: %s", obj2)


if __name__ == "__main__":
    main_h5_model()
