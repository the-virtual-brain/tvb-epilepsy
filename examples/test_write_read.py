#!/usr/bin/env python

from main_vep import main_vep
from tvb_epilepsy.base.utils.data_structures_utils import assert_equal_objects


if __name__ == "__main__":
    import os
    from tvb_epilepsy.base.constants.configurations import FOLDER_RES
    from tvb_epilepsy.base.h5_model import read_h5_model
    from tvb_epilepsy.base.model.statistical_models.stochastic_parameter import generate_stochastic_parameter
    from tvb_epilepsy.base.model.statistical_models.sde_statistical_model import SDEStatisticalModel
    parameters={}
    parameter = generate_stochastic_parameter("x1init",
                                                  low=0.0,
                                                  high=1.0,
                                                  p_shape=(),
                                                  probability_distribution="normal",
                                                  optimize=False)
    parameter.write_to_h5(FOLDER_RES, "test_parameter_model.h5")
    parameter2 = read_h5_model(os.path.join(FOLDER_RES, "test_parameter_model.h5")).convert_from_h5_model()
    print(assert_equal_objects(parameter, parameter2))
    parameters.update({parameter.name: parameter})

    parameter = generate_stochastic_parameter("zinit",
                                                  low=0.0,
                                                  high=1.0,
                                                  p_shape=(),
                                                  probability_distribution="normal",
                                                  optimize=False)
    parameters.update({parameter.name: parameter})
    stats_model = SDEStatisticalModel("sde_vep", parameters)
    stats_model.write_to_h5(FOLDER_RES, "test_stats_model.h5")
    stats_model2 = read_h5_model(os.path.join(FOLDER_RES, "test_stats_model.h5")).convert_from_h5_model()
    print(assert_equal_objects(stats_model, stats_model2))
    main_vep(test_write_read=True, pse_flag=True, sa_pse_flag=True, sim_flag=True)

