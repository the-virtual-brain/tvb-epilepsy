#!/usr/bin/env python

import os
import numpy

from tvb_epilepsy.base.constants import DATA_MODE, TVB, DATA_CUSTOM, X0_DEF, E_DEF
from tvb_epilepsy.base.utils import initialize_logger, set_time_scales, assert_equal_objects, \
    write_object_to_h5_file, read_object_from_h5_file
from tvb_epilepsy.base.h5_model import object_to_h5_model
from tvb_epilepsy.custom.read_write import write_h5_model, read_hypothesis, hyp_attributes_dict, \
    read_simulation_settings, \
    epileptor_model_attributes_dict, simulation_settings_attributes_dict
from tvb_epilepsy.custom.readers_custom import CustomReader
from tvb_epilepsy.base.disease_hypothesis import DiseaseHypothesis
from tvb_epilepsy.base.model_configuration_service import ModelConfigurationService
from tvb_epilepsy.base.lsa_service import LSAService
from tvb_epilepsy.base.pse_service import PSE_Service, pse_from_hypothesis
from tvb_epilepsy.base.sensitivity_analysis_service import SensitivityAnalysisService, \
                                                           sensitivity_analysis_pse_from_hypothesis
from tvb_epilepsy.tvb_api.simulator_tvb import setup_simulation

if DATA_MODE is TVB:
    from tvb_epilepsy.tvb_api.readers_tvb import TVBReader as Reader
else:
    from tvb_epilepsy.custom.readers_custom import CustomReader as Reader

# --------------------------Simulation preparations-----------------------------------

# TODO: maybe use a custom Monitor class
fs = 2 * 4096.0
scale_time = 2.0
time_length = 10000.0
scale_fsavg = 2.0
report_every_n_monitor_steps = 10.0
(dt, fsAVG, sim_length, monitor_period, n_report_blocks) = \
    set_time_scales(fs=fs, dt=None, time_length=time_length, scale_time=scale_time, scale_fsavg=scale_fsavg,
                    report_every_n_monitor_steps=report_every_n_monitor_steps)

model_name = "EpileptorDP"

hpf_flag = False
hpf_low = max(16.0, 1000.0 / time_length)  # msec
hpf_high = min(250.0, fsAVG)


if __name__ == "__main__":

    logger = initialize_logger(__name__)

    data_folder = os.path.join(DATA_CUSTOM, 'Head')

    reader = Reader()

    logger.info("Reading from: " + data_folder)
    head = reader.read_head(data_folder)

    logger.info("Loaded Head " + str(head))
    logger.info("Loaded Connectivity " + str(head.connectivity))
    # We don't want any time delays for the moment
    head.connectivity.tract_lengths *= 0.0

    ep_name = "ep_test1"
    FOLDER_RES = os.path.join(data_folder, ep_name)
    logger.info("Loaded epileptogenicity from " + FOLDER_RES)

    if not isinstance(reader, CustomReader):
        reader = CustomReader()
    disease_values = reader.read_epileptogenicity(data_folder, name=ep_name)
    disease_indices, = numpy.where(disease_values > numpy.min([X0_DEF, E_DEF]))
    disease_values = disease_values[disease_indices]
    if disease_values.size > 1:
        inds_split = numpy.ceil(disease_values.size * 1.0 / 2).astype("int")
        x0_indices = disease_indices[:inds_split].tolist()
        e_indices = disease_indices[inds_split:].tolist()
        x0_values = disease_values[:inds_split].tolist()
        e_values = disease_values[inds_split:].tolist()
    else:
        x0_indices = disease_indices.tolist()
        x0_values = disease_values.tolist()
        e_indices = []
        e_values = []
    disease_indices = list(disease_indices)

    n_x0 = len(x0_indices)
    n_e = len(e_indices)
    n_disease = len(disease_indices)
    all_regions_indices = numpy.array(range(head.number_of_regions))
    healthy_indices = numpy.delete(all_regions_indices, disease_indices).tolist()
    n_healthy = len(healthy_indices)

    # This is an example of Excitability Hypothesis:
    hyp_x0 = DiseaseHypothesis(head.connectivity, excitability_hypothesis={tuple(disease_indices): disease_values},
                               epileptogenicity_hypothesis={}, connectivity_hypothesis={})

    # This is an example of Epileptogenicity Hypothesis:
    hyp_E = DiseaseHypothesis(head.connectivity, excitability_hypothesis={},
                              epileptogenicity_hypothesis={tuple(disease_indices): disease_values},
                              connectivity_hypothesis={})

    if len(e_indices) > 0:
        # This is an example of x0 mixed Excitability and Epileptogenicity Hypothesis:
        hyp_x0_E = DiseaseHypothesis(head.connectivity, excitability_hypothesis={tuple(x0_indices): x0_values},
                                     epileptogenicity_hypothesis={tuple(e_indices): e_values},
                                     connectivity_hypothesis={})
        hypotheses = (hyp_x0, hyp_E, hyp_x0_E)
    else:
        hypotheses = (hyp_x0, hyp_E)

    n_samples = 30

    for hyp in hypotheses:

        model_configuration_service = ModelConfigurationService(hyp.get_number_of_regions())
        filename = "model_configuration_service_"+ hyp.name + ".h5"
        write_h5_model(model_configuration_service.prepare_for_h5(), folder_name=FOLDER_RES, file_name=filename)
        assert_equal_objects(model_configuration_service,
                             read_object_from_h5_file(model_configuration_service, os.path.join(FOLDER_RES, filename),
                                                                 attributes_dict=None, add_overwrite_fields_dict=None))

        if hyp.type == "Epileptogenicity":
            model_configuration = model_configuration_service.configure_model_from_E_hypothesis(hyp)
        else:
            model_configuration = model_configuration_service.configure_model_from_hypothesis(hyp)
        filename = "model_configuration" + hyp.name + ".h5"
        write_h5_model(model_configuration.prepare_for_h5(), folder_name=FOLDER_RES, file_name=filename)
        assert_equal_objects(model_configuration,
                             read_object_from_h5_file(model_configuration, os.path.join(FOLDER_RES, filename),
                                                      attributes_dict=None, add_overwrite_fields_dict=None))

        lsa_service = LSAService(eigen_vectors_number=None, weighted_eigenvector_sum=True)
        filename = "lsa_service" + hyp.name + ".h5"
        write_h5_model(lsa_service.prepare_for_h5(), folder_name=FOLDER_RES, file_name=filename)
        assert_equal_objects(lsa_service,
                             read_object_from_h5_file(lsa_service, os.path.join(FOLDER_RES, filename),
                                                      attributes_dict=None, add_overwrite_fields_dict=None))

        lsa_hypothesis = lsa_service.run_lsa(hyp, model_configuration)
        file_name = lsa_hypothesis.name + ".h5"
        write_h5_model(lsa_hypothesis.prepare_for_h5(), folder_name=FOLDER_RES, file_name=file_name)
        assert_equal_objects(lsa_hypothesis, read_hypothesis(path=os.path.join(FOLDER_RES, file_name)))

        # --------------Parameter Search Exploration (PSE)-------------------------------

        print "running PSE LSA..."
        pse_results = pse_from_hypothesis(lsa_hypothesis, n_samples, half_range=0.1,
                                          global_coupling=[{"indices": all_regions_indices}],
                                          healthy_regions_parameters=[{"name": "x0", "indices": healthy_indices}],
                                          model_configuration=model_configuration,
                                          model_configuration_service=model_configuration_service,
                                          lsa_service=lsa_service)[0]

        filename = "PSE_LSA_results_" + lsa_hypothesis.name + ".h5"
        write_h5_model(object_to_h5_model(pse_results), FOLDER_RES, filename)
        assert_equal_objects(pse_results, read_object_from_h5_file(pse_results, os.path.join(FOLDER_RES, filename),
                                                      attributes_dict=None, add_overwrite_fields_dict=None))

        # --------------Sensitivity Analysis Parameter Search Exploration (PSE)-------------------------------

        print "running sensitivity analysis PSE LSA..."
        sa_results, pse_results = \
            sensitivity_analysis_pse_from_hypothesis(lsa_hypothesis, n_samples, method="delta", half_range=0.1,
                                                     global_coupling=[{"indices": all_regions_indices,
                                                                       "bounds":
                                                                           [0.0, 2 *
                                                                            model_configuration_service.K_unscaled[
                                                                                0]]}],
                                                     healthy_regions_parameters=
                                                     [{"name": "x0", "indices": healthy_indices}],
                                                     model_configuration=model_configuration,
                                                     model_configuration_service=model_configuration_service,
                                                     lsa_service=lsa_service)

        filename = "SA_PSE_LSA_results_" + lsa_hypothesis.name + ".h5"
        write_h5_model(object_to_h5_model(pse_results), FOLDER_RES, filename)
        assert_equal_objects(pse_results, read_object_from_h5_file(pse_results, os.path.join(FOLDER_RES, filename),
                                                                   attributes_dict=None,
                                                                   add_overwrite_fields_dict=None))

        filename = "SA_LSA_results_" + lsa_hypothesis.name + ".h5"
        write_h5_model(object_to_h5_model(sa_results), FOLDER_RES, filename)
        assert_equal_objects(sa_results, read_object_from_h5_file(sa_results, os.path.join(FOLDER_RES, filename),
                                                                   attributes_dict=None,
                                                                   add_overwrite_fields_dict=None))

        # --------------Simulation-------------------------------
        simulator_instance = setup_simulation(model_configuration, head.connectivity, dt, sim_length, monitor_period,
                                              model_name, scale_time=scale_time, noise_intensity=10 ** -8)
        simulator_instance.config_simulation()
        model = simulator_instance.model

        filename = lsa_hypothesis.name + "_sim_settings.h5"
        write_h5_model(simulator_instance.prepare_for_h5(), folder_name=FOLDER_RES,
                       file_name=filename)

        model2, simulator_instance2 = read_simulation_settings(path=os.path.join(FOLDER_RES, filename),
                                                         output="object", hypothesis=hyp)

        assert_equal_objects(model, model2, epileptor_model_attributes_dict[model2._ui_name])
        assert_equal_objects(simulator_instance, simulator_instance2, simulation_settings_attributes_dict)

