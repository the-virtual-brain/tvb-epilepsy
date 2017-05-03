#!/usr/bin/env python

import os
import numpy
from tvb_epilepsy.base.utils import get_logger, set_time_scales, assert_equal_objects, \
    write_object_to_h5_file, read_object_from_h5_file
from tvb_epilepsy.base.hypothesis import Hypothesis
from tvb_epilepsy.custom.read_write import write_h5_model, read_hypothesis, hyp_attributes_dict, \
    read_simulation_settings, \
    epileptor_model_attributes_dict, simulation_settings_attributes_dict
from tvb_epilepsy.custom.readers_custom import CustomReader
from tvb_epilepsy.tvb_api.epileptor_models import model_build_dict
from tvb_epilepsy.tvb_api.simulator_tvb import setup_simulation

if __name__ == "__main__":
    logger = get_logger(__name__)

    logger.info("Reading from custom")
    # data_folder = os.path.join("/WORK/Episense/root-episense/trunk/demo-data", 'Head_JUNCH')
    data_folder = os.path.join("/WORK/Episense/trunk/demo-data", 'Head_JUNCH')
    reader = CustomReader()

    logger.info("We will be reading from location " + data_folder)
    head = reader.read_head(data_folder)
    logger.info("Loaded Head " + str(head))
    logger.info("Loaded Connectivity " + str(head.connectivity))

    hypothesis = Hypothesis(head.number_of_regions, head.connectivity.weights, "EP Hypothesis")

    epi_name = "ep"
    epi_complete_path = os.path.join(data_folder, epi_name)

    epileptogenicity = reader.read_epileptogenicity(data_folder, epi_name)
    logger.info("Loaded epileptogenicity from " + epi_complete_path)

    epi_indices = numpy.arange(0, 88, 1)
    hypothesis.configure_e_hypothesis(epi_indices, epileptogenicity, epi_indices)

    hypothesis_name = "hypo.h5"

    hypo_h5_model = hypothesis.prepare_for_h5()
    write_h5_model(hypo_h5_model, epi_complete_path, hypothesis_name)

    hypothesis2 = read_hypothesis(path=os.path.join(epi_complete_path, hypothesis_name), output="object",
                                  update_hypothesis=True)

    assert_equal_objects(hypothesis, hypothesis2, hyp_attributes_dict)

    hypothesis2 = read_hypothesis(path=os.path.join(epi_complete_path, hypothesis_name), output="dict")

    assert_equal_objects(hypothesis, hypothesis2, hyp_attributes_dict)

    model_name = 'EpileptorDP'
    model = model_build_dict[model_name](hypothesis)

    (fs, dt, fsAVG, scale_time, sim_length, monitor_period,
     n_report_blocks, hpf_fs, hpf_low, hpf_high) = set_time_scales(fs=2 * 4096.0, dt=None, time_length=1000.0,
                                                                   scale_time=2.0, scale_fsavg=2.0,
                                                                   hpf_low=None, hpf_high=None,
                                                                   report_every_n_monitor_steps=10.0)

    (simulator_instance, sim_settings, vois) = setup_simulation(model, dt, sim_length, monitor_period,
                                                                scale_time=scale_time,
                                                                noise_instance=None, noise_intensity=10 ** -8,
                                                                monitor_expressions=None, monitors_instance=None,
                                                                variables_names=None)

    sim, sim_settings = simulator_instance.config_simulation(head, hypothesis, settings=sim_settings)

    sim_h5_model = simulator_instance.prepare_for_h5(sim_settings)
    write_h5_model(sim_h5_model, epi_complete_path, hypothesis.name + "sim_settings.h5")

    model2, sim_settings2 = read_simulation_settings(
        path=os.path.join(epi_complete_path, hypothesis.name + "sim_settings.h5"),
        output="object", hypothesis=hypothesis)

    assert_equal_objects(model, model2, epileptor_model_attributes_dict[model2._ui_name])
    assert_equal_objects(sim_settings, sim_settings2, simulation_settings_attributes_dict)

    model2, sim_settings2 = read_simulation_settings(
        path=os.path.join(epi_complete_path, hypothesis.name + "sim_settings.h5"), output="dict")

    assert_equal_objects(model, model2, epileptor_model_attributes_dict[model2["_ui_name"]])
    assert_equal_objects(sim_settings, sim_settings2, simulation_settings_attributes_dict)

    # TODO: use write_h5_model
    write_object_to_h5_file(model2, os.path.join(epi_complete_path, hypothesis.name + "model_dict.h5"))
    model3 = read_object_from_h5_file(dict(), os.path.join(epi_complete_path, hypothesis.name + "model_dict.h5"),
                                      attributes_dict=None, add_overwrite_fields_dict=None)
    assert_equal_objects(model2, model3)
