"""
Entry point for working with VEP
"""
import os
import warnings

import numpy
from scipy.io import savemat

from tvb_epilepsy.base.constants import FOLDER_RES, FOLDER_FIGURES, SAVE_FLAG, SHOW_FLAG, SIMULATION_MODE, \
    TVB, DATA_MODE, VOIS, DATA_CUSTOM
from tvb_epilepsy.base.disease_hypothesis import DiseaseHypothesis
from tvb_epilepsy.base.model_configuration_service import ModelConfigurationService
from tvb_epilepsy.base.lsa_service import LSAService
from tvb_epilepsy.base.plot_tools import plot_nullclines_eq, plot_sim_results, plot_hypothesis_equilibrium_and_lsa
from tvb_epilepsy.base.utils import initialize_logger, set_time_scales, calculate_projection, filter_data
from tvb_epilepsy.custom.read_write import write_h5_model, write_ts_epi, write_ts_seeg_epi
from tvb_epilepsy.tvb_api.epileptor_models import EpileptorDP, EpileptorDP2D
from tvb_epilepsy.custom.simulator_custom import EpileptorModel

if DATA_MODE is TVB:
    from tvb_epilepsy.tvb_api.readers_tvb import TVBReader as Reader
else:
    from tvb_epilepsy.custom.readers_custom import CustomReader as Reader

if SIMULATION_MODE is TVB:
    from tvb_epilepsy.tvb_api.simulator_tvb import setup_simulation
else:
    from tvb_epilepsy.custom.simulator_custom import setup_simulation


def prepare_vois_ts_dict(vois):
    # Pack results into a dictionary, high pass filter, and compute SEEG
    vois_ts_dict = dict()
    for idx_voi, voi in enumerate(vois):
        vois_ts_dict[voi] = tavg_data[:, idx_voi, :].astype('f')

    return vois_ts_dict


def prepare_ts_and_seeg_h5_file(model, projections, vois_ts_dict, hpf_flag, hpf_low, hpf_high, fsAVG, dt):
    if isinstance(model, EpileptorDP2D):
        raw_data = numpy.dstack(
            [vois_ts_dict["x1"], vois_ts_dict["z"], vois_ts_dict["x1"]])
        lfp_data = vois_ts_dict["x1"]

        for idx_proj, proj in enumerate(projections):
            vois_ts_dict['seeg%d' % idx_proj] = vois_ts_dict['z'].dot(proj.T)

    else:
        if isinstance(model, EpileptorModel):
            lfp_data = vois_ts_dict["x2"] - vois_ts_dict["x1"]

        else:
            lfp_data = vois_ts_dict["lfp"]

        raw_data = numpy.dstack(
            [vois_ts_dict["x1"], vois_ts_dict["z"], vois_ts_dict["x2"]])

        for idx_proj, proj in enumerate(projections):
            vois_ts_dict['seeg%d' % idx_proj] = vois_ts_dict['lfp'].dot(proj.T)
            if hpf_flag:
                for i in range(vois_ts_dict['seeg'].shape[0]):
                    vois_ts_dict['seeg_hpf%d' % i][:, i] = filter_data(
                        vois_ts_dict['seeg%d' % i][:, i], hpf_low, hpf_high,
                        fsAVG)

    write_ts_epi(raw_data, dt, lfp_data, path=os.path.join(FOLDER_RES, hypothesis.name + "_ep_ts.h5"))

    for i in range(len(projections)):
        write_ts_seeg_epi(vois_ts_dict['seeg%d' % i], dt,
                          path=os.path.join(FOLDER_RES, hypothesis.name + "_ep_ts.h5"))


if __name__ == "__main__":

    logger = initialize_logger(__name__)

    # -------------------------------Reading data-----------------------------------

    data_folder = os.path.join(DATA_CUSTOM, 'Head_TREC')

    reader = Reader()

    logger.info("Reading from: " + data_folder)
    head = reader.read_head(data_folder)

    # --------------------------Hypothesis and LSA-----------------------------------

    SEIZURE_THRESHOLD = 0.5

    # This is an example of x0 Hypothesis

    x0_indices = range(head.connectivity.number_of_regions)
    x0_values = numpy.zeros((len(x0_indices),), dtype='float32')

    x0_indices_to_put_random_values = [20]
    x0_values[x0_indices_to_put_random_values] = 0.85

    hypothesis = DiseaseHypothesis(head.connectivity, x0_values, x0_indices, [], [], [],
                                   "Excitability", "x0_Hypothesis")

    all_regions_one = numpy.ones((hypothesis.get_number_of_regions(),), dtype=numpy.float32)

    model_configuration_service = ModelConfigurationService()
    model_configuration = model_configuration_service.configure_model_from_hypothesis(hypothesis)

    # NOTES:
    # Why not overwrite the input hypothesis with the output one?
    # Anyway, the x0/E values and indices are not overwritten. Only the output is (propagation strength and indices).
    lsa_service = LSAService()
    lsa_hypothesis = lsa_service.run_lsa(hypothesis, model_configuration)

    plot_hypothesis_equilibrium_and_lsa(lsa_hypothesis, model_configuration, "x0_hypo")

    write_h5_model(hypothesis.prepare_for_h5(), folder_name=FOLDER_RES, file_name=hypothesis.name + ".h5")
    write_h5_model(lsa_hypothesis.prepare_for_h5(), folder_name=FOLDER_RES, file_name=lsa_hypothesis.name + ".h5")
    write_h5_model(model_configuration.prepare_for_h5(), folder_name=FOLDER_RES, file_name="Config.h5")

    # ------------------------------Simulation--------------------------------------

    # TODO: maybe use a custom Monitor class
    fs = 2 * 4096.0
    scale_time = 2.0
    time_length = 3000.0
    scale_fsavg = 2.0
    report_every_n_monitor_steps = 10.0
    (dt, fsAVG, sim_length, monitor_period, n_report_blocks) = set_time_scales(fs=fs, dt=None,
                                                                               time_length=time_length,
                                                                               scale_time=scale_time,
                                                                               scale_fsavg=scale_fsavg,
                                                                               report_every_n_monitor_steps=report_every_n_monitor_steps)
    model_name = "EpileptorDP"

    # We don't want any time delays for the moment
    head.connectivity.tract_lengths *= 0.0

    simulator_instance = setup_simulation(model_configuration, head.connectivity, dt, sim_length, monitor_period,
                                          model_name, scale_time=scale_time, noise_intensity=10 ** -8)

    simulator_instance.config_simulation()
    ttavg, tavg_data, status = simulator_instance.launch_simulation(n_report_blocks)

    write_h5_model(simulator_instance.prepare_for_h5(), folder_name=FOLDER_RES,
                   file_name=hypothesis.name + "_sim_settings.h5")

    if not status:
        warnings.warn("Simulation failed!")

    else:
        tavg_data = tavg_data[:, :, :, 0]

        vois = VOIS[model_name]

        # Compute projections
        sensorsSEEG = []
        projections = []
        for sensors, projection in head.sensorsSEEG.iteritems():
            if projection is None:
                continue
            else:
                projection = calculate_projection(sensors, head.connectivity)
                head.sensorsSEEG[sensors] = projection
                print projection.shape
                sensorsSEEG.append(sensors)
                projections.append(projection)

        hpf_flag = False
        hpf_low = max(16.0, 1000.0 / time_length)  # msec
        hpf_high = min(250.0, fsAVG)

        model = simulator_instance.model

        logger.info("\nSimulated signal return shape: %s", tavg_data.shape)
        logger.info("Time: %s - %s", scale_time * ttavg[0], scale_time * ttavg[-1])
        logger.info("Values: %s - %s", tavg_data.min(), tavg_data.max())

        time = scale_time * numpy.array(ttavg, dtype='float32')
        dt = numpy.min(numpy.diff(time))

        vois_ts_dict = prepare_vois_ts_dict(vois)

        prepare_ts_and_seeg_h5_file(model, projections, vois_ts_dict, hpf_flag, hpf_low, hpf_high, fsAVG, dt)

        vois_ts_dict['time'] = time

        # Plot results
        seizure_indices = lsa_hypothesis.get_seizure_indices(SEIZURE_THRESHOLD)
        plot_nullclines_eq(model_configuration, head.connectivity.region_labels,
                           special_idx=seizure_indices,
                           model=str(model.nvar) + "d", zmode=model.zmode,
                           figure_name="Nullclines and equilibria", save_flag=SAVE_FLAG,
                           show_flag=SHOW_FLAG, figure_dir=FOLDER_FIGURES)
        plot_sim_results(model, seizure_indices,
                         hypothesis.name, head, vois_ts_dict, sensorsSEEG, hpf_flag)

        # Save results
        vois_ts_dict['time_units'] = 'msec'
        savemat(os.path.join(FOLDER_RES, hypothesis.name + "_ts.mat"), vois_ts_dict)
