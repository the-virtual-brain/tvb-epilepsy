import os
import numpy as np
from tvb_epilepsy.base.constants.config import Config
from tvb_epilepsy.base.utils.log_error_utils import initialize_logger
from tvb_epilepsy.base.utils.data_structures_utils import ensure_list, isequal_string
from tvb_epilepsy.base.computations.analyzers_utils import filter_data
from tvb_epilepsy.base.model.vep.sensors import Sensors
from tvb_epilepsy.io.h5_reader import H5Reader
from tvb_epilepsy.io.h5_writer import H5Writer
from tvb_epilepsy.plot.plotter import Plotter
from tvb_epilepsy.base.epileptor_models import EpileptorDP2D
from tvb_epilepsy.service.simulator.simulator_builder import build_simulator_TVB_realistic, \
                                   build_simulator_TVB_fitting, build_simulator_TVB_default, build_simulator_TVB_paper

logger = initialize_logger(__name__)


def prepare_vois_ts_dict(vois, data):
    # Pack results into a dictionary:
    vois_ts_dict = dict()
    for idx_voi, voi in enumerate(vois):
        vois_ts_dict[voi] = data[:, idx_voi, :].astype('f')
    return vois_ts_dict


def compute_seeg_and_write_ts_h5_file(folder, filename, model, vois_ts_dict, dt, time_length, hpf_flag=False,
                                      hpf_low=10.0, hpf_high=256.0, sensors_list=[], save_flag=True):
    fsAVG = 1000.0 / dt
    sensors_list = ensure_list(sensors_list)
    # Optionally high pass filter, and compute SEEG:
    if isinstance(model, EpileptorDP2D):
        raw_data = np.dstack(
            [vois_ts_dict["x1"], vois_ts_dict["z"], vois_ts_dict["x1"]])
        vois_ts_dict["lfp"] = vois_ts_dict["x1"]
        idx_proj = -1
        for sensor in sensors_list:
            if isinstance(sensor, Sensors):
                idx_proj += 1
                sensor_name = sensor.s_type + '%d' % idx_proj
                vois_ts_dict[sensor_name] = vois_ts_dict['x1'].dot(sensor.gain_matrix.T)
                vois_ts_dict[sensor_name] -= np.min(vois_ts_dict[sensor_name])
                vois_ts_dict[sensor_name] /= np.max(vois_ts_dict[sensor_name])
    else:
        vois_ts_dict["lfp"] = vois_ts_dict["x2"] - vois_ts_dict["x1"]
        raw_data = np.dstack([vois_ts_dict["x1"], vois_ts_dict["z"], vois_ts_dict["x2"]])
        if hpf_flag:
            hpf_low = max(hpf_low, 1000.0 / time_length)
            hpf_high = min(fsAVG / 2.0 - 10.0, hpf_high)
        idx_proj = -1
        for sensor in sensors_list:
            if isinstance(sensor, Sensors):
                idx_proj += 1
                sensor_name = sensor.s_type + '%d' % idx_proj
                vois_ts_dict[sensor_name] = vois_ts_dict['lfp'].dot(sensor.gain_matrix.T)
                if hpf_flag:
                    for i in range(vois_ts_dict[sensor_name].shape[1]):
                        vois_ts_dict[sensor_name][:, i] = \
                            filter_data(vois_ts_dict[sensor_name][:, i], fsAVG, hpf_low, hpf_high)
                vois_ts_dict[sensor_name] -= np.min(vois_ts_dict[sensor_name])
                vois_ts_dict[sensor_name] /= np.max(vois_ts_dict[sensor_name])

    if save_flag:
        h5_writer = H5Writer()
        h5_writer.write_ts_epi(raw_data, dt, os.path.join(folder, filename), vois_ts_dict["lfp"])
        # Write files:
        if idx_proj > -1:
            for i_sensor, sensor in enumerate(sensors_list):
                h5_writer.write_ts_seeg_epi(vois_ts_dict[sensor.s_type + '%d' % i_sensor], dt,
                                            os.path.join(folder, filename))
    return vois_ts_dict


def from_model_configuration_to_simulation(model_configuration, head, lsa_hypothesis,
                                           sim_type="realistic", dynamical_model="EpileptorDP2D", ts_file=None,
                                           plot_flag=True, config=Config()):
    # Choose model
    # Available models beyond the TVB Epileptor (they all encompass optional variations from the different papers):
    # EpileptorDP: similar to the TVB Epileptor + optional variations,
    # EpileptorDP2D: reduced 2D model, following Proix et all 2014 +optional variations,
    # EpleptorDPrealistic: starting from the TVB Epileptor + optional variations, but:
    #      -x0, Iext1, Iext2, slope and K become noisy state variables,
    #      -Iext2 and slope are coupled to z, g, or z*g in order for spikes to appear before seizure,
    #      -multiplicative correlated noise is also used
    # Optional variations:
    if dynamical_model is "EpileptorDP2D":
        spectral_raster_plot = False
        trajectories_plot = True
    else:
        spectral_raster_plot = False  # "lfp"
        trajectories_plot = False

    # ------------------------------Simulation--------------------------------------
    logger.info("\n\nConfiguring simulation...")
    if isequal_string(sim_type, "realistic"):
        sim, sim_settings, dynamical_model = build_simulator_TVB_realistic(model_configuration, head.connectivity)
    elif isequal_string(sim_type, "fitting"):
        sim, sim_settings, dynamical_model = build_simulator_TVB_fitting(model_configuration, head.connectivity)
    elif isequal_string(sim_type, "paper"):
        sim, sim_settings, dynamical_model = build_simulator_TVB_paper(model_configuration, head.connectivity)
    else:
        sim, sim_settings, dynamical_model = build_simulator_TVB_default(model_configuration, head.connectivity)

    writer = H5Writer()
    writer.write_simulator_model(sim.model, sim.connectivity.number_of_regions,
                                 os.path.join(config.out.FOLDER_RES, dynamical_model._ui_name + "_model.h5"))

    vois_ts_dict = {}
    if ts_file is not None and os.path.isfile(ts_file):
        logger.info("\n\nLoading previously simulated time series from file: " + ts_file)
        vois_ts_dict = H5Reader().read_dictionary(ts_file)
    else:
        logger.info("\n\nSimulating...")
        ttavg, tavg_data, status = sim.launch_simulation(report_every_n_monitor_steps=100)
        if not status:
            logger.warning("\nSimulation failed!")
        else:
            time = np.array(ttavg, dtype='float32').flatten()
            output_sampling_time = np.mean(np.diff(time))
            tavg_data = tavg_data[:, :, :, 0]
            logger.info("\n\nSimulated signal return shape: %s", tavg_data.shape)
            logger.info("Time: %s - %s", time[0], time[-1])
            logger.info("Values: %s - %s", tavg_data.min(), tavg_data.max())
            # Variables of interest in a dictionary:
            vois_ts_dict = prepare_vois_ts_dict(dynamical_model.variables_of_interest, tavg_data)
            vois_ts_dict['time'] = time
            vois_ts_dict['time_units'] = 'msec'
            vois_ts_dict = compute_seeg_and_write_ts_h5_file(config.out.FOLDER_RES, dynamical_model._ui_name + "_ts.h5",
                                                             sim.model,
                                                             vois_ts_dict, output_sampling_time,
                                                             sim_settings.simulated_period,
                                                             hpf_flag=True, hpf_low=10.0, hpf_high=512.0,
                                                             sensors_list=head.sensorsSEEG, save_flag=True)
            if isinstance(ts_file, basestring):
                writer.write_dictionary(vois_ts_dict, os.path.join(os.path.dirname(ts_file), os.path.basename(ts_file)))
    if plot_flag and len(vois_ts_dict) > 0:
        # Plot results
        Plotter(config).plot_sim_results(sim.model, lsa_hypothesis.lsa_propagation_indices, vois_ts_dict,
                                         sensorsSEEG=head.sensorsSEEG, hpf_flag=False,
                                         trajectories_plot=trajectories_plot,
                                         spectral_raster_plot=spectral_raster_plot, log_scale=True,
                                         region_labels=head.connectivity.region_labels)
    return vois_ts_dict
