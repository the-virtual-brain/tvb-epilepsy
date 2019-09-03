# coding=utf-8

import os
from copy import deepcopy
import numpy as np

from tvb_fit.tvb_epilepsy.base.constants.config import Config
from tvb_fit.tvb_epilepsy.base.computation_utils.equilibrium_computation import calc_eq_z, calc_x0
from tvb_fit.tvb_epilepsy.base.model.epileptor_models import EpileptorDP2D
from tvb_fit.tvb_epilepsy.base.model.timeseries import Timeseries
from tvb_fit.tvb_epilepsy.service.simulator.simulator_builder import build_simulator_TVB_realistic, \
    build_simulator_TVB_fitting, build_simulator_TVB_default, build_simulator_TVB_paper

from tvb_scripts.utils.log_error_utils import initialize_logger, warning
from tvb_scripts.utils.data_structures_utils import isequal_string
from tvb_scripts.utils.analyzers_utils import interval_scaling
from tvb_scripts.utils.file_utils import wildcardit, move_overwrite_files_to_folder_with_wildcard
from tvb_scripts.model.virtual_head.sensors import SensorTypes
from tvb_scripts.service.timeseries_service import TimeseriesService


LOG = initialize_logger(__name__)


def rescale_x1eq(model_config, rescale_x1eq):
    x1eq_min = np.min(model_config.x1eq)
    model_config.x1eq = \
        interval_scaling(model_config.x1eq, min_targ=x1eq_min, max_targ=rescale_x1eq,
                         min_orig=x1eq_min, max_orig=np.max(model_config.x1eq))
    zeq = calc_eq_z(model_config.x1eq, model_config.yc, model_config.Iext1, "2d",
                    np.zeros(model_config.zeq.shape), model_config.slope, model_config.a,
                    model_config.b, model_config.d)
    model_config.x0 = calc_x0(model_config.x1eq, zeq, model_config.K, model_config.connectivity,
                              model_config.zmode, z_pos=True, shape=zeq.shape, calc_mode="non_symbol")
    return model_config


def configure_simulator(input_modelconfig, connectivity, x1eq_rescale=None, sim_type="paper",
                        sim_model_path="", writer=None, logger=None, config=Config()):
    if logger is None:
        initialize_logger(__name__, config.out.FOLDER_LOGS)
    # Choose model
    # Available models beyond the TVB Epileptor (they all encompass optional variations from the different papers):
    # EpileptorDP: similar to the TVB Epileptor + optional variations,
    # EpileptorDP2D: reduced 2D model, following Proix et all 2014 +optional variations,
    # EpleptorDPrealistic: starting from the TVB Epileptor + optional variations, but:
    #      -x0, Iext1, Iext2, slope and K become noisy state variables,
    #      -Iext2 and slope are coupled to z, g, or z*g in order for spikes to appear before seizure,
    #      -multiplicative correlated noise is also used
    # Optional variations:
    model_config = deepcopy(input_modelconfig)
    if x1eq_rescale:
        model_config = rescale_x1eq(model_config, x1eq_rescale)
    logger.info("\n\nConfiguring simulation...")
    if isequal_string(sim_type, "realistic"):
        simulator, sim_settings = build_simulator_TVB_realistic(model_config, connectivity)
    elif isequal_string(sim_type, "fitting"):
        simulator, sim_settings = build_simulator_TVB_fitting(model_config, connectivity)
    elif isequal_string(sim_type, "paper"):
        simulator, sim_settings = build_simulator_TVB_paper(model_config, connectivity)
    else:
        simulator, sim_settings = build_simulator_TVB_default(model_config, connectivity)
    if writer:
        if not os.path.isdir(os.path.dirname(sim_model_path)):
            sim_model_path = os.path.join(config.out.FOLDER_RES, "SimModel" + simulator.model._ui_name + ".h5")
        else:
            sim_model_path = sim_model_path.replace("Model", "Model" + simulator.model._ui_name)
        writer.write_simulator_model(simulator.model, sim_model_path, simulator.connectivity.number_of_regions)
    return simulator, sim_settings


def compute_and_write_seeg(source_timeseries, simulator, sensors_dict,  ts_epi_filepath="", writer=None, **sim_params):
    ts_service = TimeseriesService()
    fsAVG = 1000.0 / source_timeseries.sample_period
    if isinstance(simulator.model, EpileptorDP2D):
        sim_params["hpf_flag"] = False
    if sim_params.get("hpf_flag", False):
        sim_params["hpf_low"] = max(sim_params.get("hpf_low", 10.0),
                                     1000.0 / (source_timeseries.end_time - source_timeseries.start_time))
        sim_params["hpf_high"] = min(fsAVG / 2.0 - 10.0, sim_params.get("hpf_high"))

    seeg_data = \
        ts_service.compute_seeg(source_timeseries, sensors_dict, sum_mode=sim_params.get("seeg_gain_mode", "lin"))
    for s_name, seeg in seeg_data.items():
        if sim_params.get("hpf_flag", False):
            seeg_data[s_name] = ts_service.filter(seeg, sim_params["hpf_low"], sim_params["hpf_high"],
                                                  mode='bandpass', order=3)
        else:
            seeg_data[s_name] = seeg
        if writer and os.path.isfile(ts_epi_filepath):
            try:
                writer.write_ts_seeg_epi(seeg_data[s_name], source_timeseries.sample_period,
                                         ts_epi_filepath, sensors_name=s_name)
            except:
                warning("Failed to write simulated SEEG timeseries to epi file %s!" % ts_epi_filepath)
    return seeg_data


def compute_seeg_and_write_ts_to_h5(timeseries, simulator, sensors_dict,
                                    ts_filepath="", ts_epi_filepath="", writer=None, **sim_params):
    source_timeseries = timeseries.get_source()
    if writer:
        try:
            writer.write_ts(timeseries, timeseries.sample_period, ts_filepath)
        except:
            warning("Failed to write simulated timeseries to file %s!" % ts_filepath)
        if os.path.isdir(os.path.dirname(ts_epi_filepath)):
            try:
                writer.write_ts_epi(timeseries, timeseries.sample_period, ts_epi_filepath, source_timeseries)
            except:
                warning("Failed to write simulated timeseries to epi file %s!" % ts_epi_filepath)
    if sensors_dict is not None:
        seeg_ts_all = compute_and_write_seeg(source_timeseries, simulator, sensors_dict, ts_epi_filepath, writer,
                                             **sim_params)
    else:
        seeg_ts_all = None
    return timeseries, seeg_ts_all


def simulate(simulator, head,  all_disease_indices=[], compute_seeg=True, plot_spectral_raster=False,
             ts_filepath="", ts_epi_filepath="", sim_figsfolder="", title_prefix="",
             reader=None, writer=None, plotter=None, logger=LOG, config=Config(), **sim_params):
    seeg = []
    if reader and os.path.isfile(ts_filepath):
        logger.info("\n\nLoading previously simulated time series from file: " + ts_filepath)
        sim_output = reader.read_timeseries(ts_filepath)
        seeg = TimeseriesService().compute_seeg(sim_output.get_source(),
                                                head.get_sensors(s_type=SensorTypes.TYPE_SEEG.value),
                                                sum_mode=sim_params.get("seeg_gain_mode", "lin"))
    else:
        logger.info("\n\nSimulating %s..." % sim_params.get("type", "paper"))
        sim_output, status = simulator.launch_simulation(report_every_n_monitor_steps=100, timeseries=Timeseries)
        if not status:
            logger.warning("\nSimulation failed!")
        else:
            time = np.array(sim_output.time).astype("f")
            logger.info("\n\nSimulated signal return shape: %s", sim_output.shape)
            logger.info("Time: %s - %s", time[0], time[-1])
            if compute_seeg:
                sensors = head.get_sensors(SensorTypes.TYPE_SEEG.value)
            else:
                sensors = None
            sim_output, seeg = compute_seeg_and_write_ts_to_h5(sim_output, simulator, sensors,
                                                               ts_filepath, ts_epi_filepath, writer, **sim_params)

    sim_ts = {"source": sim_output, "seeg": seeg}

    if plotter:
        # Plot results
        plotter.plot_simulated_timeseries(sim_output, simulator.model, all_disease_indices,
                                          seeg_dict=seeg, spectral_raster_plot=plot_spectral_raster,
                                          spectral_options=sim_params.get("spectral_options",
                                                                          {"log_scale": True}),
                                          title_prefix=title_prefix)
        if os.path.isdir(sim_figsfolder) and (sim_figsfolder != config.out.FOLDER_FIGURES):
            move_overwrite_files_to_folder_with_wildcard(sim_figsfolder,
                                                         os.path.join(config.out.FOLDER_FIGURES,
                                                                      wildcardit("Sim")))
    return sim_ts


def from_model_configuration_to_simulation(model_configuration, head, hypothesis,
                                           x1eq_rescale=None, sim_type="paper", compute_seeg=True,
                                           seeg_gain_mode="lin", hpf_flag=False, hpf_low=10.0, hpf_high=512.0,
                                           plot_spectral_raster=False,
                                           ts_file="", sim_model_path="", title_prefix="",
                                           reader=None, writer=None, plotter=None, logger=LOG, config=Config()):
    if not os.path.isdir(os.path.dirname(sim_model_path)):
        sim_model_path = os.path.join(config.out.FOLDER_RES, "SimModel_%s_%s.h5" % (hypothesis.name, sim_type))
    simulator, sim_settings = configure_simulator(model_configuration, head.connectivity, x1eq_rescale=x1eq_rescale,
                                                  sim_type=sim_type, sim_model_path=sim_model_path,
                                                  writer=writer, logger=logger, config=config)
    sim_params = {"seeg_gain_mode": seeg_gain_mode, "hpf_flag": hpf_flag, "hpf_low": hpf_low,
                  "hpf_high": hpf_high}
    if not os.path.isdir(os.path.dirname(ts_file)):
        ts_file = os.path.join(config.out.FOLDER_RES, "SimTS_%s_%s.h5" % (hypothesis.name, sim_type))
    ts_epi_filepath = ts_file.replace(".h5", "_epi.h5")
    sim_ts = simulate(simulator, head, hypothesis.all_disease_indices,
                      compute_seeg=compute_seeg, plot_spectral_raster=plot_spectral_raster,
                      ts_filepath=ts_file, ts_epi_filepath=ts_epi_filepath,
                      sim_figsfolder=config.out.FOLDER_FIGURES, title_prefix=title_prefix,
                      reader=reader, writer=writer, plotter=plotter, logger=logger, config=config, **sim_params)
    return sim_ts, simulator
