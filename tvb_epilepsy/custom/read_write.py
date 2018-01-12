"""
    Python Demo for reading and writing
"""

import ntpath
import os
import h5py
import numpy

from tvb_epilepsy.base.utils.log_error_utils import initialize_logger, warning, raise_value_error, raise_error
from tvb_epilepsy.base.utils.file_utils import change_filename_or_overwrite, print_metadata, write_metadata, \
    read_object_from_h5_file
# TODO: solve problems with setting up a logger
from tvb_epilepsy.base.simulators import SimulationSettings
from tvb_epilepsy.service.epileptor_model_factory import model_build_dict

PATIENT_VIRTUAL_HEAD = "/WORK/episense/episense-root/trunk/demo-data/Head_TREC"

logger = initialize_logger("log_" + ntpath.split(PATIENT_VIRTUAL_HEAD)[1])

KEY_TYPE = "EPI_Type"
KEY_VERSION = "EPI_Version"
KEY_DATE = "EPI_Last_update"
KEY_NODES = "Number_of_nodes"
KEY_SENSORS = "Number_of_sensors"
KEY_MAX = "Max_value"
KEY_MIN = "Min_value"
KEY_CHANNELS = "Number_of_channels"
KEY_SV = "Number_of_state_variables"
KEY_STEPS = "Number_of_steps"
KEY_SAMPLING = "Sampling_period"
KEY_START = "Start_time"

# Attributes to be read or written for hypothesis object and files:
hyp_attributes_dict = {"Hypothesis name": "name", "Model Epileptogenicity": "e_values",
                       "Pathological Excitability": "x0_values",
                       "LSA Propagation Strength": "lsa_ps", "x1 Equilibria": "x1EQ",
                       "z Equilibria": "zEQ", "Afferent coupling at equilibrium": "Ceq",
                       "Connectivity": "weights", "Permittivity Coupling": "K", "Iext1": "Iext1",
                       "yc": "yc", "Critical x0_values": "x0cr", "x0_values scaling": "rx0",
                       "EZ hypothesis": "seizure_indices",
                       "x1EQcr": "x1EQcr", "x1LIN": "x1LIN", "x1SQ": "x1SQ",
                       "lsa_eigvals": "lsa_eigvals", "lsa_eigvects": "lsa_eigvects", "lsa_ps_tot": "lsa_ps_tot"}

# Attributes to be read or written for Epileptor models object and files:
epileptor_attributes_dict = {"model.name": "_ui_name", "model.a": "a", "model.b": "b", "model.c": "c", "model.d": "d",
                             "model.r": "r", "model.s": "s", "model.x0_values": "x0_values", "model.Iext": "Iext",
                             "model.slope": "slope", "model.Iext2": "Iext2", "model.tau": "tau", "model.aa": "aa",
                             "model.Kvf": "Kvf", "model.Kf": "Kf", "model.Ks": "Ks", "model.tt": "tt"}

epileptorDP_attributes_dict = {"model.name": "_ui_name", "model.yc": "yc", "model.x0_values": "x0_values",
                               "model.Iext1": "Iext1",
                               "model.slope": "slope", "model.Iext2": "Iext2",
                               "model.tau2": "tau2", "model.Kvf": "Kvf", "model.Kf": "Kf", "model.K": "K",
                               "model.tau1": "tau1", "model.zmode": "zmode"}

epileptorDPrealistic_attributes_dict = {"model.name": "_ui_name", "model.yc": "yc", "model.x0_values": "x0_values",
                                        "model.Iext1": "Iext1", "model.slope": "slope", "model.Iext2": "Iext2",
                                        "model.tau2": "tau2", "model.Kvf": "Kvf", "model.Kf": "Kf", "model.K": "K",
                                        "model.tau1": "tau1", "model.zmode": "zmode", "model.pmode": "pmode"}

epileptorDP2D_attributes_dict = {"model.name": "_ui_name", "model.yc": "yc", "model.x0_values": "x0_values",
                                 "model.x0cr": "x0cr",
                                 "model.r": "r", "model.Iext1": "Iext1", "model.slope": "slope", "model.Kvf": "Kvf",
                                 "model.K": "K", "model.tau1": "tau1", "model.zmode": "zmode"}

epileptor_model_attributes_dict = {"Epileptor": epileptor_attributes_dict, "CustomEpileptor": epileptor_attributes_dict,
                                   "EpileptorDP": epileptorDP_attributes_dict,
                                   "EpileptorDPrealistic": epileptorDPrealistic_attributes_dict,
                                   "EpileptorDP2D": epileptorDP2D_attributes_dict}

# Attributes to be read or written for noise object and files:
simulation_settings_attributes_dict = {"Simulated length (ms)": "simulated_period",
                                       "Integrator type": "integrator_type",
                                       "Integration time step (ms)": "integration_step", "Time scaling": "scale_time",
                                       "Noise type": "noise_type", "Noise time scale": "noise_ntau",
                                       "Noise intensity": "noise_intensity", "Noise random seed": "noise_seed",
                                       "Monitor type": "monitor_type", "Monitor period (ms)": "monitor_sampling_period",
                                       "Monitor expressions": "monitor_expressions",
                                       "Variables names": "variables_names",
                                       "Initial conditions": "initial_conditions"}


# TODO: use new hypothesis
def read_simulation_settings(path=os.path.join(PATIENT_VIRTUAL_HEAD, "ep", "sim_ep.h5"), output="object",
                             hypothesis=None, logger=logger):
    """
    :param path: Path towards an hypothesis H5 file
    :return: hypothesis object
    """
    logger.info("Reading simulation settings from:\n" + path)
    h5_file = h5py.File(path, 'r', libver='latest')
    print_metadata(h5_file, logger)
    if output == "dict":  # or not (isinstance(hypothesis, Hypothesis)):
        model = dict()
        if hypothesis is not None:
            warning("hypothesis is not a Hypothesis object. Returning a dictionary for model.")
    else:
        if h5_file['/' + "model.name"][()] == "Epileptor":
            model = model_build_dict[h5_file['/' + "model.name"][()]](hypothesis)
        else:
            model = model_build_dict[h5_file['/' + "model.name"][()]](hypothesis,
                                                                      zmode=h5_file['/' + "model.zmode"][()])
    if h5_file['/' + "model.name"][()] != "Epileptor":
        overwrite_fields_dict = {"zmode": numpy.array(h5_file['/' + "model.zmode"][()])}
        if h5_file['/' + "model.name"][()] == "EpileptorDPrealistic":
            overwrite_fields_dict.update({"pmode": numpy.array(h5_file['/' + "model.pmode"][()])})
    read_object_from_h5_file(model, h5_file, epileptor_model_attributes_dict[h5_file['/' + "model.name"][()]],
                             add_overwrite_fields_dict=overwrite_fields_dict)
    if output == "dict":
        sim_settings = dict()
    else:
        sim_settings = SimulationSettings()
    # overwrite_fields_dict = {"monitor_expressions": h5_file['/' + "Monitor expressions"][()].tostring().split(","),
    #                          "variables_names": h5_file['/' + "Variables names"][()].tostring().split(",")}
    overwrite_fields_dict = {"monitor_expressions": h5_file['/' + "Monitor expressions"][()].tolist(),
                             "variables_names": h5_file['/' + "Variables names"][()].tolist()}
    read_object_from_h5_file(sim_settings, h5_file, simulation_settings_attributes_dict,
                             add_overwrite_fields_dict=overwrite_fields_dict)
    h5_file.close()
    return model, sim_settings


def write_ts(raw_data, sampling_period, folder=os.path.join(PATIENT_VIRTUAL_HEAD, "ep"), filename="ts_from_python.h5",
             logger=logger):
    path, overwrite = change_filename_or_overwrite(os.path.join(folder, filename))
    # if os.path.exists(path):
    #     print "TS file %s already exists. Use a different name!" % path
    #     return
    logger.info("Writing a TS at:\n" + path)
    if overwrite:
        try:
            os.remove(path)
        except:
            warning("\nFile to overwrite not found!")
    h5_file = h5py.File(path, 'a', libver='latest')
    write_metadata({KEY_TYPE: "TimeSeries"}, h5_file, KEY_DATE, KEY_VERSION)
    if isinstance(raw_data, dict):
        for data in raw_data:
            if len(raw_data[data].shape) == 2 and str(raw_data[data].dtype)[0] == "f":
                h5_file.create_dataset("/" + data, data=raw_data[data])
                write_metadata({KEY_MAX: raw_data[data].max(), KEY_MIN: raw_data[data].min(),
                                KEY_STEPS: raw_data[data].shape[0], KEY_CHANNELS: raw_data[data].shape[1],
                                KEY_SV: 1,
                                KEY_SAMPLING: sampling_period, KEY_START: 0.0
                                }, h5_file, KEY_DATE, KEY_VERSION, "/" + data)
            else:
                raise_value_error("Invalid TS data. 2D (time, nodes) numpy.ndarray of floats expected")
    elif isinstance(raw_data, numpy.ndarray):
        if len(raw_data.shape) != 2 and str(raw_data.dtype)[0] != "f":
            h5_file.create_dataset("/data", data=raw_data)
            write_metadata({KEY_MAX: raw_data.max(), KEY_MIN: raw_data.min(),
                            KEY_STEPS: raw_data.shape[0], KEY_CHANNELS: raw_data.shape[1], KEY_SV: 1,
                            KEY_SAMPLING: sampling_period, KEY_START: 0.0
                            }, h5_file, KEY_DATE, KEY_VERSION, "/data")
        else:
            raise_value_error("Invalid TS data. 2D (time, nodes) numpy.ndarray of floats expected")
    else:
        raise_value_error("Invalid TS data. Dictionary or 2D (time, nodes) numpy.ndarray of floats expected")
    h5_file.close()


def write_ts_epi(raw_data, sampling_period, lfp_data=None, folder=os.path.join(PATIENT_VIRTUAL_HEAD, "ep"),
                 filename="ts_from_python.h5", logger=logger):
    path, overwrite = change_filename_or_overwrite(folder, filename)
    # if os.path.exists(path):
    #     print "TS file %s already exists. Use a different name!" % path
    #     return
    if raw_data is None or len(raw_data.shape) != 3:
        raise_value_error("Invalid TS data 3D (time, regions, sv) expected", logger)
    logger.info("Writing a TS at:\n" + path)
    if type(lfp_data) == int:
        lfp_data = raw_data[:, :, lfp_data[1]]
        raw_data[:, :, lfp_data[1]] = []
    elif isinstance(lfp_data, list):
        lfp_data = raw_data[:, :, lfp_data[1]] - raw_data[:, :, lfp_data[0]]
    elif isinstance(lfp_data, numpy.ndarray):
        lfp_data = lfp_data.reshape((lfp_data.shape[0], lfp_data.shape[1], 1))
    else:
        raise_value_error("Invalid lfp_data 3D (time, regions, sv) expected", logger)
    if overwrite:
        try:
            os.remove(path)
        except:
            warning("\nFile to overwrite not found!")
    h5_file = h5py.File(path, 'a', libver='latest')
    h5_file.create_dataset("/data", data=raw_data)
    h5_file.create_dataset("/lfpdata", data=lfp_data)
    write_metadata({KEY_TYPE: "TimeSeries"}, h5_file, KEY_DATE, KEY_VERSION)
    write_metadata({KEY_MAX: raw_data.max(), KEY_MIN: raw_data.min(),
                    KEY_STEPS: raw_data.shape[0], KEY_CHANNELS: raw_data.shape[1], KEY_SV: raw_data.shape[2],
                    KEY_SAMPLING: sampling_period, KEY_START: 0.0
                    }, h5_file, KEY_DATE, KEY_VERSION, "/data")
    write_metadata({KEY_MAX: lfp_data.max(), KEY_MIN: lfp_data.min(),
                    KEY_STEPS: lfp_data.shape[0], KEY_CHANNELS: lfp_data.shape[1], KEY_SV: 1,
                    KEY_SAMPLING: sampling_period, KEY_START: 0.0
                    }, h5_file, KEY_DATE, KEY_VERSION, "/lfpdata")
    h5_file.close()


def write_ts_seeg_epi(seeg_data, sampling_period, folder=os.path.join(PATIENT_VIRTUAL_HEAD, "ep"),
                      filename="ts_from_python.h5", logger=logger):
    path = os.path.join(folder, filename)
    if not os.path.exists(path):
        raise_error("TS file %s does not exist. First define the raw data!" + path, logger)
        return
    sensors_name = "SeegSensors-" + str(seeg_data.shape[1])
    logger.info("Writing a TS at:\n" + path + ", " + sensors_name)
    try:
        h5_file = h5py.File(path, 'a', libver='latest')
        h5_file.create_dataset("/" + sensors_name, data=seeg_data)
        write_metadata({KEY_MAX: seeg_data.max(), KEY_MIN: seeg_data.min(),
                        KEY_STEPS: seeg_data.shape[0], KEY_CHANNELS: seeg_data.shape[1], KEY_SV: 1,
                        KEY_SAMPLING: sampling_period, KEY_START: 0.0
                        }, h5_file, KEY_DATE, KEY_VERSION, "/" + sensors_name)
        h5_file.close()
    except Exception, e:
        raise_error(e + "\nSeeg dataset already written as " + sensors_name, logger)
