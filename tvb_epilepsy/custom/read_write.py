"""
    Python Demo for reading and writing
"""

import os
import h5py
import numpy
from datetime import datetime
from tvb_epilepsy.base.utils import get_logger, ensure_unique_file
from tvb_epilepsy.base.hypothesis import Hypothesis
from tvb_epilepsy.base.constants import hyp_attributes_dict

logger = get_logger()

PATIENT_VIRTUAL_HEAD = "/WORK/episense/episense-root/trunk/demo-data/Head_TREC"

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


def _print_metadata(h5_file):
    print "Metadata:"
    for key, val in h5_file["/"].attrs.iteritems():
        print "\t", key, val


def _write_metadata(meta_dict, h5_file, path="/"):
    root = h5_file[path].attrs
    root[KEY_DATE] = str(datetime.now())
    root[KEY_VERSION] = 2
    for key, val in meta_dict.iteritems():
        root[key] = val


def generate_connectivity_variant(uq_name, new_weights, new_tracts, description, new_w=None,
                                  path=os.path.join(PATIENT_VIRTUAL_HEAD, "Connectivity.h5")):
    """
    In existing Connectivity H5 define Weights and Tracts variants
    """
    print "Writing a Connectivity Variant in:", path
    h5_file = h5py.File(path, 'a', libver='latest')

    try:
        group = h5_file.create_group('/' + uq_name)
        # Array doesn't seem to work in Python 3
        # group.attrs["Operations"] = [description]
        group.attrs["Operations"] = description
        h5_file.create_dataset("/" + uq_name + "/weights", data=new_weights)
        if new_w is not None:
            h5_file.create_dataset("/" + uq_name + "/w", data=new_w)
        h5_file.create_dataset("/" + uq_name + "/tract_lengths", data=new_tracts)
        h5_file.close()
    except Exception, e:
        print e
        print "You should specify a unique group name %s" % uq_name


def read_epileptogenicity(path=os.path.join(PATIENT_VIRTUAL_HEAD, "ep", "ep.h5")):
    """
    :param path: Path towards an epileptogenicity H5 file
    :return: epileptogenicity in a numpy array
    """
    print "Reading Epileptogenicity from:", path
    h5_file = h5py.File(path, 'r', libver='latest')

    _print_metadata(h5_file)
    print "Structures:", h5_file["/"].keys()
    print "Values expected shape:", h5_file['/values'].shape

    values = h5_file['/values'][()]
    print "Actual values shape", values.shape

    h5_file.close()
    return values


def write_epileptogenicity_hypothesis(ep_vector, folder_name=None, file_name=None):
    """
    Store X0 values to be used when launching simulations
    """

    if file_name is None:
        file_name = "ep"

    if folder_name is None:
        folder_name = file_name

    path = os.path.join(PATIENT_VIRTUAL_HEAD, folder_name, file_name + ".h5")
    if os.path.exists(path):
        print "Ep file %s already exists. Use a different name!" % path
        return
    os.makedirs(os.path.dirname(path))

    print "Writing an Epileptogenicity at:", path
    h5_file = h5py.File(path, 'a', libver='latest')

    _write_metadata({KEY_TYPE: "ModelEpileptogenicity", KEY_NODES: ep_vector.shape[0]}, h5_file)
    h5_file.create_dataset("/values", data=ep_vector)
    h5_file.close()


def write_hypothesis(hypothesis, folder_name=None, file_name=None, hypo_name=None):
    """
    Store an hypothesis object to a hdf5 file
    """

    if not(isinstance(hypo_name, basestring)):
        hypo_name = hypothesis.name

    final_path = ensure_unique_file(folder_name, file_name)

    logger.info("Writing a hypothesis at: %s" % final_path)

    h5_file = h5py.File(final_path, 'a', libver='latest')
    h5_file.attrs.create("EPI_Type", "HypothesisModel")
    h5_file.attrs.create("Number_of_nodes", hypothesis.n_regions)

    for attribute in hyp_attributes_dict:
        print "Values shape:", numpy.array(getattr(hypothesis, hyp_attributes_dict[attribute])).shape
        try:
            if attribute == "EZ hypothesis":
                seizure_indices = numpy.zeros((hypothesis.n_regions,))
                seizure_indices[hypothesis.seizure_indices] = 1
                h5_file.create_dataset("/" + attribute, data=seizure_indices.astype("float32"))
            elif attribute == "Hypothesis name":
                h5_file.create_dataset("/" + attribute, data=hypo_name)
            else:
                h5_file.create_dataset("/" + attribute,
                                data=numpy.array(getattr(hypothesis, hyp_attributes_dict[attribute])).astype("float32"))
        except:
            raise ValueError(attribute + " not found in the hypothesis object!")
        print "Values written shape:", numpy.array(h5_file['/' + attribute][()]).shape
        logger.debug("dataset %s value %s" % (attribute, h5_file['/' + attribute][()]))

    h5_file.close()


def read_hypothesis(path=os.path.join(PATIENT_VIRTUAL_HEAD, "ep", "hypo_ep.h5"), output="object", hypo_name=None,
                    update_hypothesis=True):

    """
    :param path: Path towards an hypothesis H5 file
    :return: hypothesis object
    """

    print "Reading Hypothesis from:", path
    h5_file = h5py.File(path, 'r', libver='latest')

    _print_metadata(h5_file)
    attributes = h5_file["/"].keys()
    print "Attributes:", attributes

    if not(isinstance(hypo_name, basestring)):
        hypo_name = h5_file['/Hypothesis name'][()]

    if output == "object":

        hyp = Hypothesis(h5_file['/Connectivity'][()].shape[0], h5_file['/Connectivity'][()], hypo_name)

        for attribute in attributes:
            print "Values read shape:", numpy.array(h5_file['/' + attribute][()]).shape
            try:
                if attribute == "EZ hypothesis":
                    setattr(hyp, hyp_attributes_dict[attribute],
                            numpy.array(numpy.where(h5_file['/' + attribute][()] > 0)).astype("int32").squeeze())
                elif attribute == "Hypothesis name" or attribute == "Connectivity":
                    pass
                else:
                    setattr(hyp, hyp_attributes_dict[attribute], h5_file['/' + attribute][()])
            except:
                raise ValueError(attribute + " not found in the attributes dictionary!")
            print "Values set shape:", numpy.array(getattr(hyp, hyp_attributes_dict[attribute])).shape
            logger.debug("dataset %s value %s" % (attribute, getattr(hyp, hyp_attributes_dict[attribute])))

        if update_hypothesis:
            hyp._update_parameters(hyp.seizure_indices)

    else:

        hyp = dict()

        for attribute in attributes:
            print "Values read shape:", numpy.array(h5_file['/' + attribute][()]).shape
            try:
                if attribute == "EZ hypothesis":
                    seizure_indices = numpy.where(h5_file['/' + attribute][()] > 0)
                    hyp[hyp_attributes_dict[attribute]] = numpy.array(seizure_indices).astype('int32').squeeze()
                    hyp["n_"+hyp_attributes_dict[attribute]] = len(seizure_indices)
                elif attribute == "Hypothesis name":
                    hyp[hyp_attributes_dict[attribute]] = hypo_name
                else:
                    hyp[hyp_attributes_dict[attribute]] = h5_file['/'+attribute][()]
            except:
                raise ValueError(attribute + " not found in the attributes dictionary!")
            print "Values set shape:", numpy.array(hyp[hyp_attributes_dict[attribute]]).shape
            logger.debug("dataset %s value %s" % (attribute, hyp[hyp_attributes_dict[attribute]]))

        hyp["n_regions"] = hyp["x0"].size

    h5_file.close()

    return hyp


def import_sensors(src_txt_file):
    labels = numpy.loadtxt(src_txt_file, dtype=numpy.str, usecols=[0])
    locations = numpy.loadtxt(src_txt_file, dtype=numpy.float32, usecols=[1, 2, 3])
    write_sensors(labels, locations)


def write_sensors(labels, locations, file_name=None):
    """
    Store Sensors in a file to be shared by multiple patient virtualizations (heads)
    """
    if file_name is None:
        file_name = "SensorsSEEG_" + str(len(labels))
    path = os.path.join(os.path.dirname(PATIENT_VIRTUAL_HEAD), file_name + ".h5")

    if os.path.exists(path):
        print "Sensors file %s already exists. Use a different name!" % path
        return

    print "Writing Sensors at:", path
    h5_file = h5py.File(path, 'a', libver='latest')

    _write_metadata({KEY_TYPE: "SeegSensors", KEY_SENSORS: len(labels)}, h5_file)
    h5_file.create_dataset("/labels", data=labels)
    h5_file.create_dataset("/locations", data=locations)
    h5_file.close()


def read_ts(path=os.path.join(PATIENT_VIRTUAL_HEAD, "ep", "ts.h5")):
    """
    :param path: Path towards a valid TimeSeries H5 file
    :return: Timeseries in a numpy array
    """
    print "Reading TimeSeries from:", path
    h5_file = h5py.File(path, 'r', libver='latest')

    _print_metadata(h5_file)
    print "Structures:", h5_file["/"].keys()
    print "Data expected shape:", h5_file['/data'].shape

    data = h5_file['/data'][()]
    print "Actual Data shape", data.shape
    print "First Channel sv sum", numpy.sum(data[:, 0, :], axis=1)

    h5_file.close()
    return data


def write_ts(raw_data, sampling_period, path=os.path.join(PATIENT_VIRTUAL_HEAD, "ep", "ts_from_python.h5")):
    if os.path.exists(path):
        print "TS file %s already exists. Use a different name!" % path
        return
    if raw_data is None or len(raw_data.shape) != 3:
        print "Invalid TS data 3D (time, channels, sv) expected"
        return

    print "Writing a TS at:", path
    yc = raw_data[:, :, 0]
    y2 = raw_data[:, :, 2]
    lfp_data = y2 - yc
    lfp_data = lfp_data.reshape((lfp_data.shape[0], lfp_data.shape[1], 1))

    h5_file = h5py.File(path, 'a', libver='latest')
    h5_file.create_dataset("/data", data=raw_data)
    h5_file.create_dataset("/lfpdata", data=lfp_data)

    _write_metadata({KEY_TYPE: "TimeSeries"}, h5_file)
    _write_metadata({KEY_MAX: raw_data.max(), KEY_MIN: raw_data.min(),
                     KEY_STEPS: raw_data.shape[0], KEY_CHANNELS: raw_data.shape[1], KEY_SV: raw_data.shape[2],
                     KEY_SAMPLING: sampling_period, KEY_START: 0.0
                     }, h5_file, "/data")
    _write_metadata({KEY_MAX: lfp_data.max(), KEY_MIN: lfp_data.min(),
                     KEY_STEPS: lfp_data.shape[0], KEY_CHANNELS: lfp_data.shape[1], KEY_SV: 1,
                     KEY_SAMPLING: sampling_period, KEY_START: 0.0
                     }, h5_file, "/lfpdata")
    h5_file.close()


def write_ts_seeg(seeg_data, sampling_period, path=os.path.join(PATIENT_VIRTUAL_HEAD, "ep", "ts_from_python.h5")):
    if not os.path.exists(path):
        print "TS file %s does exists. First define the raw data!" % path
        return

    sensors_name = "SeegSensors-" + str(seeg_data.shape[1])
    print "Writing a TS at:", path, sensors_name

    try:
        h5_file = h5py.File(path, 'a', libver='latest')
        h5_file.create_dataset("/" + sensors_name, data=seeg_data)

        _write_metadata({KEY_MAX: seeg_data.max(), KEY_MIN: seeg_data.min(),
                         KEY_STEPS: seeg_data.shape[0], KEY_CHANNELS: seeg_data.shape[1], KEY_SV: 1,
                         KEY_SAMPLING: sampling_period, KEY_START: 0.0
                         }, h5_file, "/" + sensors_name)
        h5_file.close()
    except Exception, e:
        print e
        print "Seeg dataset already written %s" % sensors_name


if __name__ == "__main__":
    read_epileptogenicity()
    print "----------------"
    read_ts()
    print "----------------"

    # Simulating edit of a Connectivity.
    # It need to have the same number as the original connectivity, only weights and tracts changed.
    random_weights = numpy.random.random((88, 88))
    random_tracts = numpy.random.random((88, 88))
    generate_connectivity_variant('random3', random_weights, random_tracts, "Description of connectivity")
    print "----------------"

    # Define the X0 vector, that can be later used as input in a simulation from GUI
    random_x0 = numpy.random.random((88,))
    write_epileptogenicity_hypothesis("ep-random", random_x0)
    print "----------------"

    # Write TS
    random_ts = numpy.random.random((2000, 88, 3)).astype(numpy.float32)
    write_ts(random_ts, sampling_period=0.5)

    random_seeg = numpy.random.random((1000, 50)).astype(numpy.float32)
    write_ts_seeg(random_seeg, 2.0)
    print "-----------------"

    # Import Sensors from TXT file
    src_sensors_file = "/Users/lia.domide/Downloads/Denis/sEEG_position.txt"
    import_sensors(src_sensors_file)
