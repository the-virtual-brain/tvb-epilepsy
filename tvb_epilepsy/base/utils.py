"""
Various transformation/computation functions will be placed here.
"""
import logging
import numpy
from itertools import product
from scipy.signal import butter, lfilter
from collections import OrderedDict
from tvb_epilepsy.base.constants import *
from tvb_epilepsy.tvb_api.epileptor_models import *


def initialize_logger(name, target_folder=FOLDER_LOGS):
    """
    create logger for a given module
    :param name: Logger Base Name
    :param target_folder: Folder where log files will be written
    """
    if not (os.path.isdir(target_folder)):
        os.mkdir(target_folder)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(target_folder, name + '.log'))
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)
    return logger


def linear_scaling(x,x1,x2,y1,y2):
        scaling_factor = (y2 - y1) / (x2 - x1)
        return y1 + (x - x1) * scaling_factor


def obj_to_dict(obj):
    """
    :param obj: Python object to introspect
    :return: dictionary after recursively taking obj fields and their values
    """
    if obj is None:
        return obj

    if isinstance(obj, (str, int, float)):
        return obj
    if isinstance(obj, (numpy.float32,)):
        return float(obj)

    if isinstance(obj, list):
        ret = []
        for val in obj:
            ret.append(obj_to_dict(val))
        return ret

    ret = {}
    for key in obj.__dict__:
        val = getattr(obj, key, None)
        ret[key] = obj_to_dict(val)
    return ret

def vector2scalar(x):
    if not (isinstance(x, numpy.ndarray)):
        return x
    else:
        y=numpy.squeeze(x)
    if all(y.squeeze()==y[0]):
        return y[0]
    else:
        return reg_dict(x)

def reg_dict(x, lbl=None, sort=None):
    """
    :x: a list or numpy vector 
    :lbl: a list or numpy vector of labels
    :return: dictionary 
    """

    if not (isinstance(x, (str, int, float, list, numpy.ndarray))):
        return x
    else:
        if not (isinstance(x, list)):
            x = numpy.squeeze(x)
        x_no = len(x)
        if not (isinstance(lbl, (list, numpy.ndarray))):
            lbl = numpy.repeat('', x_no)
        else:
            lbl = numpy.squeeze(lbl)
        labels_no = len(lbl)
        total_no = min(labels_no, x_no)
        if x_no <= labels_no:
            if sort=='ascend':
                ind = numpy.argsort(x).tolist()
            elif sort == 'descend':
                ind = numpy.argsort(x)
                ind = ind[::-1].tolist()
            else:
                ind = range(x_no)
        else:
            ind = range(total_no)
        d = OrderedDict()
        for i in ind:
            d[str(i) + '.' + str(lbl[i])] = x[i]
        if labels_no > total_no:
            ind_lbl = numpy.delete(numpy.array(range(labels_no)),ind).tolist()
            for i in ind_lbl:
                d[str(i) + '.' + str(lbl[i])] = None
        if x_no > total_no:
            ind_x = numpy.delete(numpy.array(range(x_no)),ind).tolist()
            for i in ind_x:
                d[str(i) + '.'] = x[i]
        return d


def formal_repr(instance, attr_dict):
    """ A formal string representation for an object.
    :param attr_dict: dictionary attribute_name: attribute_value
    :param instance:  Instance to read class name from it
    """
    class_name = instance.__class__.__name__
    formal = class_name + "{"
    for key, val in attr_dict.iteritems():
        if isinstance(val, dict):
            formal += "\n" + key + "=["
            for key2, val2 in val.iteritems():
                formal += "\n" + str(key2) + " = " + str(val2)
            formal += "]"
        else:
            formal += "\n" + str(key) + " = " + str(val)
    return formal + "}"


def normalize_weights(weights, percentile=WEIGHTS_NORM_PERCENT):  # , max_w=1.0
    # Create the normalized connectivity weights:

    normalized_w = numpy.array(weights)

    # Remove diagonal elements
    n_regions = normalized_w.shape[0]
    normalized_w *= 1 - numpy.eye(n_regions)

    # Normalize with the 95th percentile
    # if numpy.max(normalized_w) - max_w > 1e-6:
    normalized_w = numpy.array(normalized_w / numpy.percentile(normalized_w, percentile))
    #    else:
    #        normalized_w = numpy.array(weights)

    # normalized_w[normalized_w > max_w] = max_w

    return normalized_w


def calculate_in_degree(weights):
    return numpy.expand_dims(numpy.sum(weights, axis=1), 1).T


def calculate_projection(sensors, connectivity):
    n_sensors = sensors.number_of_sensors
    n_regions = connectivity.number_of_regions
    projection = numpy.zeros((n_sensors, n_regions))
    dist = numpy.zeros((n_sensors, n_regions))

    for iS, iR in product(range(n_sensors), range(n_regions)):
        dist[iS, iR] = numpy.sqrt(numpy.sum((sensors.locations[iS, :] - connectivity.centers[iR, :]) ** 2))
        projection[iS, iR] = 1 / dist[iS, iR] ** 2

    projection /= numpy.percentile(projection, 95)
    #projection[projection > 1.0] = 1.0
    return projection


def _butterworth_bandpass(lowcut, highcut, fs, order=3):
    """
    Build a diggital Butterworth filter
    """
    nyq = 0.5 * fs  # nyquist sampling rate
    low = lowcut / nyq  # normalize frequency
    high = highcut / nyq  # normalize frequency
    b, a = butter(order, [low, high], btype='band')
    return b, a


def filter_data(data, lowcut, highcut, fs, order=3):
    # get filter coefficients
    b, a = _butterworth_bandpass(lowcut, highcut, fs, order=order)
    # filter data
    y = lfilter(b, a, data)
    return y


def set_time_scales(fs=4096.0, dt=None, time_length=1000.0, scale_time=1.0, scale_fsavg=8.0,
                    hpf_low=None, hpf_high=None, report_every_n_monitor_steps=10,):
    if dt is None:
        dt = 1000.0 / fs / scale_time # msec
    else:
        dt /= scale_time

    fsAVG = fs / scale_fsavg
    monitor_period = scale_fsavg * dt
    sim_length = time_length / scale_time
    time_length_avg = numpy.round(sim_length / monitor_period)
    n_report_blocks = max(report_every_n_monitor_steps * numpy.round(time_length_avg / 100), 1.0)

    hpf_fs = fsAVG
    if hpf_low is None:
        hpf_low = max(16.0, 1000.0 / time_length)   # msec
    if hpf_high is None:
        hpf_high = min(250.0 , hpf_fs)

    return fs, dt, fsAVG, scale_time, sim_length, monitor_period, n_report_blocks, hpf_fs, hpf_low, hpf_high


def ensure_unique_file(parent_folder, filename):
    final_path = os.path.join(parent_folder, filename)

    while os.path.exists(final_path):
        filename = raw_input("File %s already exists. Enter a different name: " % parent_folder)
        final_path = os.path.join(parent_folder, filename)

    return final_path


def write_object_to_hd5_file(object, attributes_dict, h5_file):

    logger = get_logger()

    for attribute in attributes_dict:

        field = getattr(object, attributes_dict[attribute])

        try:

            if isinstance(attribute, basestring):
                print "String length: ", len(field)
                h5_file.create_dataset("/" + attribute, data=field)
                print "String written length: ", len(h5_file['/' + attribute][()])

            elif isinstance(attribute, numpy.ndarray):
                print "Numpy array shape:", field.shape
                h5_file.create_dataset("/" + attribute, data=field)
                print "Numpy array written shape: ", h5_file['/' + attribute][()].shape

            else:
                #try to write a scalar value
                try:
                    print "Writing scalar value..."
                    h5_file.create_dataset("/" + attribute, data=field)
                except:
                    raise ValueError("Failed to write "+ attribute + " as a scalar value!")

        except:
            raise ValueError(attribute + " not found in the object!")

        logger.debug("dataset %s value %s" % (attribute, h5_file['/' + attribute][()]))


def read_object_from_hd5_file(object, attributes_dict, h5_file, add_overwrite_fields_dict=None):

    logger = get_logger()

    if isinstance(object,dict):

        for attribute in attributes_dict:

            print "Reading " + attributes_dict[attribute] + "... "
            try:
                object[attributes_dict[attribute]] = h5_file['/' + attribute][()]
            except:
                raise ValueError("Failed to read " + attribute + "!")

            logger.debug("attribute %s value %s" % (attribute, object[attributes_dict[attribute]]))

        if isinstance(add_overwrite_fields_dict, dict):

            for attribute in add_overwrite_fields_dict:

                print "Setting or overwritting " + attributes_dict[attribute] + "... "
                try:
                    object[attribute] = add_overwrite_fields_dict[attribute]
                except:
                    raise ValueError("Failed to set " + attribute + "!")

                logger.debug("attribute %s value %s" % (attribute, object[attribute]))

    else:

        for attribute in attributes_dict:

            print "Reading " + attributes_dict[attribute] + "... "
            try:
                setattr(object, attributes_dict[attribute], h5_file['/' + attribute][()])
            except:
                raise ValueError("Failed to read " + attribute + "!")

            logger.debug("attribute %s value %s" % (attribute, getattr(object, attributes_dict[attribute])))

        if isinstance(add_overwrite_fields_dict, dict):

            for attribute in add_overwrite_fields_dict:

                print "Setting or overwritting " + attributes_dict[attribute] + "... "
                try:
                    setattr(object, attribute, add_overwrite_fields_dict[attribute])

                except:
                    raise ValueError("Failed to set " + attribute + "!")

                logger.debug("attribute %s value %s" % (attribute, getattr(object, attribute)))

    return object
