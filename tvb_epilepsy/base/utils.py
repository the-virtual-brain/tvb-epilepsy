"""
@version $Id: utils.py 1588 2016-08-18 23:44:14Z denis $

Various transformation/computation functions will be placed here.
"""
import logging
import numpy as np
from itertools import product
from scipy.signal import butter, lfilter
from collections import OrderedDict
from tvb_epilepsy.base.constants import *

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


def obj_to_dict(obj):
    """
    :param obj: Python object to introspect
    :return: dictionary after recursively taking obj fields and their values
    """
    if obj is None:
        return obj

    if isinstance(obj, (str, int, float)):
        return obj
    if isinstance(obj, (np.float32,)):
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
    if not (isinstance(x, np.ndarray)):
        return x
    else:
        y=np.squeeze(x)
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

    if not (isinstance(x, (str, int, float, list, np.ndarray))):
        return x
    else:
        if not (isinstance(x, list)):
            x = np.squeeze(x)
        x_no = len(x)
        if not (isinstance(lbl, (list, np.ndarray))):
            lbl = np.repeat('', x_no)
        else:
            lbl = np.squeeze(lbl)
        labels_no = len(lbl)
        total_no = min(labels_no, x_no)
        if x_no <= labels_no:
            if sort=='ascend':
                ind = np.argsort(x).tolist()
            elif sort == 'descend':
                ind = np.argsort(x)
                ind = ind[::-1].tolist()
            else:
                ind = range(x_no)
        else: 
            ind = range(total_no)
        d = OrderedDict()
        for i in ind:
            d[str(i) + '.' + str(lbl[i])] = x[i]
        if labels_no > total_no:
            ind_lbl = np.delete(np.array(range(labels_no)),ind).tolist()
            for i in ind_lbl:
                d[str(i) + '.' + str(lbl[i])] = None
        if x_no > total_no:
            ind_x = np.delete(np.array(range(x_no)),ind).tolist()
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

    normalized_w = np.array(weights)

    # Remove diagonal elements
    n_regions = normalized_w.shape[0]
    normalized_w *= 1 - np.eye(n_regions)

    # Normalize with the 95th percentile
    # if np.max(normalized_w) - max_w > 1e-6:
    normalized_w = np.array(normalized_w / np.percentile(normalized_w, percentile))
    #    else:
    #        normalized_w = np.array(weights)

    # normalized_w[normalized_w > max_w] = max_w

    return normalized_w


def calculate_in_degree(weights):
    return np.expand_dims(np.sum(weights, axis=1), 1).T


def calculate_projection(sensors, connectivity):
    n_sensors = sensors.number_of_sensors
    n_regions = connectivity.number_of_regions
    projection = np.zeros((n_sensors, n_regions))
    dist = np.zeros((n_sensors, n_regions))

    for iS, iR in product(range(n_sensors), range(n_regions)):
        dist[iS, iR] = np.sqrt(np.sum((sensors.locations[iS, :] - connectivity.centers[iR, :]) ** 2))
        projection[iS, iR] = 1 / dist[iS, iR] ** 2

    projection /= np.percentile(projection, 95)
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
