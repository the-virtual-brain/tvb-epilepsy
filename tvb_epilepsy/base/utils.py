"""
Various transformation/computation functions will be placed here.
"""
import os
import logging
from datetime import datetime
from time import sleep
import numpy
import h5py
import warnings
from itertools import product
from scipy.signal import butter, lfilter
from collections import OrderedDict
from tvb_epilepsy.base.constants import FOLDER_LOGS, WEIGHTS_NORM_PERCENT, INTERACTIVE_ELBOW_POINT
from matplotlib import use
use('Qt4Agg')
from matplotlib import pyplot

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


def list_of_strings_to_string(lstr, sep=","):
    str = lstr[0]
    for s in lstr[1:]:
        str += sep+s
    return str


def shape_to_size(shape):
    shape = numpy.array(shape)
    shape = shape[shape>0]
    return shape.prod()


def assert_arrays(params, shape=None, transpose=False):
    # type: (object, object) -> object

    if shape is None or \
        not(isinstance(shape, tuple)
            and len(shape) in range(3) and numpy.all([isinstance(s, (int, numpy.int)) for s in shape])):
        shape = None
        shapes = [] # list of all unique shapes
        n_shapes = []   # list of all unique shapes' frequencies
        size = 0    # initial shape

    else:
        size = shape_to_size(shape)

    for ip in range(len(params)):

        # Convert all accepted types to numpy arrays:

        if isinstance(params[ip], numpy.ndarray):
            pass

        elif isinstance(params[ip], (list, tuple)):
            # assuming a list or tuple of symbols...
            params[ip] = numpy.array(params[ip]).astype(type(params[ip][0]))

        elif isinstance(params[ip], (float, int, long, complex, numpy.number)):
            params[ip] = numpy.array(params[ip])

        else:
            try:
                import sympy
            except:
                raise ImportError("sympy import failed")

            if isinstance(params[ip], tuple(sympy.core.all_classes)):
                params[ip] = numpy.array(params[ip])

            else:
                raise ValueError("Input " + str(params[ip]) + " of type " + str(type(params[ip])) + " is not numeric, "
                                                                                  "of type numpy.ndarray, nor Symbol")

        if shape is None:

            # Only one size > 1 is acceptable

            if params[ip].size != size:

                if size > 1 and params[ip].size > 1:

                    raise ValueError("Inputs are of at least two distinct sizes > 1")

                elif params[ip].size > size:

                    size = params[ip].size


            # Construct a kind of histogram of all different shapes of the inputs:

            ind = numpy.array([(x == params[ip].shape) for x in shapes])

            if numpy.any(ind):
                ind = numpy.where(ind)[0]
                #TODO: handle this properly
                n_shapes[int(ind)] += 1
            else:
                shapes.append(params[ip].shape)
                n_shapes.append(1)
        else:

            if params[ip].size > size:

                raise ValueError("At least one input is of a greater size than the one given!")

    if shape is None:

        # Keep only shapes of the correct size
        ind = numpy.array([shape_to_size(s) == size for s in shapes])
        shapes = numpy.array(shapes)[ind]
        n_shapes = numpy.array(n_shapes)[ind]

        # Find the most frequent shape
        ind = numpy.argmax(n_shapes)
        shape = tuple(shapes[ind])

    if transpose and len(shape) > 1:

        if (transpose is "horizontal" or "row" and shape[0] > shape[1]) or \
           (transpose is "vertical" or "column" and shape[0] < shape[1]):
            shape = list(shape)
            temp = shape[1]
            shape[1] = shape[0]
            shape[0] = temp
            shape = tuple(shape)

    # Now reshape or tile when necessary
    for ip in range(len(params)):

        try:
            if params[ip].shape != shape:

                if params[ip].size in [0, 1]:
                    params[ip] = numpy.tile(params[ip], shape)
                else:
                    params[ip] = numpy.reshape(params[ip], shape)
        except:
            print "what the fuck??"

    if len(params) == 1:
        return params[0]
    else:
        return tuple(params)


def linear_scaling(x, x1, x2, y1, y2):
    scaling_factor = (y2 - y1) / (x2 - x1)
    return y1 + (x - x1) * scaling_factor


def weighted_vector_sum(weights, vectors, normalize=True):

    if isinstance(vectors, numpy.ndarray):
        vectors = list(vectors.T)

    if normalize:
        weights /= numpy.sum(weights)

    vector_sum = weights[0] * vectors[0]
    for iv in range(1, len(weights)):
        vector_sum += weights[iv] * vectors[iv]

    return numpy.array(vector_sum)

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

    if isinstance(obj, (numpy.ndarray)):
        return obj.tolist()

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


def curve_elbow_point(vals):

    vals = numpy.array(vals).flatten()

    if numpy.any(vals[0:-1] - vals[1:] < 0):
        warnings.warn("Sorting vals in descending order...")
        vals = numpy.sort(vals)
        vals = vals[::-1]

    cumsum_vals = numpy.cumsum(vals)

    grad = numpy.gradient(numpy.gradient(numpy.gradient(cumsum_vals)))

    elbow = numpy.argmax(grad)

    if INTERACTIVE_ELBOW_POINT:

        pyplot.ion()

        fig, ax = pyplot.subplots()

        xdata = range(len(vals))
        lines=[]
        lines.append(ax.plot(xdata, cumsum_vals, 'g*', picker=None, label="values' cumulative sum")[0])
        lines.append(ax.plot(xdata, vals, 'bo', picker=None, label="values in descending order")[0])

        lines.append(ax.plot(elbow, vals[elbow], "rd",
                             label="suggested elbow point (maximum of third central difference)")[0])

        lines.append(ax.plot(elbow, cumsum_vals[elbow], "rd")[0])

        pyplot.legend(handles=lines[:2])

        class MyClickableLines(object):

            def __init__(self, fig, ax, lines):
                self.x = None
                #self.y = None
                self.ax = ax
                title = "Mouse lef-click please to select the elbow point..." + \
                        "\n...or click ENTER to continue accepting our automatic choice in red..."
                self.ax.set_title(title)
                self.lines = lines
                self.fig = fig

            def event_loop(self):
                self.fig.canvas.mpl_connect('button_press_event', self.onclick)
                self.fig.canvas.mpl_connect('key_press_event', self.onkey)
                self.fig.canvas.draw_idle()
                self.fig.canvas.start_event_loop(timeout=-1)
                return

            def onkey(self, event):
                if event.key == "enter":
                    self.fig.canvas.stop_event_loop()
                return

            def onclick(self, event):
                if event.inaxes != self.lines[0].axes: return
                dist = numpy.sqrt((self.lines[0].get_xdata() - event.xdata) ** 2.0)  # + (self.lines[0].get_ydata() - event.ydata) ** 2.)
                self.x = numpy.argmin(dist)
                self.fig.canvas.stop_event_loop()
                return

        click_point = MyClickableLines(fig, ax, lines)
        click_point.event_loop()

        if click_point.x is not None:
            elbow = click_point.x
            print "manual selection: ", elbow
        else:
            print "automatic selection: ", elbow

        return elbow

    else:

        return elbow


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


def set_time_scales(fs=4096.0, dt=None, time_length=1000.0, scale_time=1.0, scale_fsavg=8.0, report_every_n_monitor_steps=10,):
    if dt is None:
        dt = 1000.0 / fs

    dt /= scale_time

    fsAVG = fs / scale_fsavg
    monitor_period = scale_fsavg * dt
    sim_length = time_length / scale_time
    time_length_avg = numpy.round(sim_length / monitor_period)
    n_report_blocks = max(report_every_n_monitor_steps * numpy.round(time_length_avg / 100), 1.0)

    return dt, fsAVG, sim_length, monitor_period, n_report_blocks


def ensure_unique_file(parent_folder, filename):
    final_path = os.path.join(parent_folder, filename)

    while os.path.isfile(final_path):
        user_input = raw_input("File %s already exists. " 
                               "Enter a different name or press 'Enter/Return' to overwrite: " % final_path)
        if user_input == "":
            final_path = os.path.join(parent_folder, filename)
            break
        else:
            final_path = os.path.join(parent_folder, user_input)

    return final_path


def print_metadata(h5_file):
    print "Metadata:"
    for key, val in h5_file["/"].attrs.iteritems():
        print "\t", key, val


def write_metadata(meta_dict, h5_file, key_date, key_version, path="/"):
    root = h5_file[path].attrs
    root[key_date] = str(datetime.now())
    root[key_version] = 2
    for key, val in meta_dict.iteritems():
        root[key] = val


# TODO: Will we need any of this?
def write_object_to_h5_file(object, h5_file, attributes_dict=None,  add_overwrite_fields_dict=None, keys=None):

    if isinstance(h5_file, basestring):
        print "Writing to: ", h5_file
        h5_file = h5py.File(h5_file, 'a', libver='latest')
        if isinstance(keys, dict):
            write_metadata(keys, h5_file, keys["date"], keys["version"], path="/")

    if isinstance(object, dict):
        get_field = lambda object, key: object[key]
        if not(isinstance(attributes_dict, dict)):
            attributes_dict = dict()
            for key in object.keys():
                attributes_dict.update({key: key})
    else:
        get_field = lambda object, attribute: getattr(object, attribute)
        if not(isinstance(attributes_dict, dict)):
            attributes_dict = dict()
            for key in object.__dict__.keys():
                attributes_dict.update({key: key})

    for attribute in attributes_dict:

        field = get_field(object, attributes_dict[attribute])

        try:

            print "Writing " + attributes_dict[attribute] + "..."

            if isinstance(field, basestring):

                print "String length: ", len(field)
                h5_file.create_dataset("/" + attribute, data=field)
                print "String written length: ", len(h5_file['/' + attribute][()])

            elif isinstance(field, numpy.ndarray):
                print "Numpy array shape:", field.shape
                #TODO: deal with arrays of more than 2 dimensions
                if len(field.shape) > 2:
                    field = field.squeeze()
                    if len(field.shape) > 2:
                        field = field.flatten()
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

        #print "dataset %s value %s" % (attribute, h5_file['/' + attribute][()])

    if isinstance(add_overwrite_fields_dict, dict):

        for attribute in add_overwrite_fields_dict:

            print "Adding or overwritting " + attribute + "... "

            field = add_overwrite_fields_dict[attribute][0]
            mode = add_overwrite_fields_dict[attribute][1]

            if isinstance(field, basestring):
                print "String length: ", len(field)
                if mode == "overwrite":
                    del h5_file["/" + attribute]
                h5_file.create_dataset("/" + attribute, data=field)
                print "String written length: ", len(h5_file['/' + attribute][()])

            elif isinstance(field, numpy.ndarray):
                print "Numpy array shape:", field.shape
                if mode == "overwrite":
                    del h5_file["/" + attribute]
                h5_file.create_dataset("/" + attribute, data=field)
                print "Numpy array written shape: ", h5_file['/' + attribute][()].shape

            else:
                #try to write a scalar value
                try:
                    print "Writing scalar value..."
                    if mode == "overwrite":
                        del h5_file["/" + attribute]
                    h5_file.create_dataset("/" + attribute, data=field)
                except:
                    raise ValueError("Failed to write "+ attribute + " as a scalar value!")

            #print "dataset %s value %s" % (attribute, h5_file['/' + attribute][()])

    if isinstance(h5_file, basestring):
        h5_file.close()


def read_object_from_h5_file(object, h5_file, attributes_dict=None, add_overwrite_fields_dict=None):

    if isinstance(h5_file, basestring):
        print "Reading from:", h5_file
        h5_file = h5py.File(h5_file, 'r', libver='latest')
        print_metadata(h5_file)

    if not(isinstance(attributes_dict, dict)):
        attributes_dict = dict()
        for key in h5_file.keys():
            attributes_dict.update({key: key})

    if isinstance(object, dict):
        set_field = lambda object, key, data: object.update({key: data})
        get_field = lambda object, key: object[key]
    else:
        set_field = lambda object, attribute, data: setattr(object, attribute, data)
        get_field = lambda object, attribute: getattr(object, attribute)

    for attribute in attributes_dict:

        print "Reading " + attributes_dict[attribute] + "... "

        try:
            set_field(object, attributes_dict[attribute], h5_file['/' + attribute][()])
        except:
            raise ValueError("Failed to read " + attribute + "!")

        #print "attribute %s value %s" % (attribute, get_field(object, attributes_dict[attribute]))

    if isinstance(h5_file, basestring):
        h5_file.close()

    if isinstance(add_overwrite_fields_dict, dict):

        for attribute in add_overwrite_fields_dict:

            print "Setting or overwritting " + attribute + "... "

            try:
                set_field(object, attribute, add_overwrite_fields_dict[attribute])
            except:
                raise ValueError("Failed to set " + attribute + "!")

            #print "attribute %s value %s" % (attribute, get_field(object, attribute))

    return object


def assert_equal_objects(object1, object2, attributes_dict=None):

    if isinstance(object1, dict):
        get_field1 = lambda object, key: object[key]
        if not(isinstance(attributes_dict, dict)):
            attributes_dict = dict()
            for key in object1.keys():
                attributes_dict.update({key: key})
    else:
        get_field1 = lambda object, attribute: getattr(object, attribute)
        if not (isinstance(attributes_dict, dict)):
            attributes_dict = dict()
            for key in object1.__dict__.keys():
                attributes_dict.update({key: key})

    if isinstance(object2, dict):
        get_field2 = lambda object, key: object[key]
    else:
        get_field2 = lambda object, attribute: getattr(object, attribute)

    for attribute in attributes_dict:
        #print attributes_dict[attribute]
        field1 = get_field1(object1, attributes_dict[attribute])
        field2 = get_field2(object2, attributes_dict[attribute])

        try:
            #TODO: a better hack for the stupid case of an ndarray of a string, such as model.zmode or pmode

            # For non numeric types
            if isinstance(field1, basestring) or isinstance(field1, list) or isinstance(field1, dict) \
                    or (isinstance(field1, numpy.ndarray) and field1.dtype.kind in 'OSU'):
                if numpy.any(field1 != field2):
                    raise ValueError("Original and read object field "
                                     + attributes_dict[attribute] + " not equal!")

            # For numeric types
            elif isinstance(field1, (int, float, long, complex, numpy.number, numpy.ndarray)) \
                and not (isinstance(field1, numpy.ndarray) and field1.dtype.kind in 'OSU'):
                # TODO: handle better accuracy differences and complex numbers...
                if numpy.any(numpy.float32(field1) - numpy.float32(field2) > 0):
                    raise ValueError("Original and read object field "
                                     + attributes_dict[attribute] + " not equal!")

            else:
                warnings.warn("No comparison made for field "
                                 + attributes_dict[attribute] + " because is of unknown type!")
        except:
            raise ValueError("Something went wrong when trying to compare "
                             + attributes_dict[attribute] + " !")
