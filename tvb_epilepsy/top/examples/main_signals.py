from collections import OrderedDict
import numpy
from tvb_epilepsy.base.model.timeseries import Timeseries
from tvb_epilepsy.io.h5_reader import H5Reader
import h5py


def read_ts(path):
    h5_file = h5py.File(path, 'r', libver='latest')

    data = h5_file['/data'][()]
    total_time = int(h5_file["/data"].attrs["Sampling_period"])
    nr_of_steps = int(h5_file["/data"].attrs["Number_of_steps"])
    start_time = float(h5_file["/data"].attrs["Start_time"])

    h5_file.close()

    return data, total_time, nr_of_steps, start_time

def prepare_dtype_for_2D(labels_list):
    list_of_dtype_tuples = []
    for label in labels_list:
        list_of_dtype_tuples.append((label, "f4"))

    return list_of_dtype_tuples


def prepare_dtype_for_3D(labels_list, sub_dimension_dtype=None):
    list_of_dtype_tuples = []
    if sub_dimension_dtype is None:
        list_of_dtype_tuples = prepare_dtype_for_2D(labels_list)
    else:
        for label in labels_list:
            list_of_dtype_tuples.append((label, sub_dimension_dtype))

    return list_of_dtype_tuples


def prepare_data_for_structured_array(data):
    result = []
    if len(data.shape) == 2:
        for vector in data:
            result.append(tuple(vector))
    if len(data.shape) == 3:
        for vector in data:
            subresult = []
            for subvector in vector:
                subresult.append(tuple(subvector))
            result.append(tuple(subresult))

    return result


if __name__ == "__main__":
    data, total_time, nr_of_steps, start_time = read_ts("/WORK/Episense/trunk/demo-data/Head_TREC/epHH/ts.h5")
    conn = H5Reader().read_connectivity("/WORK/Episense/trunk/demo-data/Head_TREC/Connectivity.h5")
    signal = Timeseries(data, OrderedDict({"space": list(conn.region_labels), "state_variables": ["a", "b", "c"]}),
                        start_time, total_time / float(nr_of_steps))

    timeline = signal.get_time_line()

    sv = signal.get_state_variable("a")

    subspace = signal.get_subspace_by_labels(list(conn.region_labels)[:3])
    timewindow = signal.get_time_window(10, 100)
    timewindowUnits = signal.get_time_window_by_units(timeline[10], timeline[100])

    # Numpy Structured Array example:

    # sv = prepare_dtype_for_2D(["a", "b", "c"])
    # rgs = prepare_dtype_for_3D(conn.region_labels)
    # rgs_sv = prepare_dtype_for_3D(conn.region_labels, sv)
    #
    # data2D = prepare_data_for_structured_array(data[0])
    # data3D = prepare_data_for_structured_array(data)
    #
    # struct_data2D = numpy.array(data2D, dtype=sv)
    # struct_data3D = numpy.rec.array(data3D, dtype=rgs_sv)

    # print struct_data3D[0]['Unknown']['a']
