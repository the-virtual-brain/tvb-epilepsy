from collections import OrderedDict
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
