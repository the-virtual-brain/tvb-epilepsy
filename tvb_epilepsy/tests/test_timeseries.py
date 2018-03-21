from collections import OrderedDict

import numpy as np

from tvb_epilepsy.base.model.timeseries import Timeseries
from tvb_epilepsy.io.h5_reader import H5Reader
import h5py


ts_path = "/Users/dionperd/Dropbox/Work/VBtech/VEP/results/tests/res/LSA_e_x0_Hypothesis_ts.h5"
conn_path = "/Users/dionperd/Dropbox/Work/VBtech/VEP/results/CC/TVB3/Head/Connectivity.h5"

def read_ts(path):
    h5_file = h5py.File(path, 'r', libver='latest')

    data = h5_file['/data'][()]
    sampling_period = h5_file["/data"].attrs["Sampling_period"]
    nr_of_steps = int(h5_file["/data"].attrs["Number_of_steps"])
    start_time = float(h5_file["/data"].attrs["Start_time"])

    h5_file.close()

    return data, sampling_period, nr_of_steps, start_time


if __name__ == "__main__":

    data, sampling_period, nr_of_steps, start_time = read_ts(ts_path)
    conn = H5Reader().read_connectivity(conn_path)
    signal = Timeseries(np.moveaxis(data, 2, 0),
                        OrderedDict({"space": list(conn.region_labels), "variables": ["x1", "z", "x2"]}),
                        start_time, sampling_period)
    time = signal.time
    subspace = signal.get_subspace_by_labels(list(conn.region_labels)[:3])
    timewindow = signal.get_time_window_by_index(10, 100)
    timewindowUnits = signal.get_time_window(time[10], time[100])

    dummy = signal.x1[1:101, "ctx-lh-caudalanteriorcingulate":"ctx-lh-lingual"]

    signal2 = Timeseries(signal.data, signal.dimension_labels, start_time, sampling_period)

    signal2.x1[1:101, "ctx-lh-caudalanteriorcingulate":"ctx-lh-lingual"] = dummy

    print(np.all(signal.x1[1:101, "ctx-lh-caudalanteriorcingulate":"ctx-lh-lingual"] ==
                 signal2.x1[1:101, "ctx-lh-caudalanteriorcingulate":"ctx-lh-lingual"]))
