from mne.io import read_raw_edf

import numpy as np

from tvb_epilepsy.base.utils.log_error_utils import initialize_logger
from tvb_epilepsy.base.utils.data_structures_utils import ensure_string
from tvb_epilepsy.base.model.timeseries import Timeseries, TimeseriesDimensions


def read_edf(path, sensors, rois_selection=None, label_strip_fun=None, time_units="ms",
                           logger=initialize_logger(__name__)):

    logger.info("Reading empirical dataset from mne file...")
    raw_data = read_raw_edf(path, preload=True)

    if not callable(label_strip_fun):
        label_strip_fun = lambda label: label

    rois = []
    rois_inds = []
    rois_lbls = []
    if len(rois_selection) == 0:
        rois_selection = sensors.labels

    logger.info("Selecting target signals from dataset...")
    for iR, s in enumerate(raw_data.ch_names):
        this_label = label_strip_fun(s)
        this_index = sensors.get_sensors_inds_by_sensors_labels(this_label)
        if this_label in rois_selection or (len(this_index) == 1 and this_index[0] in rois_selection):
            rois.append(iR)
            rois_inds.append(this_index[0])
            rois_lbls.append(this_label)

    data, times = raw_data[:, :]
    data = data[rois].T
    # Assuming that edf file time units is "sec"
    if ensure_string(time_units).find("ms") == 0:
        times = 1000 * times
    sort_inds = np.argsort(rois_inds)
    rois = np.array(rois)[sort_inds]
    rois_inds = np.array(rois_inds)[sort_inds]
    rois_lbls = np.array(rois_lbls)[sort_inds]
    data = data[:, sort_inds]

    return data, times, rois, rois_inds, rois_lbls


def read_edf_to_TimeSeries(path, sensors, rois_selection=None, label_strip_fun=None, time_units="ms",
                           logger=initialize_logger(__name__)):

    data, times, rois, rois_inds, rois_lbls = \
        read_edf(path, sensors, rois_selection, label_strip_fun, time_units, logger)

    return Timeseries(data, {TimeseriesDimensions.SPACE.value: rois_lbls},
                      times[0], np.mean(np.diff(times)), time_units)
