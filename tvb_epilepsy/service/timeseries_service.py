
import numpy as np
from scipy.signal import decimate, convolve, detrend, hilbert
from scipy.stats import zscore

from tvb_epilepsy.base.utils.log_error_utils import raise_value_error, initialize_logger
from tvb_epilepsy.base.utils.data_structures_utils import isequal_string, ensure_list
from tvb_epilepsy.base.computations.math_utils import select_greater_values_array_inds, \
                                                      select_by_hierarchical_group_metric_clustering
from tvb_epilepsy.base.computations.analyzers_utils import filter_data
from tvb_epilepsy.base.model.timeseries import Timeseries, TimeseriesDimensions, PossibleVariables


def decimate_signals(signals, time, decim_ratio):
    if decim_ratio > 1:
        signals = decimate(signals, decim_ratio, axis=0, zero_phase=True, ftype="fir")
        time = decimate(time, decim_ratio, zero_phase=True, ftype="fir")
        dt = np.mean(np.diff(time))
        (n_times, n_signals) = signals.shape
        return signals, time, dt, n_times


def cut_signals_tails(signals, time, cut_tails):
    signals = signals[cut_tails[0]:-cut_tails[-1]]
    time = time[cut_tails[0]:-cut_tails[-1]]
    (n_times, n_signals) = signals.shape
    return signals, time, n_times


def normalize_signals(signals, normalization=None):
    if isinstance(normalization, basestring):
        if isequal_string(normalization, "zscore"):
            signals = zscore(signals, axis=None) / 3.0
        elif isequal_string(normalization, "minmax"):
            signals -= signals.min()
            signals /= signals.max()
        elif isequal_string(normalization, "baseline-amplitude"):
            signals -= np.percentile(np.percentile(signals, 1, axis=0), 1)
            signals /= np.percentile(np.percentile(signals, 99, axis=0), 99)
        else:
            raise_value_error("Ignoring signals' normalization " + normalization +
                             ",\nwhich is not one of the currently available " +
                             "'zscore', 'minmax' and  'baseline-amplitude'!")

    return signals

#TODO: Decide upon this commented method
# def compute_envelope(data, time, samp_rate, hp_freq=5.0, lp_freq=0.1, benv_cut=100, cut_tails=None, order=3):
#     if cut_tails is not None:
#         start = cut_tails[0]
#         stop = cut_tails[1]
#     else:
#         start = int(samp_rate / lp_freq)
#         skip = int(samp_rate / (lp_freq * 3))
#     data = filter_data(data, samp_rate, 'highpass', lowcut=lp_freq, order=order)
#     data = filter_data(np.abs(data), samp_rate, 'lowpass', highcut=hp_freq, order=order)
#     data = data[:, start::skip]
#     fm = benv > 100  # bipolar 100, otherwise 300 (roughly)
#     incl_names = "HH1-2 HH2-3".split()
#     incl_idx = np.array([i for i, (name, *_) in enumerate(contacts_bip) if name in incl_names])
#     incl = np.setxor1d(
#         np.unique(np.r_[
#                       incl_idx,
#                       np.r_[:len(fm)][fm.any(axis=1)]
#                   ])
#         , afc_idx)
#     isort = incl[np.argsort([te[fm[i]].mean() for i in incl])]
#     iother = np.setxor1d(np.r_[:len(benv)], isort)
#     lbenv = np.log(np.clip(benv[isort], benv[benv > 0].min(), None))
#     lbenv_all = np.log(np.clip(benv, benv[benv > 0].min(), None))
#     return te, isort, iother, lbenv, lbenv_all


class TimeseriesService(object):

    logger = initialize_logger(__name__)

    def __init__(self, logger=initialize_logger(__name__)):

        self.logger = logger

    def decimate(self, timeseries, decim_ratio):
        if decim_ratio > 1:
            return Timeseries(timeseries.data[0:timeseries.time_length:decim_ratio], timeseries.dimension_labels,
                              timeseries.time_start, decim_ratio*timeseries.time_step, timeseries.time_unit)
        else:
            return timeseries

    def decimate_by_filtering(self, timeseries, decim_ratio):
        if decim_ratio > 1:
            decim_data, decim_time, decim_dt, decim_n_times = decimate_signals(timeseries.squeezed,
                                                                               timeseries.time_line, decim_ratio)
            return Timeseries(decim_data, timeseries.dimension_labels,
                              decim_time[0], decim_dt, timeseries.time_unit)
        else:
            return timeseries

    def convolve(self, timeseries, win_len=None, kernel=None):
        if kernel is None:
            kernel = np.ones((np.int(np.round(win_len)), 1, 1, 1))
        else:
            kernel = kernel * np.ones((np.int(np.round(win_len)), 1, 1, 1))
        return Timeseries(convolve(timeseries.data, kernel, mode='same'), timeseries.dimension_labels,
                          timeseries.time_start, timeseries.time_step, timeseries.time_unit)

    def hilbert_envelope(self, timeseries):
        return Timeseries(np.abs(hilbert(timeseries.data, axis=0)), timeseries.dimension_labels,
                          timeseries.time_start, timeseries.time_step, timeseries.time_unit)

    def detrend(self, timeseries, type='linear'):
        return Timeseries(detrend(timeseries.data, axis=0, type=type), timeseries.dimension_labels,
                          timeseries.time_start, timeseries.time_step, timeseries.time_unit)

    def normalize(self, timeseries, normalization=None):
        return Timeseries(normalize_signals(timeseries.data, normalization), timeseries.dimension_labels,
                          timeseries.time_start, timeseries.time_step, timeseries.time_unit)

    def filter(self, timeseries, lowcut=None, highcut=None, mode='bandpass', order=3):
        return Timeseries(filter_data(timeseries.data, timeseries.sampling_frequency, lowcut, highcut, mode, order),
                         timeseries.dimension_labels, timeseries.time_start, timeseries.time_step, timeseries.time_unit)

    def log(self, timeseries):
        return Timeseries(np.log(timeseries.data), timeseries.dimension_labels,
                          timeseries.time_start, timeseries.time_step, timeseries.time_unit)

    def exp(self, timeseries):
        return Timeseries(np.exp(timeseries.data), timeseries.dimension_labels,
                          timeseries.time_start, timeseries.time_step, timeseries.time_unit)

    def abs(self, timeseries):
        return Timeseries(np.abs(timeseries.data), timeseries.dimension_labels,
                          timeseries.time_start, timeseries.time_step, timeseries.time_unit)

    def power(self, timeseries):
        return np.sum(self.square(timeseries).squeezed, axis=0)

    def square(self, timeseries):
        return Timeseries(timeseries.data ** 2, timeseries.dimension_labels,
                          timeseries.time_start, timeseries.time_step, timeseries.time_unit)

    def correlation(self, timeseries):
        return np.corrcoef(timeseries.squeezed.T)

    def select_by_metric(self, timeseries, metric, metric_th=None):
        return timeseries.get_subspace_by_index(select_greater_values_array_inds(metric, metric_th))

    def select_by_power(self, timeseries, power=np.array([]), power_th=None):
        if len(power) != timeseries.number_of_labels:
            power = self.power(timeseries)
        return self.select_by_metric(timeseries, power, power_th)

    def select_by_hierarchical_group_metric_clustering(self, timeseries, distance, disconnectivity=np.array([]),
                                                       metric=None, n_groups=10, members_per_group=1):
        selection = select_by_hierarchical_group_metric_clustering(distance, disconnectivity, metric,
                                                                   n_groups, members_per_group)
        return timeseries.get_subspace_by_index(selection)

    def select_by_correlation_power(self, timeseries, correlation=np.array([]), disconnectivity=np.array([]),
                                    power=np.array([]), n_groups=10, members_per_group=1):
        if correlation.shape[0] != timeseries.number_of_labels:
            correlation = self.correlation(timeseries)
        if len(power) != timeseries.number_of_labels:
            power = self.power(timeseries)
        return self.select_by_hierarchical_group_metric_clustering(timeseries, 1-correlation,
                                                                   disconnectivity, power, n_groups, members_per_group)

    def select_by_rois_proximity(self, timeseries, proximity, proximity_th=None):
        initial_selection = range(timeseries.number_of_labels)
        selection = []
        for prox in proximity:
                selection += (
                    np.array(initial_selection)[select_greater_values_array_inds(prox, proximity_th)]).tolist()
        return timeseries.get_subspace_by_index(np.unique(selection).tolist())

    def select_by_rois(self, timeseries, rois, all_labels):
        for ir, roi in rois:
            if not(isinstance(roi, basestring)):
                rois[ir] = all_labels[roi]
        return timeseries.get_subspace_by_labels(rois)

    def compute_seeg(self, source_timeseries, sensors, sum_mode="lin"):
        if sum_mode == "exp":
            seeg_fun = lambda source, gain_matrix: compute_seeg_exp(source.squeezed, gain_matrix)
        else:
            seeg_fun = lambda source, gain_matrix: compute_seeg_lin(source.squeezed, gain_matrix)
        seeg = []
        for id, sensor in enumerate(ensure_list(sensors)):
            seeg.append(Timeseries(seeg_fun(source_timeseries, sensor.gain_matrix),
                                   {TimeseriesDimensions.SPACE.value: sensor.labels,
                                    TimeseriesDimensions.VARIABLES.value:
                                        PossibleVariables.SEEG.value + str(id)},
                                   source_timeseries.time_start, source_timeseries.time_step,
                                   source_timeseries.time_unit))
        return seeg


def compute_seeg_lin(source_timeseries, gain_matrix):
    return source_timeseries.dot(gain_matrix.T)


def compute_seeg_exp(source_timeseries, gain_matrix):
    return np.log(np.exp(source_timeseries).dot(gain_matrix.T))









