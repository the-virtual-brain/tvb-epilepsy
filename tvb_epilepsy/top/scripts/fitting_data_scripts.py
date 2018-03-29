import numpy as np
from scipy.signal import decimate
from scipy.stats import zscore
from tvb_epilepsy.base.computations.analyzers_utils import filter_data
from tvb_epilepsy.base.utils.data_structures_utils import ensure_string
from tvb_epilepsy.base.constants.model_inversion_constants import WIN_LEN_RATIO, LOW_FREQ, HIGH_FREQ, BIPOLAR, LOG_FLAG


def prepare_signal_observable(data, times, dynamical_model, on_off_set, labels, rois=[], win_len_ratio=WIN_LEN_RATIO,
                              low_freq=LOW_FREQ, high_freq=HIGH_FREQ, log_flag=LOG_FLAG, plotter=False):
    win_len = int(np.round(data.shape[0] / win_len_ratio))
    n_rois = len(rois)
    plotlabels = labels
    if n_rois > 0:
        if len(labels) > n_rois:
            plotlabels = labels[rois]
        if data.shape[1] > n_rois:
            data = data[:, rois]
    if plotter:
        plotter.plot_spectral_analysis_raster(times, data, time_units="msec", freq=np.array(range(1, 51, 1)),
                                              title='Spectral Analysis', figure_name='Spectral Analysis',
                                              labels=plotlabels, log_scale=True)
    if dynamical_model.find("2D") > -1 or dynamical_model.find("2d"):
        print("Filtering signals...")
        fs = 1000.0 / np.mean(np.diff(times))
        high_freq = np.minimum(high_freq, 512.0)
        for ii in range(data.shape[1]):
            data[:, ii] = filter_data(data[:, ii], fs, low_freq, high_freq, "bandpass", order=3)
        if plotter:
            plotter.plot_raster({"Filtering": data[4*win_len:-4*win_len, :]}, times[4*win_len:-4*win_len],
                                time_units="msec", special_idx=[], title='Filtered Time Series', offset=1.0,
                                figure_name='FilteredTimeSeries', labels=plotlabels)
            plotter.plot_spectral_analysis_raster(times, data[win_len:-win_len, :], time_units="sec",
                                                  freq=np.array(range(1, 51, 1)),
                                                  title='Spectral Analysis_Filtered',
                                                  figure_name='Spectral Analysis Filtered', labels=plotlabels,
                                                  log_scale=True)

    def cut_fun(times, data, n_times, dtimes):
        t_onset = np.where(times > on_off_set[0])[0]
        if len(t_onset) > 0:
            t_onset = t_onset[0]
        else:
            t_onset = 0
        t_onset = np.maximum(int(np.minimum(t_onset, np.ceil(dtimes / 2.0))), 0)
        dtimes = dtimes - t_onset
        t_offset = np.where(times > on_off_set[1])[0]
        if len(t_offset) > 0:
            t_offset = t_offset[0]
        else:
            t_offset = n_times
        t_offset = np.minimum(int(np.maximum(t_offset, n_times - dtimes)), n_times)
        times = times[t_onset:t_offset]
        data = data[t_onset:t_offset]
        n_times = len(times)
        return times, data, n_times

    def xdecimate(times, data, decim_ratio, plotter):
        decim_ratio *= 2
        str_decim_ratio = str(decim_ratio)
        print("Decimating signals...")
        print("Decimation ratio: " + str_decim_ratio)
        data = decimate(data, 2, axis=0, zero_phase=True)
        times = decimate(times, 2, zero_phase=True)
        n_times = len(times)
        if plotter:
            plotter.plot_raster({str_decim_ratio+"xDecimationX": data}, times,
                                time_units="msec", special_idx=[],
                                title=str_decim_ratio + 'x Decimated Time Series', offset=0.1,
                                figure_name=str_decim_ratio + 'xDecimatedTimeSeries', labels=plotlabels)
        return times, data, decim_ratio, n_times

    def convolve(times, data, win_len, decim_ratio, plotter):
        print("Convolving signals'power with square window...")
        if decim_ratio > 1:
            str_decim_ratio = str(decim_ratio) + "x"
        else:
            str_decim_ratio = ""
        win_len = np.maximum(3, np.int(np.floor(win_len / decim_ratio)))
        for iS in range(data.shape[1]):
            data[:, iS] = np.convolve(data[:, iS], np.ones((np.int(np.round(win_len), ))), mode='same')
        if plotter:
            plotter.plot_raster({str_decim_ratio+"Convolution": data}, times,
                                time_units="msec", special_idx=[], title=str_decim_ratio+'Convolved Time Series',
                                offset=0.1, figure_name=str_decim_ratio+'ConvolvedTimeSeries', labels=plotlabels)
        return times, data, win_len

    decim_ratio = 1
    # n_times = times.shape[0]
    # data = resample_poly(data, 2048, n_times)
    data = data ** 2
    n_times = len(times)
    if n_times >= 1012:
        dtimes = np.mod(n_times, 1012)
        times, data, n_times = cut_fun(times, data, n_times, dtimes)

    times, data, win_len = convolve(times, data, win_len, decim_ratio, plotter)

    if n_times >= 1012:
        dtimes = int(np.round(np.mod(n_times, 1012)))
        times, data, n_times = cut_fun(times, data, n_times, dtimes)
        while n_times >= 3.0 * 1012 and n_times > 2 * win_len:
            times, data, decim_ratio, n_times = xdecimate(times, data, decim_ratio, plotter)
            times, data, win_len = convolve(times, data, win_len, decim_ratio, plotter)
            if n_times >= 1012:
                dtimes = np.mod(n_times, 1012)
                times, data, n_times = cut_fun(times, data, n_times, dtimes)
    if n_times >= 1012:
        dtimes = n_times - 1012
        times, data, n_times = cut_fun(times, data, n_times, dtimes)
    if log_flag:
        print("Log of signals...")
        data = np.log(data)
    print("Normalizing signals...")
    data = zscore(data, axis=None) / 3.0
    # observation -= observation.min()
    # observation /= observation.max()
    if plotter:
        plotter.plot_raster({"ObservationRaster": data}, times, time_units="msec", special_idx=[], offset=1.0,
                                title='Observation Raster Plot', figure_name='ObservationRasterPlot', labels=plotlabels)
        plotter.plot_timeseries({"Observation": data}, times, time_units="msec", special_idx=[],
                                title='Observation Time Series', figure_name='ObservationTimeSeries', labels=plotlabels)
    return data, times, np.array(rois), np.array(labels)


def prepare_seeg_observable(data, times, dynamical_model, on_off_set, sensors_lbls, sensors_inds=[],
                            win_len_ratio=WIN_LEN_RATIO, low_freq=LOW_FREQ, high_freq=HIGH_FREQ, bipolar=BIPOLAR,
                            log_flag=LOG_FLAG, plotter=False):
    import re
    if len(sensors_inds) == 0:
        sensors_inds = range(data.shape[1])
    if plotter:
        plotter.plot_spectral_analysis_raster(times, data, time_units="msec", freq=np.array(range(1, 51, 1)),
                                              title='Spectral Analysis', figure_name='Spectral Analysis',
                                              labels=sensors_lbls[sensors_inds], log_scale=True)
    if bipolar:
        print("Computing bipolar signals...")
        data_bipolar = []
        bipolar_ch_inds = []
        for iS, iCh in enumerate(sensors_inds[:-1]):
            if (sensors_inds[iS+1] == iCh+1) and (sensors_lbls[iCh][0] == sensors_lbls[iCh+1][0]) and \
                   (int(re.findall(r'\d+', sensors_lbls[iCh])[0]) == int(re.findall(r'\d+', sensors_lbls[iCh+1])[0])-1):
                data_bipolar.append(data[:, iS] - data[:, iS+1])
                sensors_lbls[iCh] = sensors_lbls[iCh] + "-" + sensors_lbls[iCh+1]
                bipolar_ch_inds.append(iCh)
        data = np.array(data_bipolar).T
        sensors_lbls = np.array(sensors_lbls)
        del data_bipolar
        if plotter:
            plotter.plot_raster({"Bipolar": data}, times, time_units="msec",
                                special_idx=[], title='Bipolar Time Series', offset=0.1,
                                figure_name='BipolarTimeSeries', labels=sensors_lbls[bipolar_ch_inds])
            plotter.plot_spectral_analysis_raster(times, data, time_units="msec", freq=np.array(range(1, 51, 1)),
                                                  title='Spectral Analysis', figure_name='Spectral Analysis Bipolar',
                                                  labels=sensors_lbls[sensors_inds], log_scale=True)
    else:
        bipolar_ch_inds = sensors_inds
    return prepare_signal_observable(data, times, dynamical_model, on_off_set, sensors_lbls, rois=bipolar_ch_inds,
                                     win_len_ratio=win_len_ratio, low_freq=low_freq, high_freq=high_freq,
                                     log_flag=log_flag, plotter=plotter)


def prepare_seeg_observable_from_mne_file(seeg_path, dynamical_model, on_off_set, sensors_lbls, initial_selection_inds=[],
                                          time_units="msec", win_len_ratio=WIN_LEN_RATIO, low_freq=LOW_FREQ,
                                          high_freq=HIGH_FREQ, bipolar=BIPOLAR, log_flag=LOG_FLAG, plotter=False):
    from pylab import detrend_linear
    from mne.io import read_raw_edf
    print("Reading empirical dataset from mne file...")
    raw_data = read_raw_edf(seeg_path, preload=True)
    rois = []
    if len(initial_selection_inds) == 0:
        initial_selection_inds = range(sensors_lbls)
    sensors_inds = []
    sensors_lbls = np.array(sensors_lbls)
    #included_channels = []
    print("Selecting target signals from dataset...")
    for iR, s in enumerate(raw_data.ch_names):
        this_label = s.split("POL ")[-1]
        this_index = np.where(np.array(this_label) == sensors_lbls)[0]
        if len(this_index) == 1 and this_index[0] in initial_selection_inds:
            # if this_index[0] == 86:
            #     print("WTF?!")
            # included_channels.append(s)
            rois.append(iR)
            sensors_inds.append(this_index[0])
    # raw_data.resample(512.0)
    data, times = raw_data[:, :]
    data = data[rois].T
    if ensure_string(time_units) == "sec":
        times = 1000 * times
    sort_inds = np.argsort(sensors_inds)
    sensors_inds = np.array(sensors_inds)[sort_inds]
    data = data[:, sort_inds]
    print("Linear detrending of signals...")
    for iS in range(data.shape[1]):
        data[:, iS] = detrend_linear(data[:, iS])
    if plotter:
        plotter.plot_raster({"Detrended": data}, times, time_units="msec",
                            special_idx=[], title='Detrended Time Series', offset=0.1,
                            figure_name='DetrendedTimeSeries', labels=sensors_lbls[sensors_inds])
        plotter.plot_spectral_analysis_raster(times, data, time_units="sec", freq=np.array(range(1, 51, 1)),
                                              title='Spectral Analysis', figure_name='Spectral Analysis',
                                              labels=sensors_lbls[sensors_inds], log_scale=True)
    return prepare_seeg_observable(data, times, dynamical_model, on_off_set, sensors_lbls, sensors_inds,
                                   win_len_ratio=win_len_ratio, low_freq=low_freq, high_freq=high_freq, bipolar=bipolar,
                                   log_flag=log_flag, plotter=plotter)
