import numpy as np
from scipy.signal import decimate

from tvb_epilepsy.base.computations.analyzers_utils import filter_data
from tvb_epilepsy.base.utils.plot_utils import plot_timeseries


def get_bipolar_channels(channels_inds, channel_lbls=[]):
    import re
    n_channels = len(channels_inds)
    if channel_lbls == []:
        channel_lbls = str(range(n_channels))
    bipolar_channel_lbls = []
    bipolar_ch_inds = []
    for iS in range(n_channels - 1):
        try:
            if (channel_lbls[iS][0] == channel_lbls[iS + 1][0]) and \
                    (int(re.findall(r'\d+', channel_lbls[iS])[0]) == int(re.findall(r'\d+', channel_lbls[iS + 1])[0]) - 1):
                bipolar_channel_lbls.append(channel_lbls[iS] + "-" + channel_lbls[iS + 1])
                bipolar_ch_inds.append(channels_inds[iS])
        except:
            print("WTF?")
    return bipolar_ch_inds, bipolar_channel_lbls


def decimate_signals(time, signals, decim_ratio):
    signals = decimate(signals, decim_ratio, axis=0, zero_phase=True)
    time = decimate(time, decim_ratio, zero_phase=True)
    dt = np.mean(time)
    observation_shape = signals.shape
    (n_times, n_signals) = observation_shape
    return signals, time, dt, n_times, n_signals, observation_shape


def cut_signals_tails(time, signals, cut_tails):
    signals = signals[cut_tails[0]:-cut_tails[-1]]
    time = time[cut_tails[0]:-cut_tails[-1]]
    observation_shape = signals.shape
    (n_times, n_signals) = observation_shape
    return signals, time, n_times, n_signals, observation_shape


def prepare_seeg_observable(seeg_path, on_off_set, channels, win_len=5.0, low_freq=10.0, high_freq=None, log_flag=True,
                            plot_flag=False):
    import re
    from pylab import detrend_linear
    from mne.io import read_raw_edf
    from tvb_epilepsy.base.utils.plot_utils import plot_raster, plot_spectral_analysis_raster
    raw_data = read_raw_edf(seeg_path, preload=True)
    rois = np.where([np.in1d(s.split("POL ")[-1], channels) for s in raw_data.ch_names])[0]
    raw_data.resample(128.0)
    fs = raw_data.info['sfreq']
    data, times = raw_data[:, :]
    data = data[rois].T
    if plot_flag:
        plot_spectral_analysis_raster(times, data, time_units="sec", freq=np.array(range(1,51,1)),
                                      title='Spectral Analysis', figure_name='Spectral Analysis', labels=channels,
                                      log_scale=True, save_flag=True)
    data_bipolar = []
    bipolar_channels = []
    data_filtered = []
    bipolar_ch_inds = []
    for iS in range(data.shape[1]-1):
        if (channels[iS][0] == channels[iS+1][0]) and \
                (int(re.findall(r'\d+', channels[iS])[0]) == int(re.findall(r'\d+', channels[iS+1])[0])-1):
            data_bipolar.append(data[:, iS] - data[:, iS+1])
            bipolar_channels.append(channels[iS] + "-" + channels[iS+1])
            data_filtered.append(filter_data(data_bipolar[-1], low_freq, 60.0, fs, order=3))
            bipolar_ch_inds.append(iS)
    data_bipolar = np.array(data_bipolar).T
    data_filtered = np.array(data_filtered).T
    # filter_data, times = raw_data.filter(low_freq, 100.0, picks=rois)[:, :]
    if plot_flag:
        plot_spectral_analysis_raster(times, data_bipolar, time_units="sec", freq=np.array(range(1, 51, 1)),
                                  title='Spectral Analysis',
                                  figure_name='Spectral Analysis Bipolar', labels=bipolar_channels,
                                  show_flag=True, save_flag=True, log_scale=True)
        plot_spectral_analysis_raster(times, data_filtered, time_units="sec", freq=np.array(range(1, 51, 1)),
                                  title='Spectral Analysis_bipolar', figure_name='Spectral Analysis Filtered', labels=bipolar_channels,
                                      log_scale=True, save_flag=True)
    del data
    t_onset = np.where(times > (on_off_set[0] - 2 * win_len))[0][0]
    t_offset = np.where(times > (on_off_set[1] + 2 * win_len))[0][0]
    times = times[t_onset:t_offset]
    data_filtered = data_filtered[t_onset:t_offset]
    observation = np.abs(data_filtered)
    del data_filtered
    if log_flag:
        observation = np.log(observation)
    for iS in range(observation.shape[1]):
        observation[:, iS] = detrend_linear(observation[:, iS])
    observation -= observation.min()
    for iS in range(observation.shape[1]):
        observation[:, iS] = np.convolve(observation[:, iS], np.ones((np.int(np.round(win_len * fs),))), mode='same')
    n_times = times.shape[0]
    dtimes = n_times - 4096
    t_onset = int(np.ceil(dtimes / 2.0))
    t_offset = n_times-int(np.floor(dtimes / 2.0))
    # t_onset = np.where(times > (on_off_set[0] - win_len))[0][0]
    # t_offset = np.where(times > (on_off_set[1] + win_len))[0][0]
    times = times[t_onset:t_offset]
    observation = observation[t_onset:t_offset]
    if plot_flag:
        plot_raster(times, {"observation": observation}, time_units="sec", special_idx=None, title='Time Series',
                    offset=1.0, figure_name='TimeSeries', labels=bipolar_channels, save_flag=True)
    # n_times = times.shape[0]
    # observation = resample_poly(observation, 2048, n_times)
    observation = decimate(observation, 2, axis=0, zero_phase=True)
    times = decimate(times, 2, zero_phase=True)
    observation -= observation.min()
    observation /= observation.max()
    if plot_flag:
        plot_timeseries(times, {"observation": observation}, time_units="sec", special_idx=None, title='Time Series',
                    figure_name='TimeSeriesDecimated', labels=bipolar_channels, show_flag=True, save_flag=True) #
    return observation, times, fs/2
