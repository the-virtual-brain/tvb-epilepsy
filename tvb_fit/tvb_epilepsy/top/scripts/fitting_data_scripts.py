import numpy as np

from tvb_fit.base.utils.log_error_utils import initialize_logger, warning
from tvb_fit.base.utils.data_structures_utils import isequal_string, ensure_list
from tvb_fit.service.timeseries_service import TimeseriesService, NORMALIZATION_METHODS
from tvb_fit.io.edf import read_edf_to_Timeseries

from tvb_fit.tvb_epilepsy.base.constants.model_inversion_constants import \
    SEIZURE_LENGTH, HIGH_HPF, LOW_HPF, LOW_LPF, HIGH_LPF, WIN_LEN, BIPOLAR, TARGET_DATA_PREPROCESSING


logger = initialize_logger(__name__)


def find_interval_of_interest(data, target_time_length, win_len=WIN_LEN, title_prefix="", plotter=None):

    def find_min_max_min_inds(data):
        left_min_ind = 0
        right_min_ind = len(data) - 1
        max_ind = data.argmax()
        if max_ind > left_min_ind:
            left_min_ind = data[:max_ind].argmin()
        if max_ind < right_min_ind:
            right_min_ind = max_ind + data[max_ind:].argmin()
        return left_min_ind, max_ind, right_min_ind

    power_sum_ts = data.squeezed.sum(axis=1)
    n_win_len = int(np.floor(win_len * data.sampling_frequency / 1000))
    n_data = data.time_length
    n_kernel_len = n_win_len # int(np.round(n_data / 10))
    # from scipy.signal import convolve
    # power_sum_ts = convolve(power_sum_ts, np.ones((n_kernel_len,)), mode='same')
    if plotter:
        plotter.plot_timeseries({"TotalPower": power_sum_ts}, data.time, time_units=data.time_unit, special_idx=[],
                                title='Total power time series', figure_name=title_prefix + '_TotalPowerTS')
    min_time = data.time_start + win_len
    max_time = data.time_end - win_len
    _, _, off_time_index = \
        tuple([ind+n_kernel_len for ind in find_min_max_min_inds(power_sum_ts[n_kernel_len:-n_kernel_len])])
    off_time = np.minimum(data.time[off_time_index] + win_len, max_time)
    on_time = np.maximum(min_time, off_time - target_time_length)
    off_time = on_time + target_time_length
    # on_time_margin = on_time - win_len
    # off_time_margin = np.maximum(max_time - off_time, 0.0) # margin to the right
    # margin_sums = on_time_margin + off_time_margin
    # time_length = off_time - on_time  # current time length
    # time_length_diff = target_time_length - time_length
    # on_time = on_time - np.floor(on_time_margin/margin_sums) * time_length_diff
    # off_time = on_time + time_length
    return [on_time, off_time]


def find_intervals_of_interest(data, time_lengths, win_len=WIN_LEN, title_prefix="", plotter=None):
    n_time_lengths = len(time_lengths)
    times_dict = dict(zip(time_lengths, n_time_lengths * []))
    for time_length in time_lengths:
        try:
            times_dict[time_length] = find_interval_of_interest(data, time_length, win_len, title_prefix, plotter)
        except:
            warning("Falied to select seizure interval for time length %s" % str(time_length))
            pass
    return times_dict


def prepare_signal_observable(data, seizure_length=SEIZURE_LENGTH, on_off_set=[], rois=[],
                              preprocessing=TARGET_DATA_PREPROCESSING, low_hpf=LOW_HPF, high_hpf=HIGH_HPF,
                              low_lpf=LOW_LPF, high_lpf=HIGH_LPF, win_len=WIN_LEN,
                              plotter=None, title_prefix=""):

    title_prefix = title_prefix + str(np.where(len(title_prefix) > 0, "_", "")) + "fit_data_preproc"

    ts_service = TimeseriesService()
    # Select rois if any:
    n_rois = len(rois)
    if n_rois > 0:
        if data.number_of_labels > n_rois:
            logger.info("Selecting signals...")
            if isinstance(rois[0], basestring):
                data = data.get_subspace_by_labels(rois)
            else:
                data = data.get_subspace_by_index(rois)

    win_len = np.minimum(win_len, data.duration/10)
    on_off_set = ensure_list(on_off_set)
    if len(on_off_set) == 2:
        # First cut data close to the desired interval
        duration = on_off_set[1] - on_off_set[0]
        temp_on_off = [np.maximum(data.time_start, on_off_set[0] - 2*win_len),
                       np.minimum(data.time_end, on_off_set[1] + 2*win_len)]
        data = data.get_time_window_by_units(temp_on_off[0], temp_on_off[1])
        if plotter:
            plotter.plot_raster({"PreSelectedTimeInterval": data.squeezed}, data.time, time_units=data.time_unit,
                                special_idx=[], title='PreSelected time interval time series', offset=0.1,
                                figure_name=title_prefix + '_PreSelectedRaster', labels=data.space_labels)
            plotter.plot_timeseries({"PreSelectedTimeInterval": data.squeezed}, data.time, time_units=data.time_unit,
                                    special_idx=[], title='PreSelected time interval time series',
                                    figure_name=title_prefix + '_PreSelectedTS', labels=data.space_labels)
    else:
        temp_on_off = [data.time_start, data.time_end]
        duration = temp_on_off[1] - temp_on_off[0]

    for i_preproc, preproc in enumerate(preprocessing):

        stri_preproc = str(i_preproc+1)

        # Now filter, if needed, before decimation introduces any artifacts
        if isequal_string(preproc, "hpf"):
            high_hpf = np.minimum(high_hpf, 256.0)
            logger.info("High-pass filtering signals...")
            data = ts_service.filter(data, low_hpf, high_hpf, "bandpass", order=3)
            if plotter:
                plotter.plot_raster({"High-pass filtering": data.squeezed}, data.time, time_units=data.time_unit,
                                    special_idx=[], title='High-pass filtered Time Series', offset=0.1,
                                    figure_name=title_prefix + '_%sHpfRaster' % stri_preproc,
                                    labels=data.space_labels)
                plotter.plot_timeseries({"High-pass filtering": data.squeezed}, data.time, time_units=data.time_unit,
                                        special_idx=[], title='High-pass filtered Time Series',
                                        figure_name=title_prefix + '_%sHpfTS' % stri_preproc,
                                        labels=data.space_labels)

        if isequal_string(preproc, "mean-center"):
            logger.info("Mean center data...")
            data = ts_service.normalize(data, "mean")
            if plotter:
                plotter.plot_raster({"Mean centered signals": data.squeezed}, data.time, time_units=data.time_unit,
                                    special_idx=[], title='Mean centered Time Series', offset=0.1,
                                    figure_name=title_prefix + '_%sMeanCenteredRaster' % stri_preproc,
                                    labels=data.space_labels)
                plotter.plot_timeseries({"Mean centered signals": data.squeezed}, data.time, time_units=data.time_unit,
                                        special_idx=[], title='Mean centered Time Series',
                                        figure_name=title_prefix + '_%sMeanCenteredTS' % stri_preproc,
                                        labels=data.space_labels)

        if isequal_string(preproc, "spectrogram"):
            # or
            # ...get the signals' envelope via Hilbert transform
            temp_duration = temp_on_off[1] - temp_on_off[0]
            decim_ratio = np.maximum(1, int(
                np.floor((1.0 * data.time_length / seizure_length) * (duration / temp_duration))))
            data.data = np.array(data.data).astype("float64")
            data = ts_service.spectrogram_envelope(data, high_hpf, low_hpf, decim_ratio)
            data.data /= data.data.std()
            data.data = np.array(data.data).astype("float32")
            temp_on_off = [data.time_start, data.time_end]
            if plotter:
                plotter.plot_raster({"Spectrogram signals": data.squeezed}, data.time, time_units=data.time_unit,
                                    special_idx=[], title='Spectrogram Time Series', offset=0.1,
                                    figure_name=title_prefix + '_%sSpectrogramRaster' % stri_preproc,
                                    labels=data.space_labels)
                plotter.plot_timeseries({"Spectrogram signals": data.squeezed}, data.time, time_units=data.time_unit,
                                        special_idx=[], title='Spectrogram Time Series',
                                        figure_name=title_prefix + '_%sSpectrogramTS' % stri_preproc,
                                        labels=data.space_labels)
        plot_envelope = ""
        if preproc.lower().find("envelope") >= 0: # isequal_string(preproc, "hilbert_envelope") or isequal_string(preproc, "abs_envelope"):
            plot_envelope = preproc
            if isequal_string(preproc, "hilbert_envelope"):
                # or
                # ...get the signals' envelope via Hilbert transform
                data = ts_service.hilbert_envelope(data)
                # or
            # elif isequal_string(preproc, "square"):
            #     # Square data to get positive "power like" timeseries (introducing though higher frequencies)
            #     data = ts_service.square(data)
            else: # isequal_string(preproc, "abs_envelope"): # "abs_envelope"
                data = ts_service.abs_envelope(data)
            if plotter:
                plot_envelop_ = plot_envelope.replace(" ", "_")
                plotter.plot_raster({plot_envelop_: data.squeezed}, data.time,
                                    time_units=data.time_unit, special_idx=[],
                                    title=plot_envelope, offset=0.1,
                                    figure_name=title_prefix + "_%s" % stri_preproc + plot_envelop_ + "Raster",
                                    labels=data.space_labels)
                plotter.plot_timeseries({plot_envelop_: data.squeezed}, data.time,
                                        time_units=data.time_unit, special_idx=[], title=plot_envelope,
                                        figure_name=title_prefix + "_%s" % stri_preproc + plot_envelop_ + "TS",
                                        labels=data.space_labels)

        if isequal_string(preproc, "log"):
            logger.info("Computing log of signals...")
            data = ts_service.log(data)
            if plotter:
                plotter.plot_raster({"LogTimeSeries": data.squeezed}, data.time, time_units=data.time_unit,
                                    special_idx=[], title='Log of Time Series', offset=0.1,
                                    figure_name=title_prefix + '_%sLogRaster' % stri_preproc,
                                    labels=data.space_labels)
                plotter.plot_timeseries({"LogTimeSeries": data.squeezed}, data.time, time_units=data.time_unit,
                                        special_idx=[], title='Log of Time Series',
                                        figure_name=title_prefix + '_%sLogTS' % stri_preproc,
                                        labels=data.space_labels)

        # Now convolve or low pass filter to smooth...
        if isequal_string(preproc, "convolve"):
            str_win_len = str(int(win_len))
            data = ts_service.convolve(data, win_len)
            logger.info("Convolving signals with a square window of " + str_win_len + " ms...")
            if plotter:
                plotter.plot_raster({"ConvolutionSmoothing": data.squeezed}, data.time,
                                    time_units=data.time_unit, special_idx=[], offset=0.1,
                                    title='Convolved Time Series with a window of ' + str_win_len + " ms",
                                    figure_name=
                                       title_prefix + '_%s_%smsWinConvolRaster' % (stri_preproc, str_win_len),
                                    labels=data.space_labels)
                plotter.plot_timeseries({"ConvolutionSmoothing": data.squeezed}, data.time,
                                        time_units=data.time_unit, special_idx=[],
                                        title='Convolved Time Series with a window of ' + str_win_len + " ms",
                                        figure_name=
                                            title_prefix + '_%s_%smsWinConvolTS' % (stri_preproc, str_win_len),
                                        labels=data.space_labels)

        elif isequal_string(preproc, "lpf"):
            high_lpf = np.minimum(high_lpf, 512.0)
            logger.info("Low-pass filtering signals...")
            data = ts_service.filter(data, low_lpf, high_lpf, "bandpass", order=3)
            if plotter:
                plotter.plot_raster({"Low-pass filtering": data.squeezed}, data.time, time_units=data.time_unit,
                                    special_idx=[], title='Low-pass filtered Time Series', offset=0.1,
                                    figure_name=title_prefix + '_%sLpfRaster' % stri_preproc,
                                    labels=data.space_labels)
                if plotter:
                    plotter.plot_timeseries({"Low-pass filtering": data.squeezed}, data.time,
                                            time_units=data.time_unit,
                                            special_idx=[], title='Low-pass filtered Time Series',
                                            figure_name=title_prefix + '_%sLpfTS' % stri_preproc,
                                            labels=data.space_labels)

   # Find the desired time interval
    if len(on_off_set) != 2:
        if len(on_off_set) == 1:
            on_off_set = find_interval_of_interest(data, on_off_set[0], win_len, title_prefix, plotter)
        else:
            on_off_set = temp_on_off
    # Cut to the desired interval now:
    data = data.get_time_window_by_units(on_off_set[0], on_off_set[1])
    temp_on_off = on_off_set
    duration = on_off_set[1] - on_off_set[0]

    if "decimate" in preprocessing:
        # Now decimate to get close to seizure_length points
        temp_duration = temp_on_off[1] - temp_on_off[0]
        decim_ratio = np.maximum(1, int(np.round((1.0*data.time_length/seizure_length) * (duration/temp_duration))))
        if decim_ratio > 1:
            str_decim_ratio = str(decim_ratio)
            logger.info("Decimating signals " + str_decim_ratio + " times...")
            data = ts_service.decimate(data, decim_ratio)
            if plotter:
                plotter.plot_raster({str_decim_ratio + " wise Decimation": data.squeezed}, data.time,
                                    time_units=data.time_unit, special_idx=[],
                                    title=str_decim_ratio + " wise Decimation", offset=0.1,
                                    figure_name=
                                          title_prefix + "_%s_%sxDecimRaster" % (stri_preproc, str_decim_ratio),
                                    labels=data.space_labels)
                plotter.plot_timeseries({str_decim_ratio + " wise Decimation": data.squeezed}, data.time,
                                        time_units=data.time_unit, special_idx=[],
                                        title=str_decim_ratio + " wise Decimation",
                                        figure_name=
                                            title_prefix + "_%s_%sxDecimTS" % (stri_preproc, str_decim_ratio),
                                        labels=data.space_labels)

    for preproc in preprocessing:
        if preproc in NORMALIZATION_METHODS:
            # Finally, normalize signals
            logger.info("Normalizing signals...")
            data = ts_service.normalize(data, preproc)  # "baseline", "baseline-std", "baseline-amplitude" or "zscore
    if plotter:
        plotter.plot_raster({"ObservationRaster": data.squeezed}, data.time, time_units=data.time_unit,
                            special_idx=[], offset=0.1, title='Observation Raster Plot',
                            figure_name=title_prefix + 'ObservationRaster', labels=data.space_labels)
        plotter.plot_timeseries({"Observation": data.squeezed}, data.time, time_units=data.time_unit,
                                special_idx=[], title='Observation Time Series',
                                figure_name=title_prefix + 'ObservationTS', labels=data.space_labels)
    return data


def prepare_simulated_seeg_observable(data, sensors, seizure_length=SEIZURE_LENGTH, log_flag=True, on_off_set=[],
                                      rois=[], preprocessing=TARGET_DATA_PREPROCESSING,
                                      low_hpf=LOW_HPF, high_hpf=HIGH_HPF, low_lpf=LOW_LPF, high_lpf=HIGH_LPF,
                                      bipolar=BIPOLAR, win_len_ratio=WIN_LEN, plotter=None, title_prefix=""):

    logger.info("Computing SEEG signals...")
    data = TimeseriesService().compute_seeg(data, sensors, sum_mode=np.where(log_flag, "exp", "lin"))
    if plotter:
        plotter.plot_raster({"SEEGData": data.squeezed}, data.time, time_units=data.time_unit,
                            special_idx=[], title='SEEG Time Series', offset=0.1,
                            figure_name=title_prefix + 'SEEGRaster', labels=data.space_labels)
    if bipolar:
        logger.info("Computing bipolar signals...")
        data = data.get_bipolar()
        if plotter:
            plotter.plot_raster({"BipolarData": data.squeezed}, data.time, time_units=data.time_unit,
                                special_idx=[], title='Bipolar Time Series', offset=0.1,
                                figure_name=title_prefix + 'BipolarRaster', labels=data.space_labels)
    return prepare_signal_observable(data, seizure_length, on_off_set, rois, preprocessing, low_hpf, high_hpf,
                                     low_lpf, high_lpf, win_len_ratio, plotter, title_prefix)


def prepare_seeg_observable_from_mne_file(seeg_path, sensors, rois_selection, seizure_length=SEIZURE_LENGTH,
                                          on_off_set=[], time_units="ms", label_strip_fun=None,
                                          preprocessing=TARGET_DATA_PREPROCESSING,
                                          low_hpf=LOW_HPF, high_hpf=HIGH_HPF, low_lpf=LOW_LPF, high_lpf=HIGH_LPF,
                                          bipolar=BIPOLAR, win_len_ratio=WIN_LEN, plotter=None, title_prefix=""):
    logger.info("Reading empirical dataset from edf file...")
    data = read_edf_to_Timeseries(seeg_path, sensors, rois_selection,
                                  label_strip_fun=label_strip_fun, time_units=time_units)
    data.data = np.array(data.data).astype("float32")
    if plotter:
        plotter.plot_raster({"OriginalData": data.squeezed}, data.time, time_units=data.time_unit,
                            special_idx=[], title='Original Empirical Time Series', offset=0.1,
                            figure_name=title_prefix + '_empirical_OriginalRaster', labels=data.space_labels)
    data = TimeseriesService().detrend(data)
    if plotter:
        plotter.plot_raster({"Detrended": data.squeezed}, data.time, time_units=data.time_unit,
                            special_idx=[], title='Detrended Time Series', offset=0.1,
                            figure_name=title_prefix + '_empirical_DetrendRaster', labels=data.space_labels)
    if bipolar:
        logger.info("Computing bipolar signals...")
        data = data.get_bipolar()
        if plotter:
            plotter.plot_raster({"BipolarData": data.squeezed}, data.time, time_units=data.time_unit,
                                special_idx=[], title='Bipolar Time Series', offset=0.1,
                                figure_name=title_prefix + 'BipolarRaster', labels=data.space_labels)

    return prepare_signal_observable(data, seizure_length, on_off_set, range(data.number_of_labels),
                                     preprocessing, low_hpf, high_hpf, low_lpf, high_lpf,
                                     win_len_ratio, plotter, title_prefix)


