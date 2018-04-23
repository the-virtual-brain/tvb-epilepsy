import numpy as np
from tvb_epilepsy.base.constants.model_inversion_constants import WIN_LEN_RATIO, LOW_FREQ, HIGH_FREQ, BIPOLAR, LOG_FLAG
from tvb_epilepsy.base.utils.log_error_utils import initialize_logger
from tvb_epilepsy.io.edf import read_edf_to_Timeseries
from tvb_epilepsy.service.timeseries_service import TimeseriesService


logger = initialize_logger(__name__)


def prepare_signal_observable(data, on_off_set=[], rois=[], win_len_ratio=WIN_LEN_RATIO, filter_flag=True,
                              low_freq=LOW_FREQ, high_freq=HIGH_FREQ, log_flag=LOG_FLAG, plotter=False, title_prefix=""):
    ts_service = TimeseriesService()
    if len(on_off_set) == 0:
        on_off_set = [data.time_start, data.time_line[-1]]
    win_len = int(np.round(data.time_length / win_len_ratio))
    n_rois = len(rois)
    if n_rois > 0:
        if data.number_of_labels > n_rois:
            logger.info("Selecting signals...")
            if isinstance(rois[0], basestring):
                data = data.get_subspace_by_labels(rois)
            else:
                data = data.get_subspace_by_index(rois)
    if plotter:
        plotter.plot_spectral_analysis_raster(data.time_line, data.squeezed, time_units=data.time_unit,
                                              freq=np.array(range(1, 51, 1)), title='Spectral Analysis',
                                              figure_name=title_prefix + 'SpectralAnalysis',
                                              labels=data.space_labels, log_scale=True)
    if filter_flag:
        times = data.time_line
        fs = 1000.0 / np.mean(np.diff(times))
        high_freq = np.minimum(high_freq, 512.0)
        logger.info("Filtering signals...")
        data = ts_service.filter(data, fs, low_freq, high_freq, "bandpass", order=3)
        if plotter:
            plot_data = data.get_time_window(4*win_len, data.time_length-4*win_len).squeezed
            plot_time = data.time_line[4*win_len:-4*win_len]
            plotter.plot_raster({"Filtering": plot_data}, plot_time, time_units=data.time_unit,
                                special_idx=[], title='Filtered Time Series', offset=1.0,
                                figure_name=title_prefix + 'FilteredTimeSeries', labels=data.space_labels)
            plotter.plot_spectral_analysis_raster(plot_time, plot_data, time_units="ms", freq=np.array(range(1, 51, 1)),
                                                  title='Spectral Analysis_Filtered',
                                                  figure_name=title_prefix + 'SpectralAnalysisFiltered',
                                                  labels=data.space_labels, log_scale=True)
            del plot_data, plot_time

    def cut_fun(data, dtimes):
        times = data.time_line
        n_times = data.time_length
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
            t_offset = n_times - 1
        t_offset = np.minimum(int(np.maximum(t_offset, n_times - dtimes)), n_times - 1)
        return data.get_time_window(t_onset, t_offset)

    def xdecimate(data, decim_ratio, plotter):
        decim_ratio *= 2
        str_decim_ratio = str(decim_ratio)
        logger.info("Decimating signals...")
        logger.info("Decimation ratio: " + str_decim_ratio)
        data = ts_service.decimate(data, decim_ratio)

        if plotter:
            plotter.plot_raster({str_decim_ratio + "xDecimationX": data.squeezed}, data.time_line,
                                time_units=data.time_unit, special_idx=[],
                                title=str_decim_ratio + 'x Decimated Time Series', offset=0.1,
                                figure_name=title_prefix + str_decim_ratio + 'xDecimatedTimeSeries',
                                labels=data.space_labels)
        return data, decim_ratio

    def convolve(data, win_len, decim_ratio, plotter):
        logger.info("Convolving signals'power with square window...")
        if decim_ratio > 1:
            str_decim_ratio = str(decim_ratio) + "x"
        else:
            str_decim_ratio = ""
        win_len = np.maximum(3, np.int(np.floor(win_len / decim_ratio)))
        data = ts_service.convolve(data, win_len)
        if plotter:
            plotter.plot_raster({str_decim_ratio + "Convolution": data.squeezed}, data.time_line,
                                time_units=data.time_unit, special_idx=[],
                                title=str_decim_ratio + 'Convolved Time Series', offset=0.1,
                                figure_name=title_prefix + str_decim_ratio + 'ConvolvedTimeSeries', labels=data.space_labels)
        return data, win_len

    decim_ratio = 1
    data = ts_service.square(data)
    if data.time_length >= 1012:
        data = cut_fun(data, np.mod(data.time_length, 1012))
    data, win_len = convolve(data, win_len, decim_ratio, plotter)
    if data.time_length >= 1012:
        data = cut_fun(data, np.mod(data.time_length, 1012))
        while data.time_length >= 3.0 * 1012 and data.time_length > 2 * win_len:
            data, decim_ratio = xdecimate(data, decim_ratio, plotter)
            data, win_len = convolve(data, win_len, decim_ratio, plotter)
            if data.time_length >= 1012:
                data = cut_fun(data, np.mod(data.time_length, 1012))
    if data.time_length >= 1012:
        data = cut_fun(data, data.time_length - 1012)
    if log_flag:
        logger.info("Log of signals...")
        data = ts_service.log(data)
    logger.info("Normalizing signals...")
    data = ts_service.normalize(data, "zscore")
    if plotter:
        plotter.plot_raster({"ObservationRaster": data.squeezed}, data.time_line, time_units=data.time_unit,
                            special_idx=[], offset=1.0, title='Observation Raster Plot',
                            figure_name=title_prefix + 'ObservationRasterPlot', labels=data.space_labels)
        plotter.plot_timeseries({"Observation": data.squeezed}, data.time_line, time_units=data.time_unit,
                                special_idx=[], title='Observation Time Series',
                                figure_name=title_prefix + 'ObservationTimeSeries', labels=data.space_labels)
    return data


def prepare_seeg_observable(data, on_off_set=[], rois=[], win_len_ratio=WIN_LEN_RATIO, filter_flag=True, low_freq=LOW_FREQ,
                            high_freq=HIGH_FREQ, bipolar=BIPOLAR, log_flag=LOG_FLAG, plotter=False, title_prefix=""):
    if plotter:
        plotter.plot_spectral_analysis_raster(data.time_line, data.squeezed, time_units="ms",
                                              freq=np.array(range(1, 51, 1)), title='Spectral Analysis',
                                              figure_name=title_prefix + 'Spectral Analysis', labels=data.space_labels,
                                              log_scale=True)
    if bipolar:
        logger.info("Computing bipolar signals...")
        data = data.get_bipolar()
        if plotter:
            plotter.plot_raster({"Bipolar": data.squeezed}, data.time_line, time_units=data.time_unit,
                                special_idx=[], title='Bipolar Time Series', offset=0.1,
                                figure_name=title_prefix + 'BipolarTimeSeries', labels=data.space_labels)
            plotter.plot_spectral_analysis_raster(data.time_line, data.squeezed, time_units=data.time_unit,
                                                  freq=np.array(range(1, 51, 1)), title='Spectral Analysis',
                                                  figure_name=title_prefix + 'Spectral Analysis Bipolar',
                                                  labels=data.space_labels, log_scale=True)
    return prepare_signal_observable(data, on_off_set, rois, win_len_ratio=win_len_ratio, filter_flag=filter_flag,
                                     low_freq=low_freq, high_freq=high_freq, log_flag=log_flag, plotter=plotter,
                                     title_prefix=title_prefix)


def prepare_seeg_observable_from_mne_file(seeg_path, sensors, rois_selection, on_off_set,
                                          time_units="ms", label_strip_fun=None,
                                          win_len_ratio=WIN_LEN_RATIO, filter_flag=True, low_freq=LOW_FREQ,
                                          high_freq=HIGH_FREQ, bipolar=BIPOLAR, log_flag=LOG_FLAG, plotter=False,
                                          title_prefix=""):
    logger.info("Reading empirical dataset from edf file...")
    data = read_edf_to_Timeseries(seeg_path, sensors, rois_selection,
                                  label_strip_fun=label_strip_fun, time_units=time_units)
    data = TimeseriesService().detrend(data)
    if plotter:
        plotter.plot_raster({"Detrended": data.squeezed}, data.time_line, time_units=data.time_unit,
                            special_idx=[], title='Detrended Time Series', offset=0.1,
                            figure_name=title_prefix + 'DetrendedTimeSeries', labels=data.space_labels)
    return prepare_seeg_observable(data, on_off_set, rois=range(data.number_of_labels), win_len_ratio=win_len_ratio,
                                   filter_flag=filter_flag, low_freq=low_freq, high_freq=high_freq, bipolar=bipolar,
                                   log_flag=log_flag, plotter=plotter, title_prefix=title_prefix)
