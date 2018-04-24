import numpy as np
from tvb_epilepsy.base.constants.model_inversion_constants import WIN_LEN_RATIO, LOW_FREQ, HIGH_FREQ, BIPOLAR
from tvb_epilepsy.base.utils.log_error_utils import initialize_logger
from tvb_epilepsy.io.edf import read_edf_to_Timeseries
from tvb_epilepsy.service.timeseries_service import TimeseriesService


logger = initialize_logger(__name__)


def prepare_signal_observable(data, on_off_set=[], rois=[],  filter_flag=True, low_freq=LOW_FREQ, high_freq=HIGH_FREQ,
                              envelope_flag=True, smooth_flag=True, win_len_ratio=WIN_LEN_RATIO,
                              plotter=None, title_prefix=""):
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

    # First cut data close to the desired interval
    if len(on_off_set) == 0:
        on_off_set = [data.time_start, data.time_line[-1]]
    duration = on_off_set[1] - on_off_set[0]
    temp_on_off = [np.maximum(data.time_start, on_off_set[0] - 0.1 * duration),
                   np.minimum(data.time_line[-1], on_off_set[1] + 0.1 * duration)]
    data.get_time_window_by_units(temp_on_off[0], temp_on_off[1])

    # Now filter, if needed, before decimation introduces any artifacts
    if filter_flag:
        high_freq = np.minimum(high_freq, 512.0)
        logger.info("Filtering signals...")
        data = ts_service.filter(data, low_freq, high_freq, "bandpass", order=3)
        if plotter:
            plotter.plot_timeseries({"Filtering": data.squeezed}, data.time_line, time_units=data.time_unit,
                                special_idx=[], title='Filtered Time Series', # offset=1.0,
                                figure_name=title_prefix + 'FilteredTimeSeries', labels=data.space_labels)

    # Now decimate to get close to 1024 points
    temp_duration = temp_on_off[1] - temp_on_off[0]
    decim_ratio = int(np.round((data.time_length/1024.0) / (temp_duration/duration)))
    if decim_ratio > 1:
        str_decim_ratio = str(decim_ratio)
        logger.info("Decimating signals " + str_decim_ratio + " times...")
        data = ts_service.decimate(data, decim_ratio)
        if plotter:
            plotter.plot_timeseries({str_decim_ratio + " wise Decimation": data.squeezed}, data.time_line,
                                time_units=data.time_unit, special_idx=[],
                                title=str_decim_ratio + " wise Decimation", # offset=0.1,
                                figure_name=title_prefix + str_decim_ratio + 'xDecimatedTimeSeries',
                                labels=data.space_labels)

    # # Square data to get positive "power like" timeseries (introducing though higher frequencies)
    # data = ts_service.square(data)
    # Now get the signals' envelope via Hilbert transform
    if envelope_flag:
        data = ts_service.hilbert_envelope(data)
        if plotter:
            plotter.plot_timeseries({"Envelope": data.squeezed}, data.time_line,
                                    time_units=data.time_unit, special_idx=[],
                                    title='Envelope by Hilbert transform',  # offset=0.1,
                                    figure_name=title_prefix + 'EnvelopeTimeSeries',
                                    labels=data.space_labels)

    # Now convolve to smooth...
    if smooth_flag:
        win_len = int(np.round(1.0*data.time_length/win_len_ratio))
        str_win_len = str(win_len)
        data = ts_service.convolve(data, win_len)
        logger.info("Convolving signals with a square window of " + str_win_len + " points...")
        if plotter:
            plotter.plot_timeseries({"Smoothing": data.squeezed}, data.time_line,
                                    time_units=data.time_unit, special_idx=[],
                                    title='Convolved Time Series with a window of ' + str_win_len + " points",  # offset=0.1,
                                    figure_name=title_prefix + str_win_len + 'pointWinConvolvedTimeSeries',
                                    labels=data.space_labels)

    # Cut to the desired interval
    data = data.get_time_window_by_units(on_off_set[0], on_off_set[1])

    # Finally, normalize signals
    logger.info("Normalizing signals...")
    data = ts_service.normalize(data, "zscore")
    if plotter:
        plotter.plot_raster({"ObservationRaster": data.squeezed}, data.time_line, time_units=data.time_unit,
                            special_idx=[], offset=0.1, title='Observation Raster Plot',
                            figure_name=title_prefix + 'ObservationRasterPlot', labels=data.space_labels)
        plotter.plot_timeseries({"Observation": data.squeezed}, data.time_line, time_units=data.time_unit,
                                special_idx=[], title='Observation Time Series',
                                figure_name=title_prefix + 'ObservationTimeSeries', labels=data.space_labels)
    return data


def prepare_seeg_observable(data, on_off_set=[], rois=[], filter_flag=True, low_freq=LOW_FREQ, high_freq=HIGH_FREQ,
                            bipolar=BIPOLAR,  envelope_flag=True, smooth_flag=True, win_len_ratio=WIN_LEN_RATIO,
                            plotter=None, title_prefix=""):

    if bipolar:
        logger.info("Computing bipolar signals...")
        data = data.get_bipolar()
        if plotter:
            plotter.plot_raster({"Bipolar": data.squeezed}, data.time_line, time_units=data.time_unit,
                                special_idx=[], title='Bipolar Time Series', offset=0.1,
                                figure_name=title_prefix + 'BipolarTimeSeries', labels=data.space_labels)

    return prepare_signal_observable(data, on_off_set, rois, filter_flag=filter_flag, low_freq=low_freq,
                                     high_freq=high_freq,  envelope_flag=envelope_flag, smooth_flag=smooth_flag,
                                     win_len_ratio=win_len_ratio, plotter=plotter, title_prefix=title_prefix)

# win_len_ratio=WIN_LEN_RATIO,
def prepare_seeg_observable_from_mne_file(seeg_path, sensors, rois_selection, on_off_set,
                                          time_units="ms", label_strip_fun=None,
                                          filter_flag=True, low_freq=LOW_FREQ, high_freq=HIGH_FREQ,
                                          bipolar=BIPOLAR,  envelope_flag=True, smooth_flag=True,
                                          win_len_ratio=WIN_LEN_RATIO, plotter=None, title_prefix=""):
    logger.info("Reading empirical dataset from edf file...")
    data = read_edf_to_Timeseries(seeg_path, sensors, rois_selection,
                                  label_strip_fun=label_strip_fun, time_units=time_units)
    data = TimeseriesService().detrend(data)
    if plotter:
        plotter.plot_raster({"Detrended": data.squeezed}, data.time_line, time_units=data.time_unit,
                            special_idx=[], title='Detrended Time Series', offset=0.1,
                            figure_name=title_prefix + 'DetrendedTimeSeries', labels=data.space_labels)
    return prepare_seeg_observable(data, on_off_set, rois=range(data.number_of_labels),
                                   filter_flag=filter_flag, low_freq=low_freq, high_freq=high_freq,
                                   bipolar=bipolar,  envelope_flag=envelope_flag, smooth_flag=smooth_flag,
                                   win_len_ratio=win_len_ratio, plotter=plotter, title_prefix=title_prefix)
