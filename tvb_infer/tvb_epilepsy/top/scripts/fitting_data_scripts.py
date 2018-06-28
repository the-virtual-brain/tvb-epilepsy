import numpy as np
from tvb_infer.tvb_epilepsy.base.constants.model_inversion_constants import SEIZURE_LENGTH, HIGH_FREQ, LOW_FREQ, \
                                                                  WIN_LEN_RATIO, BIPOLAR, TARGET_DATA_PREPROCESSING
from tvb_infer.base.utils.log_error_utils import initialize_logger
from tvb_infer.base.utils.data_structures_utils import isequal_string
from tvb_infer.io.edf import read_edf_to_Timeseries
from tvb_infer.service.timeseries_service import TimeseriesService


logger = initialize_logger(__name__)


def prepare_signal_observable(data, seizure_length=SEIZURE_LENGTH, on_off_set=[], rois=[],
                              preprocessing=TARGET_DATA_PREPROCESSING, low_freq=LOW_FREQ, high_freq=HIGH_FREQ,
                              win_len_ratio=WIN_LEN_RATIO, plotter=None, title_prefix=""):
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
    temp_on_off = [np.maximum(data.time_start, on_off_set[0] - 2 * duration/win_len_ratio),
                   np.minimum(data.time_line[-1], on_off_set[1] + 2 * duration/win_len_ratio)]
    data = data.get_time_window_by_units(temp_on_off[0], temp_on_off[1])

    if plotter:
        plotter.plot_raster({"SelectedTimeInterval": data.squeezed}, data.time_line, time_units=data.time_unit,
                            special_idx=[], title='Selected time interval time series', offset=0.1,
                            figure_name=title_prefix + '_SelectedTimeSeries', labels=data.space_labels)

    for preproc in preprocessing:

        # Now filter, if needed, before decimation introduces any artifacts
        if isequal_string(preproc, "filter"):
            high_freq = np.minimum(high_freq, 512.0)
            logger.info("Filtering signals...")
            data = ts_service.filter(data, low_freq, high_freq, "bandpass", order=3)
            if plotter:
                plotter.plot_raster({"Filtering": data.squeezed}, data.time_line, time_units=data.time_unit,
                                        special_idx=[], title='Filtered Time Series',  offset=1.0,
                                        figure_name=title_prefix + '_FilteredTimeSeries', labels=data.space_labels)

        plot_envelope = ""
        if isequal_string(preproc, "square"):
            # Square data to get positive "power like" timeseries (introducing though higher frequencies)
            data = ts_service.square(data)
            plot_envelope = "Square envelope"
        elif isequal_string(preproc, "hilbert"):
            # or
            # ...get the signals' envelope via Hilbert transform
            data = ts_service.hilbert_envelope(data)
            plot_envelope = "Hilbert transform envelope"
            # or
        elif isequal_string(preproc, "abs"):
            #...the absolute value...
            data = ts_service.abs(data)
            plot_envelope = "abs envelope"
        if plot_envelope and plotter:
            plot_envelop_ = plot_envelope.replace(" ", "_")
            plotter.plot_raster({plot_envelop_: data.squeezed}, data.time_line,
                                time_units=data.time_unit, special_idx=[],
                                title=plot_envelope, offset=1.0,
                                figure_name=title_prefix + "_" + plot_envelop_ + "TimeSeries",
                                labels=data.space_labels)

        if isequal_string(preproc, "log"):
            logger.info("Computing log of signals...")
            data = TimeseriesService().log(data)
            if plotter:
                plotter.plot_raster({"LogTimeSeries": data.squeezed}, data.time_line, time_units=data.time_unit,
                                    special_idx=[], title='Log of Time Series', offset=0.1,
                                    figure_name=title_prefix + '_LogTimeSeries', labels=data.space_labels)

        # Now convolve to smooth...
        if isequal_string(preproc, "convolve"):
            win_len = int(np.round(1.0*data.time_length/win_len_ratio))
            str_win_len = str(win_len)
            data = ts_service.convolve(data, win_len)
            logger.info("Convolving signals with a square window of " + str_win_len + " points...")
            if plotter:
                plotter.plot_raster({"ConvolutionSmoothing": data.squeezed}, data.time_line,
                                    time_units=data.time_unit, special_idx=[], offset=0.1,
                                    title='Convolved Time Series with a window of ' + str_win_len + " points",
                                    figure_name=title_prefix + "_" + str_win_len + 'pointWinConvolvedTimeSeries',
                                    labels=data.space_labels)

    # Now decimate to get close to seizure_length points
    temp_duration = temp_on_off[1] - temp_on_off[0]
    decim_ratio = np.maximum(1, int(np.round((1.0*data.time_length/seizure_length) * (duration/temp_duration))))
    if decim_ratio > 1:
        str_decim_ratio = str(decim_ratio)
        logger.info("Decimating signals " + str_decim_ratio + " times...")
        data = ts_service.decimate(data, decim_ratio)
        if plotter:
            plotter.plot_raster({str_decim_ratio + " wise Decimation": data.squeezed}, data.time_line,
                                    time_units=data.time_unit, special_idx=[],
                                    title=str_decim_ratio + " wise Decimation", offset=0.1,
                                    figure_name=title_prefix + str_decim_ratio + 'xDecimatedTimeSeries',
                                    labels=data.space_labels)

    # Cut to the desired interval
    data = data.get_time_window_by_units(on_off_set[0], on_off_set[1])

    # Finally, normalize signals
    logger.info("Normalizing signals...")
    data = ts_service.normalize(data, "baseline-amplitude")  #  or "zscore"
    if plotter:
        plotter.plot_raster({"ObservationRaster": data.squeezed}, data.time_line, time_units=data.time_unit,
                            special_idx=[], offset=0.1, title='Observation Raster Plot',
                            figure_name=title_prefix + 'ObservationRasterPlot', labels=data.space_labels)
        plotter.plot_timeseries({"Observation": data.squeezed}, data.time_line, time_units=data.time_unit,
                                special_idx=[], title='Observation Time Series',
                                figure_name=title_prefix + 'ObservationTimeSeries', labels=data.space_labels)
    return data


def prepare_simulated_seeg_observable(data, sensor, seizure_length=SEIZURE_LENGTH, log_flag=True, on_off_set=[],
                                      rois=[], preprocessing=TARGET_DATA_PREPROCESSING,
                                      low_freq=LOW_FREQ, high_freq=HIGH_FREQ, bipolar=BIPOLAR,
                                      win_len_ratio=WIN_LEN_RATIO, plotter=None, title_prefix=""):

    logger.info("Computing SEEG signals...")
    data = TimeseriesService().compute_seeg(data, sensor, sum_mode=np.where(log_flag, "exp", "lin"))[0]
    if plotter:
        plotter.plot_raster({"SEEGData": data.squeezed}, data.time_line, time_units=data.time_unit,
                            special_idx=[], title='SEEG Time Series', offset=0.1,
                            figure_name=title_prefix + 'SEEGTimeSeries', labels=data.space_labels)
    if bipolar:
        logger.info("Computing bipolar signals...")
        data = data.get_bipolar()
        if plotter:
            plotter.plot_raster({"BipolarData": data.squeezed}, data.time_line, time_units=data.time_unit,
                                special_idx=[], title='Bipolar Time Series', offset=0.1,
                                figure_name=title_prefix + 'BipolarTimeSeries', labels=data.space_labels)
    return prepare_signal_observable(data, seizure_length, on_off_set, rois, preprocessing, low_freq, high_freq,
                                     win_len_ratio, plotter, title_prefix)


def prepare_seeg_observable_from_mne_file(seeg_path, sensors, rois_selection, seizure_length=SEIZURE_LENGTH,
                                          on_off_set=[], time_units="ms", label_strip_fun=None,
                                          preprocessing=TARGET_DATA_PREPROCESSING,
                                          low_freq=LOW_FREQ, high_freq=HIGH_FREQ, bipolar=BIPOLAR,
                                          win_len_ratio=WIN_LEN_RATIO, plotter=None, title_prefix=""):
    logger.info("Reading empirical dataset from edf file...")
    data = read_edf_to_Timeseries(seeg_path, sensors, rois_selection,
                                  label_strip_fun=label_strip_fun, time_units=time_units)
    if plotter:
        plotter.plot_raster({"OriginalData": data.squeezed}, data.time_line, time_units=data.time_unit,
                            special_idx=[], title='Original Empirical Time Series', offset=1.0,
                            figure_name=title_prefix + 'OriginalTimeSeries', labels=data.space_labels)
    data = TimeseriesService().detrend(data)
    if plotter:
        plotter.plot_raster({"Detrended": data.squeezed}, data.time_line, time_units=data.time_unit,
                            special_idx=[], title='Detrended Time Series', offset=1.0,
                            figure_name=title_prefix + 'DetrendedTimeSeries', labels=data.space_labels)
    if bipolar:
        logger.info("Computing bipolar signals...")
        data = data.get_bipolar()
        if plotter:
            plotter.plot_raster({"BipolarData": data.squeezed}, data.time_line, time_units=data.time_unit,
                                special_idx=[], title='Bipolar Time Series', offset=0.1,
                                figure_name=title_prefix + 'BipolarTimeSeries', labels=data.space_labels)

    return prepare_signal_observable(data, seizure_length, on_off_set, range(data.number_of_labels),
                                     preprocessing, low_freq, high_freq, win_len_ratio, plotter, title_prefix)

