import numpy as np
from tvb_fit.tvb_epilepsy.base.constants.model_inversion_constants import \
    SEIZURE_LENGTH, HIGH_HPF, LOW_HPF, LOW_LPF, HIGH_LPF, WIN_LEN_RATIO, BIPOLAR, TARGET_DATA_PREPROCESSING
from tvb_fit.base.utils.log_error_utils import initialize_logger
from tvb_fit.base.utils.data_structures_utils import isequal_string, find_labels_inds
from tvb_fit.io.edf import read_edf_to_Timeseries
from tvb_fit.service.timeseries_service import TimeseriesService, NORMALIZATION_METHODS


logger = initialize_logger(__name__)


def prepare_signal_observable(data, seizure_length=SEIZURE_LENGTH, on_off_set=[], rois=[],
                              preprocessing=TARGET_DATA_PREPROCESSING, low_hpf=LOW_HPF, high_hpf=HIGH_HPF,
                              low_lpf=LOW_LPF, high_lpf=HIGH_LPF, win_len_ratio=WIN_LEN_RATIO,
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
                            figure_name=title_prefix + '_SelectedTimeSeriesRaster', labels=data.space_labels)
        plotter.plot_timeseries({"SelectedTimeInterval": data.squeezed}, data.time_line, time_units=data.time_unit,
                                special_idx=[], title='Selected time interval time series',
                                figure_name=title_prefix + '_SelectedTimeSeries', labels=data.space_labels)

    for i_preproc, preproc in enumerate(preprocessing):

        stri_preproc = str(i_preproc+1)

        # Now filter, if needed, before decimation introduces any artifacts
        if isequal_string(preproc, "hpf"):
            high_hpf = np.minimum(high_hpf, 256.0)
            logger.info("High-pass filtering signals...")
            data = ts_service.filter(data, low_hpf, high_hpf, "bandpass", order=3)
            if plotter:
                plotter.plot_raster({"High-pass filtering": data.squeezed}, data.time_line, time_units=data.time_unit,
                                    special_idx=[], title='High-pass filtered Time Series',  offset=1.0,
                                    figure_name=title_prefix + '_%sHpfTimeSeries' % stri_preproc,
                                    labels=data.space_labels)

        if isequal_string(preproc, "invert"):
            logger.info("Invert signals' sign...")
            data.data = - data.data
            if plotter:
                plotter.plot_raster({"Sign inverted signals": data.squeezed}, data.time_line, time_units=data.time_unit,
                                    special_idx=[], title='Sign inverted Time Series',  offset=1.0,
                                    figure_name=title_prefix + '_%sSignInversion' % stri_preproc,
                                    labels=data.space_labels)

        plot_rectify = ""
        if isequal_string(preproc, "hilbert"):
            # or
            # ...get the signals' envelope via Hilbert transform
            data = ts_service.hilbert_envelope(data)
            plot_rectify = "Hilbert transform amplitude"
            # or
        # elif isequal_string(preproc, "square"):
        #     # Square data to get positive "power like" timeseries (introducing though higher frequencies)
        #     data = ts_service.square(data)
        elif isequal_string(preproc, "abs") or isequal_string(preproc, "abs-mean"):
            if isequal_string(preproc, "abs-mean"):
                data = ts_service.normalize(data, "mean")
            #...the absolute value...
            data = ts_service.abs(data)
        if plotter:
            plot_envelop_ = plot_rectify.replace(" ", "_")
            plotter.plot_raster({plot_envelop_: data.squeezed}, data.time_line,
                                time_units=data.time_unit, special_idx=[],
                                title=plot_rectify, offset=1.0,
                                figure_name=title_prefix + "_%s" % stri_preproc + plot_envelop_ + "TimeSeries",
                                labels=data.space_labels)

        if isequal_string(preproc, "log"):
            logger.info("Computing log of signals...")
            data = TimeseriesService().log(data)
            if plotter:
                plotter.plot_raster({"LogTimeSeries": data.squeezed}, data.time_line, time_units=data.time_unit,
                                    special_idx=[], title='Log of Time Series', offset=0.1,
                                    figure_name=title_prefix + '_%sLogTimeSeries' % stri_preproc,
                                    labels=data.space_labels)

        # Now convolve or low pass filter to smooth...
        if isequal_string(preproc, "convolve"):
            win_len = int(np.round(1.0*data.time_length/win_len_ratio))
            str_win_len = str(win_len)
            data = ts_service.convolve(data, win_len)
            logger.info("Convolving signals with a square window of " + str_win_len + " points...")
            if plotter:
                plotter.plot_raster({"ConvolutionSmoothing": data.squeezed}, data.time_line,
                                    time_units=data.time_unit, special_idx=[], offset=1,
                                    title='Convolved Time Series with a window of ' + str_win_len + " points",
                                    figure_name=
                                       title_prefix + '_%s_%spointWinConvolvedTimeSeries' % (stri_preproc, str_win_len),
                                    labels=data.space_labels)

        elif isequal_string(preproc, "lpf"):
            high_lpf = np.minimum(high_lpf, 512.0)
            logger.info("Low-pass filtering signals...")
            data = ts_service.filter(data, low_lpf, high_lpf, "bandpass", order=3)
            if plotter:
                plotter.plot_raster({"Low-pass filtering": data.squeezed}, data.time_line, time_units=data.time_unit,
                                    special_idx=[], title='Low-pass filtered Time Series',  offset=1.0,
                                    figure_name=title_prefix + '_%sLpfTimeSeries' % stri_preproc,
                                    labels=data.space_labels)

    if "decimate" in preprocessing:
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
                                        figure_name=
                                          title_prefix + "_%s_%sxDecimatedTimeSeries" % (stri_preproc, str_decim_ratio),
                                        labels=data.space_labels)

    # # Cut to the desired interval
    data = data.get_time_window_by_units(on_off_set[0], on_off_set[1])

    for preproc in preprocessing:
        if preproc in NORMALIZATION_METHODS:
            # Finally, normalize signals
            logger.info("Normalizing signals...")
            data = ts_service.normalize(data, preproc)  # "baseline", "baseline-std", "baseline-amplitude" or "zscore
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
                                      low_hpf=LOW_HPF, high_hpf=HIGH_HPF, low_lpf=LOW_LPF, high_lpf=HIGH_LPF,
                                      bipolar=BIPOLAR, win_len_ratio=WIN_LEN_RATIO, plotter=None, title_prefix=""):

    logger.info("Computing SEEG signals...")
    data = TimeseriesService().compute_seeg(data, sensor, sum_mode=np.where(log_flag, "exp", "lin"))
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
    return prepare_signal_observable(data, seizure_length, on_off_set, rois, preprocessing, low_hpf, high_hpf,
                                     low_lpf, high_lpf, win_len_ratio, plotter, title_prefix)


def prepare_seeg_observable_from_mne_file(seeg_path, sensors, rois_selection, seizure_length=SEIZURE_LENGTH,
                                          on_off_set=[], time_units="ms", label_strip_fun=None,
                                          preprocessing=TARGET_DATA_PREPROCESSING,
                                          low_hpf=LOW_HPF, high_hpf=HIGH_HPF, low_lpf=LOW_LPF, high_lpf=HIGH_LPF,
                                          bipolar=BIPOLAR, win_len_ratio=WIN_LEN_RATIO, plotter=None, title_prefix=""):
    logger.info("Reading empirical dataset from edf file...")
    data = read_edf_to_Timeseries(seeg_path, sensors, rois_selection,
                                  label_strip_fun=label_strip_fun, time_units=time_units)
    data.data = np.array(data.data).astype("float32")
    if plotter:
        plotter.plot_raster({"OriginalData": data.squeezed}, data.time_line, time_units=data.time_unit,
                            special_idx=[], title='Original Empirical Time Series', offset=1.0,
                            figure_name=title_prefix + '_empirical_OriginalTimeSeries', labels=data.space_labels)
    data = TimeseriesService().detrend(data)
    if plotter:
        plotter.plot_raster({"Detrended": data.squeezed}, data.time_line, time_units=data.time_unit,
                            special_idx=[], title='Detrended Time Series', offset=1.0,
                            figure_name=title_prefix + '_empirical_DetrendedTimeSeries', labels=data.space_labels)
    if bipolar:
        logger.info("Computing bipolar signals...")
        data = data.get_bipolar()
        if plotter:
            plotter.plot_raster({"BipolarData": data.squeezed}, data.time_line, time_units=data.time_unit,
                                special_idx=[], title='Bipolar Time Series', offset=0.1,
                                figure_name=title_prefix + 'BipolarTimeSeries', labels=data.space_labels)

    return prepare_signal_observable(data, seizure_length, on_off_set, range(data.number_of_labels),
                                     preprocessing, low_hpf, high_hpf, low_lpf, high_lpf,
                                     win_len_ratio, plotter, title_prefix)

