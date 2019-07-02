
import os

import numpy as np

from tvb_fit.base.config import Config
from tvb_fit.base.utils.log_error_utils import initialize_logger, raise_value_error, warning
from tvb_fit.service.timeseries_service import TimeseriesService
from tvb_fit.io.h5_reader import H5Reader
from tvb_fit.io.h5_writer import H5Writer
from tvb_fit.plot.plotter import Plotter


class FittingDataPreparation(object):

    _ts_service = None
    _config = None
    _logger = None
    _reader = None
    _writer = None
    _plotter = None

    _res_folder = ""
    _figs_folder = ""
    _title_prefix = ""

    _sensors = None
    _include_sensors_labels = []
    _exclude_sensors_labels = []

    _workflow_sequence = []
    _time_length = 0.0
    _time_lims = [0.0, 0.0]
    _filter_props = {"low_hpf": 10.0, "high_hpf": 256.0, "low_lpf": 1.0, "high_lpf": 10.0, "order": 3}
    _convolution_win_len = 1000.0
    _decim_ratio = 1
    _bipolar_flag = False

    def __init__(self, ts_service=TimeseriesService(), config=Config(),
                 reader=H5Reader(), writer=H5Writer(), logger=None, plotter=None):
        self._config = config
        self._reader = reader
        self._writer = writer
        if logger is None:
            self._logger = initialize_logger(__name__, config.out.FOLDER_LOGS)
        else:
            self._logger = logger
        self._ts_service = ts_service
        self._ts_service.logger = self._logger
        if plotter is None:
            self._plotter = Plotter(self._config)

    def set_attr(self, attr, value):
        try:
            getattr(self, attr)
            setattr(self, attr, value)
        except:
            try:
                getattr(self, "_"+attr)
                setattr(self, "_"+attr, value)
            except:
                raise_value_error("Failed to set attribute %s of object %s to value %s"%
                                  (str(attr), str(self), str(value)))
        return self

    def set_attributes(self, attrs, values):
        for attr, val, in zip(attrs, values):
            try:
                self.set_attr(attr, val)
            except:
                warning("Failed to set attribute %s of object %s to value %s" % (str(attr), str(self), str(val)))

    @property
    def res_folder(self):
        if os.path.isdir(self._res_folder):
            self._res_folder = self._config.out.FOLDER_RES
        return self._res_folder

    @property
    def figs_folder(self):
        if os.path.isdir(self._figs_folder):
            self._figs_folder = self._config.out.FOLDER_FIGURES
        return self._figs_folder

    @property
    def title_prefix(self):
        if len(self._title_prefix):
            return self._title_prefix + "_"
        else:
            return self._title_prefix

    def _hpf(self, signals):
        return self._ts_service.filter(signals, self._filter_props["low_hpf"], self._filter_props["high_hpf"],
                                       "bandpass", order=self._filter_props["order"])

    def _lpf(self, signals):
        return self._ts_service.filter(signals, self._filter_props["low_lpf"], self._filter_props["high_lpf"],
                                       "bandpass", order=self._filter_props["order"])

    def _decimate(self, signals):
        return self._ts_service.decimate(signals, self._decim_ratio)

    def _convolve(self, signals)
        return self._ts_service.convolve(signals, self._convolution_win_len)

    def _log(self, signals):
        return self._ts_service.log(signals)

    def _hilbert_envelope(self, signals):
        return self._ts_service.hilbert_envelope(signals)

    def _abs_envelope(self, signals):
        return self._ts_service.abs_envelope(signals)

    def _spectrogram(self, signals):
        return self._ts_service.spectrogram_envelope(signals, self._filter_props["high_hpf"],
                                                     self._filter_props["low_hpf"], self._decim_ratio)

    def _time_select(self, signals, time_lims):
        return signals.get_time_window_by_units(time_lims[0], time_lims[1])

    def _plot_raster(self, signals, title, offset=0.1):
        name = title.replace(" ", "_")
        self._plotter.plot_raster({name: signals.squeezed}, signals.time, time_units=signals.time_unit,
                                  title=title, offset=offset, figure_name = self.title_prefix + name,
                                  labels=signals.space_labels)

    def _plot_ts(self, signals, title):
        name = title.replace(" ", "_")
        self._plotter.plot_timeseries({name: signals.squeezed}, signals.time, time_units=signals.time_unit,
                                       title=title, figure_name = self.title_prefix + name, labels=signals.space_labels)

def label_strip_fun(label):
    pass


def concatenate_signals(self, signals):
    # Concatenate only the labels that exist in all signals:
    labels = signals[0].space_labels
    for signal in signals[1:]:
        labels = np.intersect1d(labels, signal.space_labels)
    signals = TimeseriesService().concatenate_in_time(signals, labels)
    return signals

