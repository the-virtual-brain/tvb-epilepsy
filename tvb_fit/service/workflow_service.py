from abc import ABCMeta  # abstractmethod,

import os

import numpy as np

from tvb_fit.base.config import Config

from tvb_fit.io.h5_reader import H5Reader
from tvb_fit.io.h5_writer import H5Writer
from tvb_fit.plot.plotter import Plotter

from tvb_scripts.utils.log_error_utils import initialize_logger, raise_value_error, warning
from tvb_scripts.utils.file_utils import wildcardit
from tvb_scripts.model.virtual_head.head import Head
from tvb_scripts.io.tvb_data_reader import TVBReader


class WorkflowService(object):
    __metaclass__ = ABCMeta

    def __init__(self, config=Config(), reader=None, writer=None, plotter=None):
        if isinstance(config, Config):
            self._config = config
        else:
            self._config = Config()
        self._logger = initialize_logger(__name__, config.out.FOLDER_LOGS)
        if isinstance(reader, (H5Reader, TVBReader)):
            self._reader = reader
        else:
            if self._config.input.IS_TVB_MODE:
                self._reader = TVBReader()
            else:
                self._reader = H5Reader()
        if isinstance(writer, H5Writer):
            self._writer = writer
        else:
            self._writer = H5Writer()
        if isinstance(plotter, Plotter):
            self._plotter = plotter
        else:
            self._plotter = Plotter(self._config)
        self._head = None
        self._number_of_regions = 0
        self._random_seed = 0
        self._write_output_files = True
        self._plot_output = True
        self._head_folder = ""
        self._res_folder = ""
        self._figs_folder = ""

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

    def _get_foldername(self, path):
        return os.path.dirname(path).split(os.sep)[-1]

    def _ensure_folder(self, folderpath):
        if not os.path.isdir(folderpath):
            os.makedirs(folderpath)

    def _plot_flag(self, plot_flag=None):
        return self._plot_output * plot_flag

    def _write_flag(self, write_flag=True):
        return self._write_output_files * write_flag

    def read_head(self, plot_head=True):
        if self._plot_flag(plot_head):
            plotter = self._plotter
        else:
            plotter = None
        self._head = read_head(self._reader, self._config, self._plotter, self._logger)

    def _add_prefix(self, name, prefix):
        if len(prefix) > 0:
            return "_".join([prefix, name])
        else:
            return name

    def _wildcardit(self, name, front=True, back=True):
        return wildcardit(name, front, back)

    @property
    def head_folder(self):
        if not os.path.isdir(self._head_folder):
            self._head_folder = self._config.input.HEAD
        return self._head_folder

    @property
    def res_folder(self):
        if len(self._res_folder) == 0:
            self._res_folder = self._config.out.FOLDER_RES
        return self._res_folder

    @property
    def figs_folder(self):
        if len(self._figs_folder) == 0:
            self._figs_folder = self._config.out.FOLDER_FIGURES
        return self._figs_folder

    @property
    def head(self):
        if not (isinstance(self._head, Head)):
            self.read_head(False)
        return self._head

    def _assert_number_of_regions(self):
        if self._number_of_regions < 1:
            self._number_of_regions = self.head.number_of_regions
        else:
            assert self._number_of_regions == self.head.number_of_regions

    @property
    def number_of_regions(self):
        self._assert_number_of_regions()
        return self._number_of_regions

    @property
    def all_regions_indices(self):
        return np.arange(self.number_of_regions)

    @property
    def connectivity(self):
        return self.head.connectivity

    @property
    def normalized_weights(self):
        return self.connectivity.normalized_weights

    @property
    def region_labels(self):
        return self.connectivity.region_labels


def read_head(reader, config, plotter=None, logger=None):
    # -------------------------------Reading model_data-----------------------------------
    if logger:
        logger.info("Reading from: " + config.input.HEAD)
    head = reader.read_head(config.input.HEAD)
    if plotter:
        plotter.plot_head(head)
    return head
